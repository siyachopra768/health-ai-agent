import json
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from .embed_guidelines import EmbedConfig, GuidelineEmbedder

BASE_DIR = Path(__file__).resolve().parent.parent
CHUNK_DIR = BASE_DIR / "data/chunks"
BM25_CACHE_PATH = BASE_DIR / "data/bm25_index.pkl"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


@dataclass
class HybridConfig:
    embed_config: EmbedConfig = field(default_factory=EmbedConfig)
    reranker_model_name: str = RERANKER_MODEL
    bm25_cache_path: Path = BM25_CACHE_PATH
    rrf_k: int = 60


class BM25IndexManager:
    def __init__(self, chunk_dir: Path, cache_path: Path):
        self.chunk_dir = chunk_dir
        self.cache_path = cache_path
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: List[Dict[str, Any]] = []

    def _tokenize(self, text: str) -> List[str]:
        return text.lower().split()

    def load_or_build(self, force_rebuild: bool = False) -> None:
        if not force_rebuild and self.cache_path.exists():
            logger.info("Loading cached BM25 index from %s", self.cache_path)
            with open(self.cache_path, "rb") as f:
                data = pickle.load(f)
                self.bm25 = data["bm25"]
                self.chunks = data["chunks"]
            return

        logger.info("Building BM25 index from chunks in %s...", self.chunk_dir)
        all_chunks = []
        for file_path in self.chunk_dir.glob("*_chunks.json"):
            if file_path.name == "chunking_manifest.json":
                continue
            try:
                content = json.loads(file_path.read_text(encoding="utf-8"))
                all_chunks.extend(content)
            except Exception as e:
                logger.error("Failed loading %s: %s", file_path, e)

        if not all_chunks:
            raise RuntimeError(f"No chunk files found in {self.chunk_dir}")

        corpus = [self._tokenize(c["content"]) for c in all_chunks]
        self.bm25 = BM25Okapi(corpus)
        self.chunks = all_chunks

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunks": self.chunks}, f)
        logger.info("BM25 index cached successfully (%d chunks).", len(all_chunks))

    def search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        if self.bm25 is None or not self.chunks:
            raise RuntimeError("BM25 index is not initialized. Call load_or_build() first.")

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [self.chunks[i] for i in top_indices if scores[i] > 0]


class HybridRetriever:
    def __init__(self, config: Optional[HybridConfig] = None):
        self.config = config or HybridConfig()
        self.embedder = GuidelineEmbedder(self.config.embed_config)
        self.bm25_manager = BM25IndexManager(
            chunk_dir=self.config.embed_config.chunk_dir,
            cache_path=self.config.bm25_cache_path,
        )
        self._reranker: Optional[CrossEncoder] = None
        self._bm25_initialized: bool = False

    @property
    def reranker(self) -> CrossEncoder:
        if self._reranker is None:
            logger.info("Loading Reranker Model: %s", self.config.reranker_model_name)
            self._reranker = CrossEncoder(self.config.reranker_model_name)
        return self._reranker

    def initialize(self, force_rebuild_bm25: bool = False) -> None:
        if not self._bm25_initialized:
            self.bm25_manager.load_or_build(force_rebuild=force_rebuild_bm25)
            self._bm25_initialized = True

    def _dense_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        query_emb = self.embedder.model.encode([query]).tolist()
        results = self.embedder.collection.query(
            query_embeddings=query_emb,
            n_results=top_k,
            include=["documents", "metadatas"],
        )
        candidates = []
        if results["ids"] and results["ids"][0]:
            for cid, doc, meta in zip(results["ids"][0], results["documents"][0], results["metadatas"][0]):
                candidates.append({"chunk_id": cid, "content": doc, "metadata": meta})
        return candidates

    def _reciprocal_rank_fusion(self, dense_results, sparse_results):
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Dict[str, Any]] = {}
        for rank, chunk in enumerate(dense_results):
            cid = chunk["chunk_id"]
            chunk_map[cid] = chunk
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.config.rrf_k + rank + 1))
        for rank, chunk in enumerate(sparse_results):
            cid = chunk["chunk_id"]
            chunk_map[cid] = chunk
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.config.rrf_k + rank + 1))
        sorted_cids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        return [chunk_map[cid] for cid in sorted_cids]

    def _rerank(self, query, candidates, top_k):
        if not candidates:
            return []
        pairs = [[query, c["content"]] for c in candidates]
        scores = self.reranker.predict(pairs)
        for chunk, score in zip(candidates, scores):
            chunk["rerank_score"] = float(score)
        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

    def search(self, query: str, top_k: int = 5, fetch_k: int = 20, enable_reranking: bool = True):
        if not query.strip():
            return []

        # Self-initializing: callers shouldn't have to remember a separate
        # setup step before search() works.
        self.initialize()

        try:
            dense_candidates = self._dense_search(query, top_k=fetch_k)
        except Exception as e:
            logger.error("Dense search failed, continuing with sparse only: %s", e)
            dense_candidates = []

        try:
            sparse_candidates = self.bm25_manager.search(query, top_k=fetch_k)
        except Exception as e:
            logger.error("Sparse search failed, continuing with dense only: %s", e)
            sparse_candidates = []

        if not dense_candidates and not sparse_candidates:
            logger.warning("Both retrieval branches returned nothing for query: %r", query)
            return []

        fused_candidates = self._reciprocal_rank_fusion(dense_candidates, sparse_candidates)
        if not fused_candidates:
            return []
        if enable_reranking:
            try:
                candidate_pool = fused_candidates[: fetch_k * 2]
                return self._rerank(query, candidate_pool, top_k=top_k)
            except Exception as e:
                logger.error("Reranking failed, falling back to RRF order: %s", e)
                return fused_candidates[:top_k]
        return fused_candidates[:top_k]