import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

CHUNK_DIR = Path("data/chunks")
DB_DIR = Path("data/chroma_db")
DB_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dim, f
#ast, good quality
BATCH_SIZE = 64
COLLECTION_NAME = "medical_guidelines"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
,
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

@dataclass
class EmbedConfig:
    model_name: str = MODEL_NAME
    batch_size: int = BATCH_SIZE
    collection_name: str = COLLECTION_NAME
    persist_dir: Path = DB_DIR
    chunk_dir: Path = CHUNK_DIR


class GuidelineEmbedder:
    """Embeds guideline chunks and stores in ChromaDB."""

    def __init__(self, config: Optional[EmbedConfig] = None):
        self.config = config or EmbedConfig()
        self._model: Optional[SentenceTransformer] = None
        self._client: Optional[chromadb.Client] = None
        self._collection: Optional[chromadb.Collection] = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading embedding model: %s", self.config.model_name)
            self._model = SentenceTransformer(self.config.model_name)
        return self._model

    @property
    def client(self) -> chromadb.Client:
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=str(self.config.persist_dir),
                settings=Settings(anonymized_telemetry=False),
            )
        return self._client

    @property
    def collection(self) -> chromadb.Collection:
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def _load_chunks(self, guideline_id: str) -> List[Dict[str, Any
]]:
        path = self.config.chunk_dir / f"{guideline_id}_chunks.json"
        return json.loads(path.read_text())

    def _prepare_batch(self, chunks: List[Dict[str, Any]]) -> tuple:
        ids = [c["chunk_id"] for c in chunks]
        documents = [c["content"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        return ids, documents, metadatas

    def embed_guideline(self, guideline_id: str) -> Dict[str, Any]:
        """Embed all chunks for a single guideline."""
        logger.info("Embedding: %s", guideline_id)

        chunks = self._load_chunks(guideline_id)
        if not chunks:
            return {"guideline_id": guideline_id, "status": "error"
, "error": "No chunks found"}

        total = len(chunks)
        embedded = 0

        for i in range(0, total, self.config.batch_size):
            batch = chunks[i:i + self.config.batch_size]
            ids, docs, metas = self._prepare_batch(batch)

            embeddings = self.model.encode(docs, show_progress_bar=
False).tolist()

            self.collection.add(
                ids=ids,
                documents=docs,
                metadatas=metas,
                embeddings=embeddings,
            )
            embedded += len(batch)

        logger.info("  Embedded %d/%d chunks", embedded, total)
        return {"guideline_id": guideline_id, "status": "success",
"chunks_embedded": embedded}

    def embed_all(self) -> Dict[str, Any]:
        """Embed all guidelines from chunk directory."""
        chunk_files = list(self.config.chunk_dir.glob("*_chunks.json"))
        guideline_ids = [f.stem.replace("_chunks", "") for f in chunk_files]

        logger.info("Found %d guidelines to embed", len(guideline_ids))

        results = []
        for gid in tqdm(guideline_ids, desc="Embedding guidelines"):
            results.append(self.embed_guideline(gid))

        successful = [r for r in results if r["status"] == "success"]
        total_chunks = sum(r.get("chunks_embedded", 0) for r in successful)

        logger.info("=" * 50)
        logger.info("EMBEDDING COMPLETE: %d/%d successful, %d total chunks",
                    len(successful), len(guideline_ids), total_chunks)

        return {
            "total_guidelines": len(guideline_ids),
            "successful": len(successful),
            "total_chunks": total_chunks,
            "modell": self.config.model_name,
            "collection": self.config.collection_name,
            "results": results,
        }


def main():
    config = EmbedConfig()
    embedder = GuidelineEmbedder(config)
    embedder.embed_all()


if __name__ == "__main__":
    main()