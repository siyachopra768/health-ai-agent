"""
eval_rag.py — Evaluation harness for the hybrid RAG retriever.

Tests retrieval quality of the BM25 + dense + cross-encoder reranker pipeline
against hand-labeled relevant chunks.

Metrics:
  - Precision@k   = relevant_in_top_k / k
  - Recall@k      = relevant_in_top_k / total_relevant
  - MRR@k         = mean reciprocal rank of first relevant chunk
  - Coverage      = queries_returning >= 1 result / total
  - Recall by Stage: dense-only vs sparse-only vs hybrid-reranked

All tests run against the real chunk data in data/chunks/.
The dense retriever (sentence-transformers) and reranker (cross-encoder)
are loaded on first use — may take a few seconds to warm up.

Run:  python -m evals.eval_rag
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import pytest

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Paths
# --------------------------------------------------------------------------- #
DATA_FILE = Path(__file__).resolve().parent / "eval_data" / "rag_queries.json"


# --------------------------------------------------------------------------- #
#  Data classes
# --------------------------------------------------------------------------- #

@dataclass
class RagCaseResult:
    case_id: str
    query: str
    relevant_chunks: list[str]
    relevant_guideline: str

    # Results
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    retrieved_scores: list[float] = field(default_factory=list)
    latency_ms: float = 0.0

    # Per-k metrics
    precision_at_k: dict[int, float] = field(default_factory=dict)
    recall_at_k: dict[int, float] = field(default_factory=dict)
    mrr_at_k: dict[int, float] = field(default_factory=dict)  # per-case, just reciprocal rank

    # Stage comparison
    dense_chunk_ids: list[str] = field(default_factory=list)
    sparse_chunk_ids: list[str] = field(default_factory=list)

    # Status
    passed: bool = False
    error: str | None = None


@dataclass
class RagEvalMetrics:
    total_queries: int = 0
    precision_at_3: float = 0.0
    recall_at_3: float = 0.0
    mrr_at_3: float = 0.0
    coverage_at_3: float = 0.0
    avg_latency_ms: float = 0.0
    per_query: list[dict] = field(default_factory=list)
    stage_comparison: dict[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def _load_cases() -> list[dict]:
    data = json.loads(DATA_FILE.read_text())
    return data["cases"]


def _precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Precision@k: fraction of top-k retrieved that are relevant."""
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for cid in top_k if cid in relevant)
    return hits / len(top_k)


def _recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Recall@k: fraction of relevant chunks that appear in top-k."""
    if not relevant:
        return 1.0  # no relevant chunks to find
    top_k = retrieved[:k]
    hits = sum(1 for cid in top_k if cid in relevant)
    return hits / len(relevant)


def _mrr_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """MRR@k: reciprocal rank of first relevant chunk, or 0 if none in top-k."""
    top_k = retrieved[:k]
    for rank, cid in enumerate(top_k, start=1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0


def _get_all_chunks() -> dict[str, dict]:
    """Load all chunks from all chunk files into a dict keyed by chunk_id."""
    chunk_dir = Path(__file__).resolve().parent.parent / "data" / "chunks"
    all_chunks = {}
    for f in chunk_dir.glob("*_chunks.json"):
        if f.name == "chunking_manifest.json":
            continue
        data = json.loads(f.read_text())
        for chunk in data:
            all_chunks[chunk["chunk_id"]] = chunk
    return all_chunks


def _evaluate_case(
    case: dict,
    retriever: Any,
    all_chunks: dict[str, dict],
    k: int = 3,
) -> RagCaseResult:
    """Run a single RAG retrieval eval case."""
    from rag.hybrid_retrieval import HybridRetriever

    relevant_set = set(case["relevant_chunks"])
    result = RagCaseResult(
        case_id=case["id"],
        query=case["query"],
        relevant_chunks=case["relevant_chunks"],
        relevant_guideline=case["relevant_guideline"],
    )

    start = time.perf_counter()

    try:
        # Full hybrid search (with reranking)
        results = retriever.search(case["query"], top_k=k, fetch_k=20, enable_reranking=True)
        result.retrieved_chunk_ids = [r["chunk_id"] for r in results]
        result.retrieved_scores = [r.get("rerank_score", 0.0) for r in results]
        result.latency_ms = (time.perf_counter() - start) * 1000

        # Also get dense-only and sparse-only results for stage comparison
        dense_results = retriever._dense_search(case["query"], top_k=20)
        result.dense_chunk_ids = [r["chunk_id"] for r in dense_results[:k]]

        sparse_results = retriever.bm25_manager.search(case["query"], top_k=20)
        result.sparse_chunk_ids = [r["chunk_id"] for r in sparse_results[:k]]

        # Compute per-k metrics
        for kk in [1, 3, 5]:
            result.precision_at_k[kk] = _precision_at_k(result.retrieved_chunk_ids, relevant_set, kk)
            result.recall_at_k[kk] = _recall_at_k(result.retrieved_chunk_ids, relevant_set, kk)
            result.mrr_at_k[kk] = _mrr_at_k(result.retrieved_chunk_ids, relevant_set, kk)

        # A case "passes" if the first relevant chunk appears in top-3
        result.passed = result.mrr_at_k[3] > 0.0

    except Exception as e:
        result.error = str(e)
        result.latency_ms = (time.perf_counter() - start) * 1000
        logger.exception("Error evaluating RAG case %s", case["id"])

    return result


# --------------------------------------------------------------------------- #
#  Aggregation
# --------------------------------------------------------------------------- #

def compute_rag_metrics(results: list[RagCaseResult]) -> dict:
    """Aggregate per-case RAG results into summary metrics."""
    total = len(results)
    valid = [r for r in results if r.error is None]
    errors = [r for r in results if r.error is not None]

    k = 3
    avg_p3 = sum(r.precision_at_k.get(k, 0) for r in valid) / len(valid) if valid else 0.0
    avg_r3 = sum(r.recall_at_k.get(k, 0) for r in valid) / len(valid) if valid else 0.0
    avg_mrr3 = sum(r.mrr_at_k.get(k, 0) for r in valid) / len(valid) if valid else 0.0
    coverage = len(valid) / total if total else 0.0
    avg_latency = sum(r.latency_ms for r in valid) / len(valid) if valid else 0.0

    # Stage comparison: precision@3 and recall@3 for each retrieval stage
    stage_comparison = {}
    for stage_name, stage_field in [
        ("dense_only", "dense_chunk_ids"),
        ("sparse_only_bm25", "sparse_chunk_ids"),
        ("hybrid_reranked", "retrieved_chunk_ids"),
    ]:
        precisions, recalls = [], []
        for r in valid:
            stage_ids = getattr(r, stage_field)
            relevant = set(r.relevant_chunks)
            precisions.append(_precision_at_k(stage_ids, relevant, k))
            recalls.append(_recall_at_k(stage_ids, relevant, k))
        stage_comparison[f"{stage_name}_precision_at_3"] = round(sum(precisions) / len(precisions) * 100, 1) if precisions else 0.0
        stage_comparison[f"{stage_name}_recall_at_3"] = round(sum(recalls) / len(recalls) * 100, 1) if recalls else 0.0

    return {
        "total_queries": total,
        "successful_queries": len(valid),
        "error_queries": len(errors),
        "precision_at_3": round(avg_p3 * 100, 1),
        "recall_at_3": round(avg_r3 * 100, 1),
        "mrr_at_3": round(avg_mrr3, 4),
        "coverage": round(coverage * 100, 1),
        "avg_latency_ms": round(avg_latency, 2),
        "stage_comparison": stage_comparison,
        "per_query": [asdict(r) for r in results],
    }


# --------------------------------------------------------------------------- #
#  Runner
# --------------------------------------------------------------------------- #

def _build_retriever():
    """Build and initialize the HybridRetriever."""
    from rag.hybrid_retrieval import HybridRetriever
    retriever = HybridRetriever()
    retriever.initialize()
    return retriever


def run_rag_evals() -> dict:
    """Run all RAG retrieval eval cases and return aggregated metrics."""
    import os
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    cases = _load_cases()
    all_chunks = _get_all_chunks()

    print("  Loading retriever (may take a few seconds for embeddings/reranker)...\n")
    retriever = _build_retriever()
    print(f"  Retriever ready. {len(all_chunks)} chunks loaded.\n")

    results = []
    for case in cases:
        result = _evaluate_case(case, retriever, all_chunks, k=3)
        status = "✅" if result.passed else "❌"
        p3 = result.precision_at_k.get(3, 0)
        r3 = result.recall_at_k.get(3, 0)
        mrr3 = result.mrr_at_k.get(3, 0)
        if result.error:
            print(f"  {status} {result.case_id} | ERROR: {result.error}")
        else:
            print(f"  {status} {result.case_id} | P@3={p3:.0%} R@3={r3:.0%} "
                  f"MRR@3={mrr3:.2f} | {result.latency_ms:.0f}ms")
        results.append(result)

    metrics = compute_rag_metrics(results)

    print(f"\n  ── RAG Retrieval Summary ──")
    print(f"  Queries:           {metrics['total_queries']}")
    print(f"  Successful:        {metrics['successful_queries']}")
    print(f"  Errors:            {metrics['error_queries']}")
    print(f"  Precision@3:       {metrics['precision_at_3']}%")
    print(f"  Recall@3:          {metrics['recall_at_3']}%")
    print(f"  MRR@3:             {metrics['mrr_at_3']}")
    print(f"  Coverage:          {metrics['coverage']}%")
    print(f"  Avg Latency:       {metrics['avg_latency_ms']:.0f}ms")

    return metrics


# --------------------------------------------------------------------------- #
#  pytest integration
# --------------------------------------------------------------------------- #

# Build retriever lazily for pytest (cached)
_rag_retriever = None
_rag_chunks = None

def _get_rag_retriever():
    global _rag_retriever, _rag_chunks
    if _rag_retriever is None:
        _rag_retriever = _build_retriever()
        _rag_chunks = _get_all_chunks()
    return _rag_retriever, _rag_chunks


class TestRagEvals:
    """Pytest wrapper for RAG retrieval evals."""

    cases_data = _load_cases()

    @pytest.mark.parametrize("case", cases_data, ids=[c["id"] for c in cases_data])
    def test_rag_retrieval(self, case):
        retriever, all_chunks = _get_rag_retriever()
        result = _evaluate_case(case, retriever, all_chunks, k=3)
        assert result.error is None, f"Case {case['id']} raised: {result.error}"
        assert result.passed, (
            f"No relevant chunk found in top-3 for query '{case['query']}'. "
            f"Retrieved: {result.retrieved_chunk_ids}, Expected relevant: {case['relevant_chunks']}"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    print("=" * 70)
    print("🔍 RAG Retrieval Evaluation")
    print("=" * 70)
    run_rag_evals()
