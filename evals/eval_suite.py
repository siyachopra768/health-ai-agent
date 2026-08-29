"""
eval_suite.py — Unified evaluation runner.

Runs all four eval harnesses and produces:
  1. A structured JSON report  (eval_suite_results.json)
  2. A human-readable markdown summary (eval_suite_report.md)

Eval modules:
  eval_lab_parser.py       Lab report extraction + hallucination detection
  eval_intent_router.py    Deterministic intent classification + entity F1
  eval_rag.py             Hybrid retrieval Precision@k, Recall@k, MRR@k
  eval_hybrid_router.py   End-to-end routing accuracy (async, may be slow)

Run everything:
  python -m evals.eval_suite

Run individual modules:
  python -m evals.eval_lab_parser
  python -m evals.eval_intent_router
  python -m evals.eval_rag
  python -m evals.eval_hybrid_router

Run with pytest:
  pytest evals/ -v
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
#  Config
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_JSON = PROJECT_ROOT / "eval_suite_results.json"
OUTPUT_MD   = PROJECT_ROOT / "eval_suite_report.md"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval_suite")


# --------------------------------------------------------------------------- #
#  Phase runners
# --------------------------------------------------------------------------- #

def run_lab_parser() -> dict[str, Any]:
    """Run lab parser evals. Synchronous — uses mocks."""
    logger.info("Running: eval_lab_parser")
    from evals.eval_lab_parser import run_lab_parser_evals
    t0 = time.perf_counter()
    metrics = run_lab_parser_evals()
    elapsed = time.perf_counter() - t0
    logger.info("eval_lab_parser done in %.1fs", elapsed)
    return metrics


def run_intent_router() -> dict[str, Any]:
    """Run intent router evals. Synchronous — pure Python."""
    logger.info("Running: eval_intent_router")
    from evals.eval_intent_router import run_intent_router_evals
    t0 = time.perf_counter()
    metrics = run_intent_router_evals()
    elapsed = time.perf_counter() - t0
    logger.info("eval_intent_router done in %.1fs", elapsed)
    return metrics


def run_rag() -> dict[str, Any]:
    """Run RAG retrieval evals. Synchronous but loads ML models."""
    logger.info("Running: eval_rag")
    from evals.eval_rag import run_rag_evals
    t0 = time.perf_counter()
    metrics = run_rag_evals()
    elapsed = time.perf_counter() - t0
    logger.info("eval_rag done in %.1fs", elapsed)
    return metrics


async def run_hybrid_router() -> dict[str, Any]:
    """Run hybrid router evals. Async — calls real router with booking API."""
    logger.info("Running: eval_hybrid_router")
    from evals.eval_hybrid_router import run_hybrid_router_evals
    t0 = time.perf_counter()
    metrics = await run_hybrid_router_evals()
    elapsed = time.perf_counter() - t0
    logger.info("eval_hybrid_router done in %.1fs", elapsed)
    return metrics


# --------------------------------------------------------------------------- #
#  Report generation
# --------------------------------------------------------------------------- #

def build_summary(results: dict[str, Any]) -> dict[str, Any]:
    """Pull key metrics from each module result into a one-level summary."""
    summary: dict[str, Any] = {
        "lab_parser": {},
        "intent_router": {},
        "rag": {},
        "hybrid_router": {},
    }

    lr = results.get("lab_parser", {})
    if "overall" in lr:
        o = lr["overall"]
        summary["lab_parser"] = {
            "precision": o.get("precision"),
            "recall": o.get("recall"),
            "f1_score": o.get("f1_score"),
            "hallucination_rate": o.get("hallucination_rate"),
            "verifier_filter_rate": o.get("verifier_filter_rate"),
            "regex_coverage": o.get("regex_coverage"),
            "cases": o.get("true_positives", 0) + o.get("false_positives", 0) + o.get("false_negatives", 0),
        }

    ir = results.get("intent_router", {})
    summary["intent_router"] = {
        "intent_accuracy": ir.get("intent_accuracy"),
        "actionable_accuracy": ir.get("actionable_accuracy"),
        "missing_detection_rate": ir.get("missing_detection_rate"),
        "total_cases": ir.get("total_cases"),
    }

    rg = results.get("rag", {})
    summary["rag"] = {
        "precision_at_3": rg.get("precision_at_3"),
        "recall_at_3": rg.get("recall_at_3"),
        "mrr_at_3": rg.get("mrr_at_3"),
        "coverage": rg.get("coverage"),
        "total_queries": rg.get("total_queries"),
    }

    hr = results.get("hybrid_router", {})
    summary["hybrid_router"] = {
        "route_accuracy": hr.get("route_accuracy"),
        "outcome_accuracy": hr.get("outcome_accuracy"),
        "avg_latency_ms": hr.get("avg_latency_ms"),
        "total_cases": hr.get("total_cases"),
        "multiturn_success": hr.get("multiturn_success"),
        "multiturn_cases": hr.get("multiturn_cases"),
    }

    return summary


def generate_markdown(results: dict[str, Any]) -> str:
    """Render a human-readable markdown report."""
    summary = build_summary(results)
    ts = results.get("timestamp", datetime.now().isoformat())
    total_time = results.get("total_time_seconds", 0)

    lp = summary.get("lab_parser", {})
    ir = summary.get("intent_router", {})
    rg = summary.get("rag", {})
    hr = summary.get("hybrid_router", {})

    # Grade assignments
    def grade(pct: float | None, thresholds=("F", "D", "C", "B", "A"), cuts=(50, 65, 80, 90)):
        if pct is None:
            return "N/A"
        # Auto-detect scale: if value looks like a fraction (<=1.0), rescale to 0-100
        if pct <= 1.0:
            pct_display = pct * 100
        else:
            pct_display = pct
        for letter, cut in zip(reversed(thresholds), reversed(cuts)):
            if pct_display >= cut:
                return f"{letter} ({pct_display:.1f}%)"
        return f"F ({pct_display:.1f}%)"

    lines: list[str] = []
    lines.append(f"# 🩺 AI Health Agent — Evaluation Report")
    lines.append(f"")
    lines.append(f"**Generated:** {ts}")
    lines.append(f"**Total runtime:** {total_time:.1f}s")
    lines.append("")

    # ── Lab Parser ──────────────────────────────────────────────────────────────
    lines.append("## 🧪 Lab Report Parser")
    lines.append("")
    lines.append("| Metric | Value | Grade |")
    lines.append("|--------|-------|-------|")
    lines.append(f"| Precision | {grade(lp.get('precision'), cuts=(80, 85, 90, 95))} | |")
    lines.append(f"| Recall | {grade(lp.get('recall'), cuts=(80, 85, 90, 95))} | |")
    lines.append(f"| F1 Score | {grade(lp.get('f1_score'), cuts=(80, 85, 90, 95))} | |")
    lines.append(f"| Hallucination Rate | {grade(1-(lp.get('hallucination_rate') or 0), cuts=(50, 65, 80, 90), thresholds=('F','D','C','B','A'))} | |")
    lines.append(f"| Verifier Filter Rate | {grade(lp.get('verifier_filter_rate'), cuts=(60, 75, 85, 95))} | |")
    lines.append(f"| Regex Coverage | {grade(lp.get('regex_coverage'), cuts=(60, 75, 85, 95))} | |")
    lines.append("")
    lines.append(f"**Test cases:** {lp.get('cases', 'N/A')}")
    lines.append("")

    # ── Intent Router ───────────────────────────────────────────────────────────
    lines.append("## 🎯 Intent Router")
    lines.append("")
    lines.append("| Metric | Value | Grade |")
    lines.append("|--------|-------|-------|")
    lines.append(f"| Intent Accuracy | {grade(ir.get('intent_accuracy'))} | |")
    lines.append(f"| Actionable Accuracy | {grade(ir.get('actionable_accuracy'))} | |")
    lines.append(f"| Missing Detection Rate | {grade(ir.get('missing_detection_rate'))} | |")
    lines.append("")
    lines.append(f"**Test cases:** {ir.get('total_cases', 'N/A')}")
    lines.append("")

    # ── RAG ───────────────────────────────────────────────────────────────────
    lines.append("## 📚 Hybrid RAG Retrieval")
    lines.append("")
    lines.append("| Metric | Value | Grade |")
    lines.append("|--------|-------|-------|")
    lines.append(f"| Precision@3 | {grade(rg.get('precision_at_3'))} | |")
    lines.append(f"| Recall@3 | {grade(rg.get('recall_at_3'))} | |")
    lines.append(f"| MRR@3 | {rg.get('mrr_at_3', 'N/A')} | |")
    lines.append(f"| Coverage | {grade(rg.get('coverage'))} | |")
    lines.append("")
    lines.append(f"**Test queries:** {rg.get('total_queries', 'N/A')}")
    lines.append("")

    # ── Hybrid Router ──────────────────────────────────────────────────────────
    lines.append("## 🔀 Hybrid Router (End-to-End)")
    lines.append("")
    lines.append("| Metric | Value | Grade |")
    lines.append("|--------|-------|-------|")
    lines.append(f"| Route Accuracy | {grade(hr.get('route_accuracy'))} | |")
    lines.append(f"| Outcome Accuracy | {grade(hr.get('outcome_accuracy'))} | |")
    lines.append(f"| Avg Latency | {hr.get('avg_latency_ms', 'N/A')} ms | |")
    lines.append("")
    lines.append(f"**Multi-turn:** {hr.get('multiturn_success', '?')}/{hr.get('multiturn_cases', '?')} succeeded")
    lines.append(f"**Total cases:** {hr.get('total_cases', 'N/A')}")
    lines.append("")

    # ── Stage comparison (RAG) ────────────────────────────────────────────────
    rag_res = results.get("rag", {})
    stage_comp = rag_res.get("stage_comparison", {})
    if stage_comp:
        lines.append("## 🔬 RAG Stage Comparison (Dense vs Sparse vs Hybrid)")
        lines.append("")
        lines.append("| Stage | Precision@3 | Recall@3 |")
        lines.append("|-------|-------------|----------|")
        stages = [
            ("Dense Only", "dense_only"),
            ("Sparse (BM25)", "sparse_only_bm25"),
            ("Hybrid + Rerank", "hybrid_reranked"),
        ]
        for label, key in stages:
            p = stage_comp.get(f"{key}_precision_at_3", "N/A")
            r = stage_comp.get(f"{key}_recall_at_3", "N/A")
            lines.append(f"| {label} | {p}% | {r}% |")
        lines.append("")

    # ── Entity F1 (Intent Router) ─────────────────────────────────────────────
    ir_res = results.get("intent_router", {})
    entity_metrics = ir_res.get("entity_metrics", {})
    if entity_metrics:
        lines.append("## 📊 Intent Router — Entity F1 by Type")
        lines.append("")
        lines.append("| Entity | Precision | Recall | F1 |")
        lines.append("|--------|-----------|--------|-----|")
        for field_name in ["specialty", "city", "date", "time", "appointment_id", "patient_email"]:
            m = entity_metrics.get(field_name, {})
            lines.append(
                f"| {field_name} | {m.get('precision', 'N/A')}% | "
                f"{m.get('recall', 'N/A')}% | {m.get('f1', 'N/A')}% |"
            )
        lines.append("")

    # ── Per-path accuracy (Hybrid Router) ──────────────────────────────────────
    hr_res = results.get("hybrid_router", {})
    path_acc = hr_res.get("route_accuracy_by_path", {})
    if path_acc:
        lines.append("## 🔀 Hybrid Router — Accuracy by Path")
        lines.append("")
        lines.append("| Path | Accuracy | Cases |")
        lines.append("|------|----------|-------|")
        for path, m in path_acc.items():
            lines.append(f"| {path} | {m.get('accuracy', 'N/A')}% | {m.get('tests', '?')} |")
        lines.append("")

    # ── ATS Summary ────────────────────────────────────────────────────────────
    lines.append("## 📋 ATS Readiness Summary")
    lines.append("")

    # Compute overall grades
    metrics_raw = [
        ("Lab Parser F1", lp.get("f1_score"), 0.80),
        ("Intent Accuracy", ir.get("intent_accuracy"), 80),
        ("RAG Precision@3", rg.get("precision_at_3"), 80),
        ("Router Route Accuracy", hr.get("route_accuracy"), 80),
    ]

    lines.append("| Component | Threshold | Actual | Status |")
    lines.append("|-----------|-----------|--------|--------|")
    ats_ready = True
    for label, actual, threshold in metrics_raw:
        # Auto-detect display scale: 0-1 fractions → percentage for display
        display_val = round(actual * 100, 1) if actual is not None and actual <= 1.0 else actual
        if actual is None:
            status = "⚠️  N/A"
            ats_ready = False
        elif actual >= threshold:
            status = "✅ Pass"
        else:
            status = "❌ Fail"
            ats_ready = False
        lines.append(f"| {label} | ≥{threshold} | {display_val} | {status} |")

    lines.append("")
    if ats_ready:
        lines.append("**🎉 All components meet ATS thresholds — ready for production evaluation.**")
    else:
        lines.append("**⚠️  One or more components below ATS threshold — needs improvement.**")
    lines.append("")
    lines.append("---")
    lines.append(f"*Report generated by eval_suite.py at {ts}*")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

async def run_all() -> dict[str, Any]:
    """Run all eval modules and save results."""
    t0 = time.perf_counter()

    results: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "version": "1.0",
    }

    # Lab parser (synchronous)
    try:
        results["lab_parser"] = run_lab_parser()
    except Exception as e:
        logger.exception("eval_lab_parser failed")
        results["lab_parser"] = {"error": str(e)}

    # Intent router (synchronous)
    try:
        results["intent_router"] = run_intent_router()
    except Exception as e:
        logger.exception("eval_intent_router failed")
        results["intent_router"] = {"error": str(e)}

    # RAG (synchronous, loads ML models)
    try:
        results["rag"] = run_rag()
    except Exception as e:
        logger.exception("eval_rag failed")
        results["rag"] = {"error": str(e)}

    # Hybrid router (async)
    try:
        results["hybrid_router"] = await run_hybrid_router()
    except Exception as e:
        logger.exception("eval_hybrid_router failed")
        results["hybrid_router"] = {"error": str(e)}

    results["total_time_seconds"] = round(time.perf_counter() - t0, 1)

    return results


def main():
    print("=" * 70)
    print("🩺 AI Health Agent — Evaluation Suite")
    print("=" * 70)
    print()

    results = asyncio.run(run_all())

    # Save JSON
    OUTPUT_JSON.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n💾 JSON report → {OUTPUT_JSON}")

    # Generate and save markdown
    md = generate_markdown(results)
    OUTPUT_MD.write_text(md)
    print(f"📄 Markdown report → {OUTPUT_MD}")

    print()
    print("=" * 70)
    print("📋 ATS SUMMARY")
    print("=" * 70)
    summary = build_summary(results)

    for module, metrics in summary.items():
        if "error" in metrics:
            print(f"  ❌ {module}: ERROR — {metrics['error']}")
        else:
            key_metric = {
                "lab_parser": ("F1", metrics.get("f1_score")),
                "intent_router": ("Accuracy", metrics.get("intent_accuracy")),
                "rag": ("Precision@3", metrics.get("precision_at_3")),
                "hybrid_router": ("Route Acc", metrics.get("route_accuracy")),
            }
            label, val = key_metric.get(module, ("?", None))
            if val is not None:
                # Auto-detect 0-1 scale for percentage-like metrics
                if val <= 1.0:
                    val_pct = val * 100
                else:
                    val_pct = val
                grade_str = grade(val)
                print(f"  {grade_str} {module:20s}  {label}: {val_pct:.1f}%")
            else:
                print(f"  ❌ {module:20s}  (no data)")

    print()
    print(f"Total runtime: {results['total_time_seconds']:.1f}s")
    print(f"Reports: {OUTPUT_JSON}  {OUTPUT_MD}")


if __name__ == "__main__":
    main()
