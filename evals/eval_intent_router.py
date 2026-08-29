"""
eval_intent_router.py — Evaluation harness for the deterministic intent router.

Tests intent classification accuracy and entity extraction quality across
all booking-related actions (book, search, cancel, reschedule, list, unknown).

Metrics:
  - Intent Classification Accuracy  = correct_actions / total
  - Entity Precision / Recall / F1  (per entity type: specialty, city, date,
    time, appointment_id, patient_email)
  - Missing Required Detection Rate  = correctly_identified_missing / total_missing_cases
  - Actionable Flag Accuracy         = correct_is_actionable / total

All tests are deterministic (pure Python, zero LLM). No API keys required.

Run:  python -m evals.eval_intent_router
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import pytest

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = Path(__file__).resolve().parent / "eval_data" / "intent_cases.json"

# Entity fields to check
ENTITY_FIELDS = ["specialty", "city", "date", "time", "appointment_id", "patient_email"]


# --------------------------------------------------------------------------- #
#  Data classes
# --------------------------------------------------------------------------- #

@dataclass
class IntentCaseResult:
    case_id: str
    category: str
    user_input: str

    # Actual results
    predicted_action: str
    predicted_entities: dict[str, Any]
    predicted_actionable: bool
    predicted_missing: list[str]

    # Correctness
    intent_correct: bool
    entities_correct: dict[str, bool]  # per-entity correctness
    missing_correct: bool
    actionable_correct: bool

    # Metrics
    passed: bool
    latency_ms: float = 0.0


@dataclass
class IntentEvalMetrics:
    total_cases: int = 0
    intent_accuracy: float = 0.0
    avg_latency_ms: float = 0.0

    entity_metrics: dict[str, dict] = field(default_factory=dict)
    missing_detection_rate: float = 0.0
    actionable_accuracy: float = 0.0

    by_category: dict[str, dict] = field(default_factory=dict)
    per_case: list[dict] = field(default_factory=list)


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def _load_cases() -> list[dict]:
    data = json.loads(DATA_FILE.read_text())
    return data["cases"]


def _evaluate_case(case: dict) -> IntentCaseResult:
    """Run a single intent classification case."""
    from booking_appointment.intent_router import classify_intent

    start = time.perf_counter()

    intent = classify_intent(case["user_input"])

    latency = (time.perf_counter() - start) * 1000

    predicted_entities = {
        f: getattr(intent, f) for f in ENTITY_FIELDS
    }

    # Check intent action
    intent_correct = intent.action == case["expected_action"]

    # Check each entity
    entities_correct = {}
    for field_name in ENTITY_FIELDS:
        expected_val = case["expected_entities"].get(field_name)
        actual_val = predicted_entities[field_name]
        if expected_val is None:
            entities_correct[field_name] = actual_val is None
        else:
            entities_correct[field_name] = (
                actual_val is not None and
                str(actual_val).strip().lower() == str(expected_val).strip().lower()
            )

    # Check missing required fields
    expected_missing_set = set(case.get("expected_missing", []))
    actual_missing_set = set(intent.missing_required)
    missing_correct = expected_missing_set == actual_missing_set

    # Check actionable flag
    actionable_correct = bool(intent.is_actionable()) == case["expected_actionable"]

    # Overall pass: intent correct AND all entities correct AND missing correct
    all_entities_correct = all(entities_correct.values())
    passed = intent_correct and all_entities_correct and missing_correct

    return IntentCaseResult(
        case_id=case["id"],
        category=case["category"],
        user_input=case["user_input"],
        predicted_action=intent.action,
        predicted_entities=predicted_entities,
        predicted_actionable=intent.is_actionable(),
        predicted_missing=list(intent.missing_required),
        intent_correct=intent_correct,
        entities_correct=entities_correct,
        missing_correct=missing_correct,
        actionable_correct=actionable_correct,
        passed=passed,
        latency_ms=latency,
    )


# --------------------------------------------------------------------------- #
#  Aggregation
# --------------------------------------------------------------------------- #

def compute_intent_metrics(results: list[IntentCaseResult]) -> dict:
    """Aggregate per-case results into summary metrics."""
    total = len(results)

    # Intent accuracy
    intent_correct_count = sum(r.intent_correct for r in results)
    intent_accuracy = intent_correct_count / total * 100 if total else 0.0

    # Per-entity precision/recall/F1
    # For entity extraction, we treat each entity prediction as a "classification":
    # TP = predicted correctly, FP = present but wrong, FN = expected but absent or wrong
    entity_metrics = {}
    for field_name in ENTITY_FIELDS:
        tp = sum(1 for r in results if r.entities_correct[field_name])
        # FP: entity was present in prediction but didn't match expected (or expected was None but got something)
        fp = sum(
            1 for r in results
            if not r.entities_correct[field_name] and r.predicted_entities[field_name] is not None
        )
        # FN: entity was expected but not correctly predicted
        fn = sum(
            1 for r in results
            if not r.entities_correct[field_name]
            and r.predicted_entities[field_name] is None
            and any(case["expected_entities"].get(field_name) for case in _load_cases() if case["id"] == r.case_id)
        )

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        entity_metrics[field_name] = {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(precision * 100, 1),
            "recall": round(recall * 100, 1),
            "f1": round(f1 * 100, 1),
        }

    # Missing required detection
    missing_cases = [r for r in results if r.predicted_missing]
    missing_correct_count = sum(r.missing_correct for r in results if r.predicted_missing)
    missing_detection_rate = missing_correct_count / len(missing_cases) * 100 if missing_cases else 100.0

    # Actionable accuracy
    actionable_correct = sum(r.actionable_correct for r in results)
    actionable_accuracy = actionable_correct / total * 100 if total else 0.0

    avg_latency = sum(r.latency_ms for r in results) / total if total else 0.0

    # Per-category
    by_category = {}
    for cat in sorted(set(r.category for r in results)):
        cat_results = [r for r in results if r.category == cat]
        n = len(cat_results)
        by_category[cat] = {
            "tests": n,
            "intent_accuracy": round(sum(r.intent_correct for r in cat_results) / n * 100, 1) if n else 0,
            "avg_latency_ms": round(sum(r.latency_ms for r in cat_results) / n, 2) if n else 0,
        }

    return {
        "total_cases": total,
        "intent_accuracy": round(intent_accuracy, 1),
        "avg_latency_ms": round(avg_latency, 2),
        "entity_metrics": entity_metrics,
        "missing_detection_rate": round(missing_detection_rate, 1),
        "actionable_accuracy": round(actionable_accuracy, 1),
        "by_category": by_category,
        "per_case": [asdict(r) for r in results],
    }


# --------------------------------------------------------------------------- #
#  Runner
# --------------------------------------------------------------------------- #

def run_intent_router_evals() -> dict:
    """Run all intent router eval cases and return aggregated metrics."""
    cases = _load_cases()
    results = []

    for case in cases:
        result = _evaluate_case(case)
        status = "✅" if result.passed else "❌"
        print(f"  {status} {result.case_id} | "
              f"intent={result.predicted_action} "
              f"(expected={case['expected_action']}) | "
              f"{result.latency_ms:.1f}ms")
        results.append(result)

    metrics = compute_intent_metrics(results)

    print(f"\n  ── Intent Router Summary ──")
    print(f"  Cases:              {metrics['total_cases']}")
    print(f"  Intent Accuracy:    {metrics['intent_accuracy']}%")
    print(f"  Actionable Acc:     {metrics['actionable_accuracy']}%")
    print(f"  Missing Detection:  {metrics['missing_detection_rate']}%")
    print(f"  Avg Latency:        {metrics['avg_latency_ms']:.1f}ms")

    print(f"\n  Entity F1 per type:")
    for field_name in ENTITY_FIELDS:
        m = metrics["entity_metrics"][field_name]
        print(f"    {field_name:20s}  P={m['precision']}%  "
              f"R={m['recall']}%  F1={m['f1']}%")

    # Print per-category
    print(f"\n  Per-category:")
    for cat, m in metrics["by_category"].items():
        print(f"    {cat:12s}  acc={m['intent_accuracy']}%  "
              f"lat={m['avg_latency_ms']:.1f}ms  n={m['tests']}")

    return metrics


# --------------------------------------------------------------------------- #
#  pytest integration
# --------------------------------------------------------------------------- #

class TestIntentRouterEvals:
    """Pytest wrapper — runs all intent cases as parametrized tests."""

    cases_data = _load_cases()

    @pytest.mark.parametrize("case", cases_data, ids=[c["id"] for c in cases_data])
    def test_intent_case(self, case):
        result = _evaluate_case(case)
        assert result.intent_correct, (
            f"Intent mismatch: got '{result.predicted_action}', "
            f"expected '{case['expected_action']}'"
        )
        assert all(result.entities_correct.values()), (
            f"Entity mismatch: {result.entities_correct}"
        )
        assert result.missing_correct, (
            f"Missing fields mismatch: got {result.predicted_missing}, "
            f"expected {case.get('expected_missing', [])}"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    print("=" * 70)
    print("🎯 Intent Router Evaluation")
    print("=" * 70)
    run_intent_router_evals()
