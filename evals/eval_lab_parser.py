"""
eval_lab_parser.py — Evaluation harness for the lab report parser.

Measures extraction quality across two stages:
  1. Regex extraction (Stage 1, deterministic)
  2. LLM fallback extraction + hallucination verification (Stage 2)

Key metrics:
  - Extraction Precision  = TP / (TP + FP)
  - Extraction Recall     = TP / (TP + FN)
  - Hallucination Rate    = FP / (TP + FP)  [values extracted that shouldn't be]
  - Verifier Filter Rate  = (hallucinated - FP) / hallucinated  [LLM lies caught by verify_value_against_source]
  - F1 Score
  - Regex Coverage        = regex_cases_passing / total_regex_cases

The LLM fallback path mocks the Groq client so tests are deterministic
and don't require an API key.

Run:  python -m evals.eval_lab_parser
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = Path(__file__).resolve().parent / "eval_data" / "lab_reports.json"

# --------------------------------------------------------------------------- #
#  Data classes
# --------------------------------------------------------------------------- #


@dataclass
class LabEvalCaseResult:
    case_id: str
    case_type: str
    description: str

    # Counts
    tp: int = 0                       # true positives: correctly extracted
    fp: int = 0                       # false positives: hallucinated value that slipped through
    fn: int = 0                       # false negatives: ground truth missed
    hallucinated_in_mock: int = 0     # total hallucinated values the LLM produced
    filtered_by_verifier: int = 0    # hallucinated values caught by verify_value_against_source

    # Per-case metrics
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    hallucination_rate: float = 0.0
    verifier_filter_rate: float = 0.0
    latency_ms: float = 0.0

    # Diagnostics
    expected_keys: list[str] = field(default_factory=list)
    extracted_keys: list[str] = field(default_factory=list)
    hallucinated_keys_in_mock: list[str] = field(default_factory=list)
    hallucinated_that_slipped_through: list[str] = field(default_factory=list)
    expected_not_extracted: list[str] = field(default_factory=list)

    # Status
    passed: bool = False
    error: str | None = None


@dataclass
class LabEvalMetrics:
    total_cases: int = 0
    regex_cases: int = 0
    llm_cases: int = 0

    # Overall
    overall_tp: int = 0
    overall_fp: int = 0
    overall_fn: int = 0
    overall_hallucinated: int = 0
    overall_filtered: int = 0

    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    hallucination_rate: float = 0.0
    verifier_filter_rate: float = 0.0
    regex_coverage: float = 0.0  # % of regex cases where regex extracted >=1 value correctly
    avg_latency_ms: float = 0.0

    per_case: list[LabEvalCaseResult] = field(default_factory=list)


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def _load_cases() -> list[dict]:
    data = json.loads(DATA_FILE.read_text())
    return data["cases"]


def _make_mock_groq_response(llm_output: dict) -> MagicMock:
    """Build a mock Groq chat completion response that returns *llm_output*
    as a JSON string in the expected location."""
    raw_json = json.dumps(llm_output)
    return MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(content=raw_json)
            )
        ]
    )


def _normalize_value_for_comparison(val) -> str:
    """Normalize a numeric value to the string form the verifier uses."""
    return str(val).replace(".0", "").strip()


def _evaluate_case(case: dict) -> LabEvalCaseResult:
    """Run a single lab-parser eval case and compute metrics."""
    from parser import LabValueExtractor

    result = LabEvalCaseResult(
        case_id=case["id"],
        case_type=case["type"],
        description=case["description"],
        expected_keys=list(case["expected_extraction"].keys()),
    )

    start = time.perf_counter()

    try:
        extractor = LabValueExtractor(groq_api_key="test-key-not-used")
        raw_text = case["raw_text"]
        expected = case["expected_extraction"]

        if case["type"] == "regex":
            # Deterministic path — test _extract_regex directly
            extracted = extractor._extract_regex(raw_text)

        elif case["type"] == "llm_fallback_mocked":
            # Force LLM path by stubbing _extract_regex to return empty,
            # then mock the Groq client to return the controlled output.
            mock_llm_output = case["mock_llm_output"]
            hallucinated_keys = case.get("hallucinated_keys_in_mock", [])

            with patch.object(extractor, "_extract_regex", return_value={}), \
                 patch.object(extractor.client, "chat") as mock_chat:
                mock_chat.completions.create.return_value = _make_mock_groq_response(mock_llm_output)
                extracted = extractor._extract_llm(raw_text)

            result.hallucinated_in_mock = len(hallucinated_keys)
            result.hallucinated_keys_in_mock = hallucinated_keys
        else:
            raise ValueError(f"Unknown case type: {case['type']}")

        result.latency_ms = (time.perf_counter() - start) * 1000
        result.extracted_keys = list(extracted.keys())

        # ---- Compute TP / FP / FN ----
        # TP: a key is in extracted AND its value matches expected
        # FP: a key is in extracted but NOT in expected (hallucination slip-through)
        # FN: a key is in expected but NOT in extracted (over-filtered)
        tp_keys = []
        fp_keys = []
        fn_keys = []

        for key in extracted:
            if key in expected:
                # Verify the value matches
                exp_val = expected[key]["value"]
                act_val = extracted[key]["value"]
                if abs(float(exp_val) - float(act_val)) < 1e-6:
                    tp_keys.append(key)
                else:
                    fp_keys.append(key)  # value mismatch = hallucination
            else:
                fp_keys.append(key)  # key not in expected = hallucination

        for key in expected:
            if key not in extracted:
                fn_keys.append(key)

        result.tp = len(tp_keys)
        result.fp = len(fp_keys)
        result.fn = len(fn_keys)

        # ---- Verifier filter rate (LLM cases only) ----
        if case["type"] == "llm_fallback_mocked":
            # Of the hallucinated values the LLM produced, how many did the
            # verifier catch (i.e., are NOT in the extracted output)?
            slip_through = [k for k in result.hallucinated_keys_in_mock if k in extracted]
            caught = len(result.hallucinated_keys_in_mock) - len(slip_through)
            result.filtered_by_verifier = caught
            result.hallucinated_that_slipped_through = slip_through

        # ---- Per-case metrics ----
        total_extracted = result.tp + result.fp
        total_expected = result.tp + result.fn

        result.precision = result.tp / total_extracted if total_extracted > 0 else 1.0
        result.recall = result.tp / total_expected if total_expected > 0 else 1.0
        result.f1 = (
            2 * result.precision * result.recall / (result.precision + result.recall)
            if (result.precision + result.recall) > 0 else 0.0
        )
        result.hallucination_rate = result.fp / total_extracted if total_extracted > 0 else 0.0

        if case["type"] == "llm_fallback_mocked" and result.hallucinated_in_mock > 0:
            result.verifier_filter_rate = result.filtered_by_verifier / result.hallucinated_in_mock
        else:
            result.verifier_filter_rate = 1.0  # no hallucinations to filter

        result.expected_not_extracted = fn_keys

        # A case "passes" if precision == 1.0 (no hallucinations slipped through)
        # and recall == 1.0 (no expected values missed)
        result.passed = (result.precision == 1.0 and result.recall == 1.0)

    except Exception as e:
        result.error = str(e)
        result.passed = False
        logger.exception("Error evaluating case %s: %s", case["id"], e)

    return result


# --------------------------------------------------------------------------- #
#  Aggregation
# --------------------------------------------------------------------------- #


def compute_lab_metrics(results: list[LabEvalCaseResult]) -> dict:
    """Aggregate per-case results into summary metrics."""
    total = len(results)
    regex_cases = [r for r in results if r.case_type == "regex"]
    llm_cases = [r for r in results if r.case_type == "llm_fallback_mocked"]

    tp = sum(r.tp for r in results)
    fp = sum(r.fp for r in results)
    fn = sum(r.fn for r in results)
    hallucinated = sum(r.hallucinated_in_mock for r in results)
    filtered = sum(r.filtered_by_verifier for r in results)
    latencies = [r.latency_ms for r in results if r.error is None]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    hallucination_rate = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    verifier_filter_rate = filtered / hallucinated if hallucinated > 0 else 1.0

    # Regex coverage: % of regex cases that successfully extracted at least one value
    regex_passes = len([r for r in regex_cases if r.tp > 0])
    regex_coverage = regex_passes / len(regex_cases) if regex_cases else 0.0

    # LLM-specific
    llm_tp = sum(r.tp for r in llm_cases)
    llm_fp = sum(r.fp for r in llm_cases)
    llm_fn = sum(r.fn for r in llm_cases)
    llm_hallucinated = sum(r.hallucinated_in_mock for r in llm_cases)
    llm_filtered = sum(r.filtered_by_verifier for r in llm_cases)

    llm_precision = llm_tp / (llm_tp + llm_fp) if (llm_tp + llm_fp) > 0 else 1.0
    llm_hallucination_rate = llm_fp / (llm_tp + llm_fp) if (llm_tp + llm_fp) > 0 else 0.0
    llm_verifier_filter_rate = llm_filtered / llm_hallucinated if llm_hallucinated > 0 else 1.0

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    return {
        "total_cases": total,
        "regex_cases": len(regex_cases),
        "llm_cases": len(llm_cases),
        "overall": {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "hallucination_rate": round(hallucination_rate, 4),
            "verifier_filter_rate": round(verifier_filter_rate, 4),
            "regex_coverage": round(regex_coverage, 4),
            "avg_latency_ms": round(avg_latency, 2),
        },
        "by_stage": {
            "regex": {
                "cases": len(regex_cases),
                "passed": len([r for r in regex_cases if r.passed]),
                "coverage": round(regex_coverage, 4),
            },
            "llm_fallback": {
                "cases": len(llm_cases),
                "hallucinated_values_produced": llm_hallucinated,
                "hallucinated_filtered_by_verifier": llm_filtered,
                "hallucinated_slipped_through": llm_fp,
                "precision": round(llm_precision, 4),
                "hallucination_rate": round(llm_hallucination_rate, 4),
                "verifier_filter_rate": round(llm_verifier_filter_rate, 4),
            },
        },
        "per_case": [asdict(r) for r in results],
    }


# --------------------------------------------------------------------------- #
#  Runner
# --------------------------------------------------------------------------- #

def run_lab_parser_evals() -> dict:
    """Run all lab parser eval cases and return aggregated metrics."""
    cases = _load_cases()
    results = []

    for case in cases:
        result = _evaluate_case(case)
        status = "✅" if result.passed else "❌"
        print(f"  {status} {result.case_id} ({result.case_type}) | "
              f"P={result.precision:.0%} R={result.recall:.0%} "
              f"F1={result.f1:.0%} Hall={result.hallucination_rate:.0%} "
              f"Verify={result.verifier_filter_rate:.0%} "
              f"| {result.latency_ms:.0f}ms")
        results.append(result)

    metrics = compute_lab_metrics(results)

    o = metrics["overall"]
    print(f"\n  ── Lab Parser Summary ──")
    print(f"  Cases:           {o['true_positives'] + o['false_positives'] + o['false_negatives']}")
    print(f"  True Positives:  {o['true_positives']}")
    print(f"  False Positives: {o['false_positives']}")
    print(f"  False Negatives: {o['false_negatives']}")
    print(f"  Precision:       {o['precision']:.1%}")
    print(f"  Recall:          {o['recall']:.1%}")
    print(f"  F1 Score:        {o['f1_score']:.1%}")
    print(f"  Hallucination Rate: {o['hallucination_rate']:.1%}")
    print(f"  Verifier Filter Rate: {o['verifier_filter_rate']:.1%}")
    print(f"  Regex Coverage:  {o['regex_coverage']:.1%}")
    print(f"  Avg Latency:     {o['avg_latency_ms']:.0f}ms")

    # Print hallucination details for LLM cases
    llm_results = [r for r in results if r.case_type == "llm_fallback_mocked"]
    if llm_results:
        slipped = [r for r in llm_results if r.hallucinated_that_slipped_through]
        if slipped:
            print(f"\n  ⚠️  {len(slipped)} case(s) had hallucinated values slip past verification:")
            for r in slipped:
                print(f"    - {r.case_id}: {r.hallucinated_that_slipped_through} slipped through")

    return metrics


# --------------------------------------------------------------------------- #
#  pytest integration (so `pytest evals/` also works)
# --------------------------------------------------------------------------- #

class TestLabParserEvals:
    """Pytest wrapper — runs the same eval cases as a test suite."""

    cases_data = _load_cases()

    @pytest.mark.parametrize("case", cases_data, ids=[c["id"] for c in _load_cases()])
    def test_extraction_case(self, case):
        result = _evaluate_case(case)
        assert result.error is None, f"Case {case['id']} raised error: {result.error}"
        assert result.passed, (
            f"Case {case['id']} failed: "
            f"TP={result.tp}, FP={result.fp}, FN={result.fn}, "
            f"slipped_through={result.hallucinated_that_slipped_through}"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 70)
    print("🧪 Lab Parser Evaluation")
    print("=" * 70)
    run_lab_parser_evals()
