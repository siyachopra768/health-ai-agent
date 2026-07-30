"""
verifier.py — All deterministic validation logic in one place.
"""

import re


TEST_NAME_ALIASES = {
    "t. cholesterol": "Total Cholesterol",
    "s. cholesterol": "Total Cholesterol",
    "cholesterol, total": "Total Cholesterol",
    "hb": "Hemoglobin",
    "hgb": "Hemoglobin",
    "s. creatinine": "Creatinine",
    "sr. creatinine": "Creatinine",
}


def normalize_test_name(name: str) -> str:
    key = name.strip().lower()
    return TEST_NAME_ALIASES.get(key, name.strip())


def parse_reference_range(range_text: str) -> tuple[float, float]:
    range_text = range_text.strip()

    match = re.match(r"[>≥]\s*([\d.]+)", range_text)
    if match:
        return float(match.group(1)), float("inf")

    match = re.match(r"[<≤]\s*([\d.]+)", range_text)
    if match:
        return 0.0, float(match.group(1))

    match = re.match(r"([\d.]+)\s*[-–]\s*([\d.]+)", range_text)
    if match:
        return float(match.group(1)), float(match.group(2))

    raise ValueError(f"Could not parse reference range: {range_text!r}")


def normalize_value_for_matching(value) -> str:
    return str(value).replace(".0", "").strip()


def is_degenerate_value(value: float, normalized: str) -> bool:
    return value == 0.0 or len(normalized) < 2


def verify_value_against_source(value: float, raw_text: str) -> bool:
    normalized = normalize_value_for_matching(value)
    if is_degenerate_value(value, normalized):
        return False
    return normalized in raw_text


def has_direction_mismatch(query: str, chunk_topic: str) -> bool:
    query_lower = query.lower()
    topic_lower = chunk_topic.lower()
    if "low" in query_lower and "high" in topic_lower:
        return True
    if "high" in query_lower and "low" in topic_lower:
        return True
    return False


def is_likely_health_related(message: str, keywords: set[str] | None = None) -> bool:
    default_keywords = {
        "pain", "fever", "report", "doctor", "hospital", "appointment",
        "cholesterol", "risk", "symptom", "medicine", "test", "blood",
        "weak", "hemoglobin", "creatinine", "sugar", "diabetes",
    }
    keywords = keywords or default_keywords
    words = set(message.lower().split())
    return bool(words & keywords) or len(message.split()) > 3