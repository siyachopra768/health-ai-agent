"""
intent_router.py — Deterministic intent classification & entity extraction
Pure Python, zero LLM calls. Returns structured Intent for downstream handlers.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Final, Literal

# ── Logging ────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ── Constants ──────────────────────────────────────────────────────────────────
SPECIALTY_ALIASES: Final[dict[str, str]] = {
    "heart doctor": "Cardiologist",
    "heart specialist": "Cardiologist",
    "cardiologist": "Cardiologist",
    "cardiology": "Cardiologist",
    "diabetes doctor": "Endocrinologist",
    "diabetes specialist": "Endocrinologist",
    "endocrinologist": "Endocrinologist",
    "thyroid doctor": "Endocrinologist",
    "hormone doctor": "Endocrinologist",
    "endocrinology": "Endocrinologist",
    "blood doctor": "Hematologist",
    "blood specialist": "Hematologist",
    "hematologist": "Hematologist",
    "hematology": "Hematologist",
    "gp": "General Physician",
    "general physician": "General Physician",
    "physician": "General Physician",
    "family doctor": "General Physician",
    "primary care": "General Physician",
    "internist": "Internal Medicine",
    "internal medicine": "Internal Medicine",
}

CITY_ALIASES: Final[dict[str, str]] = {
    "delhi": "Delhi",
    "new delhi": "Delhi",
    "gurgaon": "Gurgaon",
    "gurugram": "Gurgaon",
    "jaipur": "Jaipur",
    "kota": "Kota",
}

# Pre-compiled regex patterns
DATE_ISO_PATTERN = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
DATE_DMY_PATTERN = re.compile(r"\b(\d{2})[/-](\d{2})[/-](\d{4})\b")
TIME_24H_PATTERN = re.compile(r"\b(\d{1,2}):(\d{2})\b")
TIME_12H_PATTERN = re.compile(r"\b(\d{1,2})\s*(am|pm)\b", re.IGNORECASE)
APPT_ID_PATTERN = re.compile(r"\b([a-zA-Z0-9]{6,12})\b")
EMAIL_PATTERN = re.compile(r"\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b")

ActionType = Literal["book", "search", "cancel", "reschedule", "list", "unknown"]


# ── Data Classes ───────────────────────────────────────────────────────────────
@dataclass(slots=True, frozen=True)
class Intent:
    """Structured intent with extracted entities."""
    action: ActionType
    specialty: str | None = None
    city: str | None = None
    date: str | None = None          # ISO format YYYY-MM-DD
    time: str | None = None          # 24h format HH:MM
    appointment_id: str | None = None
    patient_email: str | None = None
    confidence: float = 0.0
    missing_required: list[str] = field(default_factory=list)

    def is_actionable(self) -> bool:
        """True if intent is clear and has minimum required entities."""
        return self.action != "unknown" and self.confidence >= 0.6

    def __str__(self) -> str:
        parts = [f"action={self.action}", f"confidence={self.confidence:.2f}"]
        for f in ("specialty", "city", "date", "time", "appointment_id", "patient_email"):
            v = getattr(self, f)
            if v:
                parts.append(f"{f}={v}")
        if self.missing_required:
            parts.append(f"missing={self.missing_required}")
        return f"Intent({', '.join(parts)})"


# ── Extraction Functions ───────────────────────────────────────────────────────
def _norm(text: str) -> str:
    return text.lower().strip()


def extract_specialty(text: str) -> str | None:
    norm = _norm(text)
    for alias, canon in SPECIALTY_ALIASES.items():
        if alias in norm:
            logger.debug("specialty: '%s' -> '%s'", alias, canon)
            return canon
    return None


def extract_city(text: str) -> str | None:
    norm = _norm(text)
    for alias, canon in CITY_ALIASES.items():
        if alias in norm:
            logger.debug("city: '%s' -> '%s'", alias, canon)
            return canon
    return None


def extract_date(text: str) -> str | None:
    norm = _norm(text)
    today = date.today()

    if "today" in norm:
        return today.isoformat()
    if "tomorrow" in norm:
        return (today + timedelta(days=1)).isoformat()
    if "day after tomorrow" in norm:
        return (today + timedelta(days=2)).isoformat()

    m = DATE_ISO_PATTERN.search(text)
    if m:
        try:
            return date.fromisoformat(m.group(1)).isoformat()
        except ValueError:
            pass

    m = DATE_DMY_PATTERN.search(text)
    if m:
        try:
            d, m_, y = m.groups()
            return date(int(y), int(m_), int(d)).isoformat()
        except ValueError:
            pass
    return None


def extract_time(text: str) -> str | None:
    norm = _norm(text)

    m = TIME_24H_PATTERN.search(text)
    if m:
        h, mi = int(m.group(1)), int(m.group(2))
        if 0 <= h <= 23 and 0 <= mi <= 59:
            return f"{h:02d}:{mi:02d}"

    m = TIME_12H_PATTERN.search(norm)
    if m:
        h = int(m.group(1))
        period = m.group(2).lower()
        if period == "pm" and h != 12:
            h += 12
        elif period == "am" and h == 12:
            h = 0
        if 0 <= h <= 23:
            return f"{h:02d}:00"
    return None


def extract_appointment_id(text: str) -> str | None:
    m = APPT_ID_PATTERN.search(text)
    return m.group(1) if m else None


def extract_email(text: str) -> str | None:
    m = EMAIL_PATTERN.search(text)
    return m.group(1) if m else None


# ── Main Router ────────────────────────────────────────────────────────────────
def classify_intent(message: str) -> Intent:
    """
    Classify user message into structured Intent.
    Pure deterministic logic — no LLM.
    """
    if not message or not message.strip():
        return Intent(action="unknown", confidence=0.0)

    text = _norm(message)
    logger.info("Classifying: %s", message[:80])

    # Extract all entities
    specialty = extract_specialty(message)
    city = extract_city(message)
    date_str = extract_date(message)
    time_str = extract_time(message)
    appt_id = extract_appointment_id(message)
    email = extract_email(message)

    # Score actions
    scores: dict[ActionType, int] = {
        "book": 0, "search": 0, "cancel": 0, "reschedule": 0, "list": 0
    }

    # Explicit verbs
    if any(k in text for k in ("cancel", "delete appointment")):
        scores["cancel"] += 3
    if any(k in text for k in ("reschedule", "move appointment", "change appointment", "shift appointment")):
        scores["reschedule"] += 3
    if any(k in text for k in ("book", "schedule", "make appointment", "set up appointment")):
        scores["book"] += 3

    # "meet/see/visit/consult" + specialty/doctor → book
    if any(k in text for k in ("meet", "see", "visit", "consult")):
        if specialty or "doctor" in text or "specialist" in text:
            scores["book"] += 2

    # Search signals
    if any(k in text for k in ("search", "find", "show", "list hospitals", "available", "slots", "timings")):
        scores["search"] += 2

    # List signals
    if any(k in text for k in ("my appointments", "upcoming", "show appointments", "list appointments")):
        scores["list"] += 3

    # Context boosters
    if appt_id:
        if "cancel" in text:
            scores["cancel"] += 2
        elif any(k in text for k in ("reschedule", "move", "change")):
            scores["reschedule"] += 2
        else:
            scores["cancel"] += 1

    if specialty or city:
        scores["search"] += 1
        scores["book"] += 1

    if date_str or time_str:
        scores["book"] += 1
        scores["reschedule"] += 1

    # Determine best action
    if all(v == 0 for v in scores.values()):
        action: ActionType = "unknown"
        confidence = 0.0
    else:
        action = max(scores, key=scores.get)
        total = sum(scores.values())
        confidence = scores[action] / total if total else 0.0

    # Identify missing required fields for actionable intents
    missing = []
    if action == "book":
        if not specialty: missing.append("specialty")
        if not date_str: missing.append("date")
        if not time_str: missing.append("time")
    elif action == "reschedule":
        if not appt_id: missing.append("appointment_id")
        if not date_str and not time_str: missing.append("new_date_or_time")
    elif action == "cancel":
        if not appt_id: missing.append("appointment_id")
    elif action == "list":
        pass  # email optional
    elif action == "search":
        pass  # specialty/city optional

    intent = Intent(
        action=action,
        specialty=specialty,
        city=city,
        date=date_str,
        time=time_str,
        appointment_id=appt_id,
        patient_email=email,
        confidence=confidence,
        missing_required=missing,
    )

    logger.info("Result: %s", intent)
    return intent


# ── Self-Test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    tests = [
        "I want to meet cardiologist tomorrow at 10am",
        "Book endocrinologist at AIIMS Delhi for 2026-08-15 at 10:00",
        "Find hematologists in Gurgaon",
        "Cancel appointment abc12345",
        "Reschedule abc12345 to tomorrow at 11:00",
        "Show my upcoming appointments for john@example.com",
        "What does high cholesterol mean?",
        "Search for heart doctor in Delhi",
        "I need a diabetes specialist",
        "Available slots at Medanta tomorrow",
    ]

    for msg in tests:
        intent = classify_intent(msg)
        print(f"\n{msg}")
        print(f"  {intent}")
        print(f"  actionable: {intent.is_actionable()}")