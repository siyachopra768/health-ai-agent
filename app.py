"""
app.py — Streamlit frontend for Health AI Agent
Uses the hybrid router for appointment booking, guideline RAG, and
general medical queries.
"""

import streamlit as st
import json
import hashlib
import logging
import asyncio
import re
from typing import Optional
from pathlib import Path

from parser import LabValueExtractor
from utils import analyze_values, calculate_risk_score

# Import the hybrid router
from booking_appointment.hybrid_router import route_message

# ── Logging ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger(__name__)

# ── Validation Helpers ─────────────────────────────────────────────────────────

PDF_EXTENSIONS = {".pdf"}
MAX_PDF_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB max
ALLOWED_EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
MAX_MESSAGE_LENGTH = 5000  # Max characters per user message


def validate_pdf_file(file) -> tuple[bool, str]:
    """
    Validate uploaded PDF file.

    Args:
        file: Uploaded file object from Streamlit file_uploader

    Returns:
        Tuple of (is_valid, error_message)
    """
    if file is None:
        return False, "No file uploaded"

    # Check file extension
    try:
        file_name = Path(file.name)
        ext = file_name.suffix.lower()
        if ext not in PDF_EXTENSIONS:
            return False, f"Invalid file type: {ext}. Only PDF files are accepted."
    except Exception:
        return False, "File name validation failed"

    # Check file size
    try:
        file_bytes = file.getvalue()
        file_size = len(file_bytes)
        if file_size == 0:
            return False, "Uploaded file is empty"
        if file_size > MAX_PDF_SIZE_BYTES:
            return False, f"File too large: {file_size // 1024}KB. Maximum size is 10MB."
        # Check that file has meaningful content (look for PDF header)
        if len(file_bytes) < 10 or not file_bytes.startswith(b"%PDF"):
            return False, "File does not appear to be a valid PDF format"
    except Exception as e:
        logger.error(f"File validation error: {e}")
        return False, "File validation failed"

    return True, ""


def validate_email(email: str) -> bool:
    """
    Validate email format using regex pattern.

    Args:
        email: Email address to validate

    Returns:
        True if valid, False otherwise
    """
    if not email or not isinstance(email, str):
        return False
    return bool(ALLOWED_EMAIL_REGEX.match(email.strip()))


def sanitize_user_text(text: str, max_length: int = MAX_MESSAGE_LENGTH) -> str:
    """
    Sanitize user-provided text to prevent injection attacks.

    Args:
        text: User input text
        max_length: Maximum allowed length

    Returns:
        Sanitized text
    """
    if not text or not isinstance(text, str):
        return ""

    # Remove excessive whitespace and control characters
    text = text.strip()
    text = re.sub(r"\s+", " ", text)  # Collapse multiple spaces
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)  # Remove control characters

    # Limit length
    if len(text) > max_length:
        logger.warning(f"User message truncated from {len(text)} to {max_length} chars")
        text = text[:max_length]

    return text


def initialize_session_state():
    """Initialize session state with default values."""
    defaults = {
        "analysis": None,
        "last_file": None,
        "last_file_hash": None,
        "appointments": [],
        "chat": [],
        "booking_context": {},  # multi-turn booking state (hospital choice, patient info)
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


# ── Load hospital data ────────────────────────────────────────────────────────
try:
    hospitals_path = Path("booking_appointment/hospitals.json")
    if not hospitals_path.exists():
        logger.error(f"Hospital data file not found: {hospitals_path}")
        hospitals = []
    else:
        with open(hospitals_path, "r") as f:
            hospitals = json.load(f)
except Exception as e:
    logger.error(f"Failed to load hospital data: {e}")
    hospitals = []


# ── SESSION STATE ────────────────────────────────────────────────────────
# Initialise session state variables, including placeholders for uploaded
# file metadata so we can avoid re‑parsing the same file on every message.
for key, default in {
    "analysis": None,
    "last_file": None,
    "last_file_hash": None,
    "uploaded_file_bytes": None,   # raw bytes of the currently loaded PDF
    "uploaded_file_name": None,    # name of the currently loaded PDF
    "appointments": [],
    "chat": [],
    "booking_context": {},         # multi‑turn booking state (hospital choice, patient info)
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# =========================================================
# 🧠 DEFICIENCY SUMMARY
# =========================================================
def generate_summary() -> Optional[str]:
    """Generate a summary of lab analysis results."""
    data = st.session_state.analysis

    if not data:
        return None

    if not isinstance(data, dict):
        logger.error(f"Invalid analysis data type: {type(data)}")
        return None

    abnormal = [
        f"{k} ({v['status']})"
        for k, v in data.items()
        if isinstance(v, dict) and v.get("status") != "normal"
    ]

    if not abnormal:
        return "All parameters are within normal range."

    return "Key abnormalities detected: " + ", ".join(abnormal[:3]) + "."


# =========================================================
# 🧠 HEALTH CHECK
# =========================================================
def run_health_check() -> dict:
    """Run health check on critical components."""
    health = {
        "pdf_processor": "ok",
        "hospital_data": "ok" if hospitals else "warning",
        "router": "ok",
    }

    if not hospitals:
        health["hospital_data"] = "warning"

    return health


# =========================================================
# 🧠 MAIN AGENT ENGINE — routes through the hybrid router
# =========================================================
async def handle_async(message: str) -> str:
    """
    Async handler for processing user messages.

    Args:
        message: User input message

    Returns:
        Response string from the router
    """
    # Sanitize message
    sanitized = sanitize_user_text(message)

    try:
        result = await route_message(sanitized, st.session_state.booking_context)
        return result
    except asyncio.TimeoutError:
        logger.error(f"Timeout processing message: {sanitized[:100]}")
        return "Request timed out. Please try again with a simpler question."
    except Exception as e:
        logger.exception(f"Error processing message: {e}")
        return f"I encountered an error processing your request. Please try again."


def handle(message: str) -> str:
    """
    Handle user message by routing through the hybrid router.
    The router uses deterministic handlers for clear booking intents,
    guideline RAG for medical/informational questions, and falls back to
    the LangGraph booking agent for ambiguous cases.

    `booking_context` is passed through and persisted in session state so
    multi-turn booking flows (choosing a hospital, providing patient info)
    survive across Streamlit reruns.

    Args:
        message: User input message

    Returns:
        Response string from the appropriate handler
    """
    if not message or not message.strip():
        return "I didn't receive a message. How can I help you?"

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(handle_async(message))
    finally:
        loop.close()


# =========================================================
# 🩺 UI
# =========================================================
st.set_page_config(
    page_title="AI Health Assistant",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 AI Health Assistant (Production Ready)")

# Health check status
health = run_health_check()
if health["hospital_data"] != "ok":
    st.warning("Hospital data may not be fully loaded")

file = st.file_uploader("Upload Medical Report", type=["pdf"])

if file:
    # Validate file first
    is_valid, error_msg = validate_pdf_file(file)

    if not is_valid:
        st.error(error_msg)
        logger.warning(f"File validation failed: {error_msg}")
    elif st.session_state.uploaded_file_bytes is None or st.session_state.uploaded_file_name != file.name:
        # New or different file: store bytes and re‑parse
        file_bytes = file.getvalue()
        st.session_state.uploaded_file_bytes = file_bytes
        st.session_state.uploaded_file_name = file.name

        try:
            with st.spinner("Analyzing your medical report..."):
                extractor = LabValueExtractor()
                text = extractor.load_pdf(file_bytes)
                parsed = extractor.extract(text)
                st.session_state.analysis = analyze_values(parsed)
                logger.info(f"Successfully processed file: {file.name}")
        except Exception as e:
            logger.exception(f"Error processing file {file.name}")
            st.error("Failed to process the medical report. Please try another file.")
            st.session_state.analysis = None

if st.session_state.analysis:
    if isinstance(st.session_state.analysis, dict) and st.session_state.analysis:
        score, triage = calculate_risk_score(st.session_state.analysis)

        color = "green" if score < 30 else "orange" if score < 70 else "red"
        st.progress(score / 100)
        st.success(f"**{triage}** | Risk Score: {score}/100")

        summary = generate_summary()
        if summary:
            st.info(summary)
    else:
        st.error("Unable to generate analysis from the provided report.")


# =========================================================
# 💬 CHAT UI
# =========================================================
for c in st.session_state.chat:
    if isinstance(c, dict) and "role" in c and "msg" in c:
        with st.chat_message(c["role"]):
            st.write(c["msg"])

msg = st.chat_input("Ask something about your health or book an appointment...")

if msg:
    # Sanitize message before adding to chat
    sanitized_msg = sanitize_user_text(msg)

    st.session_state.chat.append({"role": "user", "msg": sanitized_msg})

    reply = handle(sanitized_msg)

    st.session_state.chat.append({"role": "assistant", "msg": reply})

    st.rerun()


# =========================================================
# 📌 SIDEBAR
# =========================================================
st.sidebar.title("📋 Appointments")
st.sidebar.markdown("---")

if st.session_state.appointments:
    for a in st.session_state.appointments:
        if isinstance(a, dict):
            st.sidebar.write(f"🏥 **{a.get('hospital', 'Unknown')}**")
            st.sidebar.write(f"📅 {a.get('time', 'Unknown time')}")
            st.sidebar.markdown("---")
else:
    st.sidebar.info("No appointments scheduled")

# Clear chat button
if st.sidebar.button("Clear Chat"):
    st.session_state.chat = []
    st.session_state.booking_context = {}
    st.rerun()

# Debug controls (development only)
if hasattr(st, "secrets") and "debug" in st.secrets:
    st.sidebar.markdown("---")
    st.sidebar.write("**Debug Info**")
    st.sidebar.write(f"Chat messages: {len(st.session_state.chat)}")
    st.sidebar.write(f"Hospitals loaded: {len(hospitals)}")