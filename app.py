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
from parser import LabValueExtractor
from utils import analyze_values, calculate_risk_score

# Import the hybrid router
from booking_appointment.hybrid_router import route_message

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

# ── Load hospital data ────────────────────────────────────────────────────────
with open("booking_appointment/hospitals.json", "r") as f:
    hospitals = json.load(f)


# =========================================================
# 🧠 SESSION STATE
# =========================================================
for key, default in {
    "analysis": None,
    "appointments": [],
    "chat": [],
    "booking_context": {},  # multi-turn booking state (hospital choice, patient info) -- see hybrid_router.route()
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# =========================================================
# 🧠 DEFICIENCY SUMMARY
# =========================================================
def generate_summary():
    data = st.session_state.analysis

    if not data:
        return None

    abnormal = [
        f"{k} ({v['status']})"
        for k, v in data.items()
        if v["status"] != "normal"
    ]

    if not abnormal:
        return "All parameters are within normal range."

    return "Key abnormalities detected: " + ", ".join(abnormal[:3]) + "."


# =========================================================
# 🧠 MAIN AGENT ENGINE — routes through the hybrid router
# =========================================================
def handle(message):
    """
    Handle user message by routing through the hybrid router.
    The router uses deterministic handlers for clear booking intents,
    guideline RAG for medical/informational questions, and falls back to
    the LangGraph booking agent for ambiguous cases.

    `booking_context` is passed through and persisted in session state so
    multi-turn booking flows (choosing a hospital, providing patient info)
    survive across Streamlit reruns.
    """
    if not message or not message.strip():
        return "I didn't receive a message. How can I help you?"

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            route_message(message, st.session_state.booking_context)
        )
    finally:
        loop.close()


# =========================================================
# 🩺 UI
# =========================================================
st.title("🩺 AI Health Assistant (Final Agent System)")

file = st.file_uploader("Upload Medical Report")

if file:
    file_bytes = file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()

    if st.session_state.analysis is None or st.session_state.get("last_file") != file.name:

        extractor = LabValueExtractor()
        text = extractor.load_pdf(file)
        parsed = extractor.extract(text)
        st.session_state.analysis = analyze_values(parsed)
        st.session_state.last_file_hash = file_hash

if st.session_state.analysis:
    score, triage = calculate_risk_score(st.session_state.analysis)

    st.success(f"{triage} | Risk Score: {score}/100")

    summary = generate_summary()
    if summary:
        st.info(summary)


# =========================================================
# 💬 CHAT UI
# =========================================================
for c in st.session_state.chat:
    with st.chat_message(c["role"]):
        st.write(c["msg"])

msg = st.chat_input("Ask something...")

if msg:
    st.session_state.chat.append({"role": "user", "msg": msg})

    reply = handle(msg)

    st.session_state.chat.append({"role": "assistant", "msg": reply})

    st.rerun()


# =========================================================
# 📌 SIDEBAR
# =========================================================
st.sidebar.title("Appointments")

for a in st.session_state.appointments:
    st.sidebar.write(f"🏥 {a['hospital']} at {a['time']}")