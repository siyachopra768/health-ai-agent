

import streamlit as st
import json
import re
import hashlib

from parser import LabValueExtractor
from utils import get_llm, analyze_values, calculate_risk_score

# =========================================================
# 🧠 LLM (GROQ)
# =========================================================
llm = get_llm()

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""
You are a medical AI assistant.

Use:
- Patient medical report
- Hospital database
- Conversation history

========================
HISTORY
========================
{history}

========================
REPORT
========================
{report}

========================
HOSPITALS
========================
{hospitals}

========================
USER
========================
{input}

RULES:
- First check: is the USER message actually about the patient's health, report, symptoms, or hospital/appointment topics?
- If the USER message is unrelated to health/reports/appointments, respond only with: "I can only help with your medical report, symptoms, or hospital appointments. Please ask something related to that."
- Do NOT summarize or reference the report unless the question is actually about it.
- Use report if available and the question is relevant to it
- Suggest doctor type first if needed
- Use only given hospitals
- Be concise and helpful

IMPORTANT: Output ONLY your final answer to the user. Do NOT repeat, echo, or reference the section headers (HISTORY, REPORT, HOSPITALS, USER) or any part of this prompt in your response.

""")

chain = prompt | llm


# =========================================================
# 🏥 LOAD DATA
# =========================================================
with open("hospitals.json", "r") as f:
    hospitals = json.load(f)


# =========================================================
# 🧠 SESSION STATE
# =========================================================
for key, default in {
    "analysis": None,
    "pending_hospitals": None,
    "appointments": [],
    "chat": []
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# =========================================================
# 🔧 HELPERS
# =========================================================
def clean_response(text):
    for marker in ["HISTORY\n===","REPORT\n===","HOSPITALS\n===","USER\n==="]:
        if marker in text:
            text = text.split(marker)[-1]
    return text.strip()        






def extract_time(text):
    match = re.search(r'(\d{1,2})\s*(am|pm)', text.lower())
    return f"{match.group(1)} {match.group(2).upper()}" if match else None


def extract_choice(msg, options):
    if not options:
        return None

    match = re.search(r'\b(\d+)\b', msg)
    if match:
        idx = int(match.group(1)) - 1
        if 0 <= idx < len(options):
            return options[idx]["name"]

    for h in options:
        if h["name"].lower() in msg.lower():
            return h["name"]

    return None


def get_report_context():
    data = st.session_state.analysis
    if not data:
        return "No report uploaded."

    return "\n".join(
        [f"{k}: {v['value']} ({v['status']})" for k, v in data.items()]
    )


def get_hospital_context():
    return "\n".join(
        [f"{h['name']} | ⭐{h['rating']} | {', '.join(h.get('specialties', []))}" for h in hospitals]
    )


def get_chat_context():
    return "\n".join(
        [f"{m['role']}: {m['msg']}" for m in st.session_state.chat[-6:]]
    )


def detect_intent(msg):
    msg = msg.lower()

    if "cancel" in msg:
        return "cancel"

    if any(k in msg for k in ["book", "appointment", "schedule"]):
        return "book"

    if "risk" in msg:
        return "risk"

    if any(k in msg for k in ["pain", "fever", "weak", "hemoglobin", "report", "doctor"]):
        return "medical"

    

    if any(k in msg for k in ["delete all", "clear chat", "clear conversation", "reset chat"]):
        return "clear_chat"

    if "cancel" in msg:
        return "cancel"
    

    return "chat"

# =========================================================
# 🧠 DEFICIENCY SUMMARY (NEW FEATURE)
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
# 🧠 MAIN AGENT ENGINE
# =========================================================
def handle(message):

    msg = message.lower().strip()
    intent = detect_intent(msg)

    # =====================================================
    # ❌ CANCEL (HIGHEST PRIORITY)
    # =====================================================
    if intent == "cancel":
        st.session_state.appointments.clear()
        st.session_state.pending_hospitals = None
        return "❌ Appointment cancelled successfully."

    # =====================================================
    # 🏥 BOOK STEP 1
    # =====================================================
    if intent == "book":
        st.session_state.pending_hospitals = hospitals

        return "🏥 Available Hospitals:\n\n" + "\n".join(
            [f"{i+1}. {h['name']} ⭐{h['rating']}" for i, h in enumerate(hospitals)]
        ) + "\n\n👉 Reply like: '1 at 5 pm'"

    # =====================================================
    # 🏥 BOOK STEP 2
    # =====================================================
    if st.session_state.pending_hospitals:

        hospital = extract_choice(msg, st.session_state.pending_hospitals)
        time = extract_time(msg)

        if hospital and time:
            st.session_state.appointments.append({
                "hospital": hospital,
                "time": time
            })
            st.session_state.pending_hospitals = None
            return f"✅ Appointment booked at {hospital} at {time}"

        return "👉 Reply like: '1 at 5 pm'"

    # =====================================================
    # 📊 RISK
    # =====================================================
    if intent == "risk" and st.session_state.analysis:
        score, triage = calculate_risk_score(st.session_state.analysis)
        return f"{triage}\nRisk Score: {score}/100"


    if intent == "clear_chat":
        if not st.session_state.chat:
            return "chat empty"

        st.session_state.chat.clear()
        return "🗑️ Conversation cleared."

    # =====================================================
    if intent == "medical":

        if not st.session_state.analysis:
            return "📄 Please upload your medical report first."

        try:
            reply = chain.invoke({
                "input": message,
                "report": get_report_context(),
                "hospitals": get_hospital_context(),
                "history": get_chat_context()
            }).content

            # ⭐ ADD SUMMARY INSIDE CHAT RESPONSE
            summary = generate_summary()
            if summary:
                reply += "\n\n🧠 Summary: " + summary

            return reply

        except:
            return "⚠️ Unable to process request."

    # =====================================================
    # 💬 CHAT MODE
    # =====================================================
    try:
        return chain.invoke({
            "input": message,
            "report": get_report_context(),
            "hospitals": get_hospital_context(),
            "history": get_chat_context()
        }).content
        return clean_response(reply)
    except:
        return "👋 I can help with reports, doctors, or appointments."


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

    # ⭐ SUMMARY DISPLAY
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