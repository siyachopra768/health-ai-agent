"""
api.py  —  FastAPI backend for Health AI Agent
Run with:  uvicorn api:app --reload

Endpoints:
  POST /upload-report     → upload PDF, get lab analysis + risk score
  POST /chat              → send a message, get AI reply
  GET  /hospitals         → list all hospitals
  POST /book-appointment  → book an appointment
  GET  /appointments      → view all booked appointments
  DELETE /appointments    → cancel all appointments
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os
import io

# ── Import your existing modules (no changes needed in them) ──────────────────
from parser import load_pdf, extract_lab_values
from utils import get_llm, analyze_values, calculate_risk_score
from langchain_core.prompts import ChatPromptTemplate

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Health AI Agent API",
    description="REST API for medical report analysis, risk scoring, and hospital booking",
    version="1.0.0"
)

# Allow Streamlit (or any frontend) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── LLM + Prompt (same as app.py) ────────────────────────────────────────────
llm = get_llm()

prompt = ChatPromptTemplate.from_template("""
You are a medical AI assistant.
Use the patient's report, hospital database, and conversation history.

HISTORY: {history}
REPORT:  {report}
HOSPITALS: {hospitals}
USER: {input}

Rules:
- Use report data if available
- Suggest doctor type before hospital if needed
- Only use the given hospitals
- Be concise and helpful
""")

chain = prompt | llm

# ── Load hospital data ────────────────────────────────────────────────────────
with open("hospitals.json", "r") as f:
    HOSPITALS = json.load(f)

# ── In-memory session state (per server instance) ─────────────────────────────
# NOTE: This resets on server restart. Good enough for now.
session = {
    "analysis": None,       # lab analysis results
    "appointments": [],     # list of booked appointments
    "chat_history": [],     # last N messages for LLM context
}

# ── Pydantic models (these define what JSON your API accepts/returns) ─────────

class ChatRequest(BaseModel):
    message: str

class AppointmentRequest(BaseModel):
    hospital_name: str
    time: str           # e.g. "5 PM"

# ── Helper: format data for LLM context ──────────────────────────────────────

def get_report_context() -> str:
    if not session["analysis"]:
        return "No report uploaded yet."
    return "\n".join(
        f"{k}: {v['value']} ({v['status']})"
        for k, v in session["analysis"].items()
    )

def get_hospital_context() -> str:
    return "\n".join(
        f"{h['name']} | ⭐{h['rating']} | {', '.join(h.get('specialties', []))}"
        for h in HOSPITALS
    )

def get_chat_context() -> str:
    return "\n".join(
        f"{m['role']}: {m['content']}"
        for m in session["chat_history"][-6:]  # last 3 turns
    )

# =============================================================================
# ENDPOINTS
# =============================================================================

# ── 1. Upload PDF report ──────────────────────────────────────────────────────
@app.post("/upload-report")
async def upload_report(file: UploadFile = File(...)):
    """
    Upload a PDF medical report.
    Returns: extracted lab values, risk score, and triage category.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        # Read file bytes → pass to your existing load_pdf function
        contents = await file.read()
        pdf_file = io.BytesIO(contents)

        text = load_pdf(pdf_file)
        parsed = extract_lab_values(text)
        analysis = analyze_values(parsed)
        score, triage = calculate_risk_score(analysis)

        # Store in session for later chat context
        session["analysis"] = analysis

        # Build a clean summary to return
        abnormal = [
            {"test": k, "value": v["value"], "status": v["status"], "severity": v["severity"]}
            for k, v in analysis.items()
            if v["status"] != "normal"
        ]

        return {
            "risk_score": score,
            "triage": triage,
            "abnormal_values": abnormal,
            "all_values": analysis,
            "message": f"Report analysed. {len(abnormal)} abnormal value(s) found."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process report: {str(e)}")


# ── 2. Chat endpoint ──────────────────────────────────────────────────────────
@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Send a message to the health AI assistant.
    Uses uploaded report + hospital context if available.
    Returns: AI reply.
    """
    user_message = request.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        reply = chain.invoke({
            "input": user_message,
            "report": get_report_context(),
            "hospitals": get_hospital_context(),
            "history": get_chat_context(),
        }).content

        # Save to history
        session["chat_history"].append({"role": "user", "content": user_message})
        session["chat_history"].append({"role": "assistant", "content": reply})

        return {"reply": reply}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")


# ── 3. List hospitals ─────────────────────────────────────────────────────────
@app.get("/hospitals")
def get_hospitals():
    """
    Returns all hospitals in the database.
    """
    return {"hospitals": HOSPITALS}


# ── 4. Book appointment ───────────────────────────────────────────────────────
@app.post("/book-appointment")
def book_appointment(request: AppointmentRequest):
    """
    Book an appointment at a specific hospital at a given time.
    Example body: { "hospital_name": "City Care Hospital", "time": "5 PM" }
    """
    # Check hospital exists
    hospital_names = [h["name"] for h in HOSPITALS]
    if request.hospital_name not in hospital_names:
        raise HTTPException(
            status_code=404,
            detail=f"Hospital '{request.hospital_name}' not found. Available: {hospital_names}"
        )

    appointment = {
        "hospital": request.hospital_name,
        "time": request.time
    }
    session["appointments"].append(appointment)

    return {
        "message": f"✅ Appointment booked at {request.hospital_name} at {request.time}",
        "appointment": appointment
    }


# ── 5. View appointments ──────────────────────────────────────────────────────
@app.get("/appointments")
def get_appointments():
    """
    Returns all booked appointments for this session.
    """
    return {"appointments": session["appointments"]}


# ── 6. Cancel all appointments ────────────────────────────────────────────────
@app.delete("/appointments")
def cancel_appointments():
    """
    Cancels all booked appointments.
    """
    session["appointments"].clear()
    return {"message": "❌ All appointments cancelled."}


# ── Health check (always useful) ──────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "message": "Health AI Agent API is running 🩺"}
