"""
booking_api.py — FastAPI backend for appointment booking
Run with:  uvicorn booking_api:app --reload

Endpoints:
  GET  /hospitals/search?specialty=Endocrinologist&city=Delhi  → search hospitals
  GET  /hospitals/{hospital_name}/slots?date=2026-08-15       → get available slots
  POST /appointments/book                                     → book appointment
  POST /appointments/reschedule                               → reschedule appointment
  POST /appointments/cancel                                   → cancel appointment
  GET  /appointments/upcoming                                 → list upcoming appointments
  GET  /appointments/{appointment_id}                         → get appointment details
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ── Configuration ──────────────────────────────────────────────────────────────
DEFAULT_SLOTS = [
    "09:00", "09:30", "10:00", "10:30", "11:00", "11:30",
    "12:00", "12:30", "13:00", "13:30", "14:00", "14:30",
    "15:00", "15:30", "16:00", "16:30", "17:00"
]

HOSPITALS_PATH = os.path.join(os.path.dirname(__file__), "hospitals.json")


# ── Data Loading ───────────────────────────────────────────────────────────────
with open(HOSPITALS_PATH, "r") as f:
    HOSPITALS: list[dict] = json.load(f)

HOSPITAL_NAMES = {h["name"] for h in HOSPITALS}
HOSPITAL_BY_NAME = {h["name"]: h for h in HOSPITALS}


# ── In-Memory Store ────────────────────────────────────────────────────────────
# NOTE: Resets on server restart. For production, replace with persistent DB.
appointments_store: dict[str, dict] = {}


# ── Enums & Models ─────────────────────────────────────────────────────────────
class AppointmentStatus(str, Enum):
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    RESCHEDULED = "rescheduled"


class BookAppointmentRequest(BaseModel):
    hospital_name: str = Field(..., description="Hospital name from /hospitals")
    specialty: str = Field(..., description="Medical specialty")
    date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="YYYY-MM-DD")
    time: str = Field(..., pattern=r"^\d{2}:\d{2}$", description="HH:MM (24-hour)")
    patient_name: str = Field(..., min_length=1)
    patient_email: str = Field(..., pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class RescheduleAppointmentRequest(BaseModel):
    appointment_id: str
    new_date: Optional[str] = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    new_time: Optional[str] = Field(None, pattern=r"^\d{2}:\d{2}$")


class CancelAppointmentRequest(BaseModel):
    appointment_id: str


class AppointmentResponse(BaseModel):
    id: str
    hospital: str
    specialty: str
    date: str
    time: str
    patient_name: str
    patient_email: str
    status: str
    created_at: str


# ── Helper Functions ───────────────────────────────────────────────────────────
def filter_hospitals(specialty: Optional[str] = None, city: Optional[str] = None) -> list[dict]:
    """Filter hospitals by specialty and/or city (case-insensitive)."""
    results = HOSPITALS
    if specialty:
        spec_lower = specialty.lower()
        results = [h for h in results if spec_lower in {s.lower() for s in h.get("specialties", [])}]
    if city:
        city_lower = city.lower()
        results = [h for h in results if h.get("city", "").lower() == city_lower]
    return results


def get_booked_times(hospital_name: str, date: str) -> set[str]:
    """Return set of booked times for a hospital on a given date (confirmed only)."""
    active_statuses = {AppointmentStatus.CONFIRMED,AppointmentStatus.RESCHEDULED}
    return {
        a["time"]
        for a in appointments_store.values()
        if a["hospital"] == hospital_name
        and a["date"] == date
        and a["status"] in active_statuses
    }


def get_available_slots(hospital_name: str, date: str) -> list[str]:
    """Return available slots for hospital/date (DEFAULT_SLOTS minus booked)."""
    return [s for s in DEFAULT_SLOTS if s not in get_booked_times(hospital_name, date)]


def parse_date(date_str: str) -> datetime:
    """Parse YYYY-MM-DD string to datetime.date, raise ValueError if invalid."""
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def validate_future_date(date_str: str) -> None:
    """Validate date format and that it's not in the past."""
    try:
        if parse_date(date_str) < datetime.now().date():
            raise ValueError("Date cannot be in the past")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


def validate_time(time_str: str) -> None:
    """Validate HH:MM format."""
    try:
        datetime.strptime(time_str, "%H:%M")
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid time format. Use HH:MM (24-hour).") from e


def get_appointment_or_404(appointment_id: str) -> dict:
    """Fetch appointment or raise 404."""
    appt = appointments_store.get(appointment_id)
    if not appt:
        raise HTTPException(status_code=404, detail=f"Appointment '{appointment_id}' not found.")
    return appt


def check_slot_available(hospital_name: str, date: str, time: str, exclude_id: Optional[str] = None) -> None:
    """Raise 409 if slot is booked (optionally excluding one appointment ID)."""
    booked = get_booked_times(hospital_name, date)
    if exclude_id and exclude_id in appointments_store:
        booked.discard(appointments_store[exclude_id]["time"])
    if time in booked:
        available = [s for s in DEFAULT_SLOTS if s not in booked]
        raise HTTPException(
            status_code=409,
            detail=f"Slot {time} on {date} is not available. Available: {available}"
        )


def validate_hospital_and_specialty(hospital_name: str, specialty: str) -> dict:
    """Validate hospital exists and offers the specialty. Return hospital dict."""
    if hospital_name not in HOSPITAL_NAMES:
        raise HTTPException(
            status_code=404,
            detail=f"Hospital '{hospital_name}' not found. Available: {sorted(HOSPITAL_NAMES)}"
        )
    hospital = HOSPITAL_BY_NAME[hospital_name]
    if specialty.lower() not in {s.lower() for s in hospital.get("specialties", [])}:
        raise HTTPException(
            status_code=400,
            detail=f"Hospital '{hospital_name}' does not offer '{specialty}'. Specialties: {hospital.get('specialties', [])}"
        )
    return hospital


# ── FastAPI App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Health AI Agent - Booking API",
    description="REST API for hospital search, appointment booking, rescheduling, and cancellation",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/hospitals/search")
def search_hospitals(
    specialty: Optional[str] = Query(None, description="Medical specialty (e.g., Endocrinologist)"),
    city: Optional[str] = Query(None, description="City name (e.g., Delhi)")
):
    """Search hospitals by specialty and/or city."""
    results = filter_hospitals(specialty, city)
    return {"hospitals": results, "count": len(results)}


@app.get("/hospitals/{hospital_name}/slots")
def get_hospital_slots(
    hospital_name: str,
    date: str = Query(..., description="Date in YYYY-MM-DD format")
):
    """Get available appointment slots for a hospital on a specific date."""
    validate_future_date(date)
    if hospital_name not in HOSPITAL_NAMES:
        raise HTTPException(status_code=404, detail=f"Hospital '{hospital_name}' not found.")
    return {"hospital": hospital_name, "date": date, "available_slots": get_available_slots(hospital_name, date)}


@app.post("/appointments/book")
def book_appointment(request: BookAppointmentRequest):
    """Book an appointment at a specific hospital."""
    validate_hospital_and_specialty(request.hospital_name, request.specialty)
    validate_future_date(request.date)
    validate_time(request.time)
    check_slot_available(request.hospital_name, request.date, request.time)

    appointment_id = str(uuid.uuid4())[:8]
    now = datetime.now().isoformat()

    appointment = {
        "id": appointment_id,
        "hospital": request.hospital_name,
        "specialty": request.specialty,
        "date": request.date,
        "time": request.time,
        "patient_name": request.patient_name,
        "patient_email": request.patient_email,
        "status": AppointmentStatus.CONFIRMED,
        "created_at": now,
    }

    appointments_store[appointment_id] = appointment

    return {
        "message": (
            f"✅ Appointment booked at {request.hospital_name} for "
            f"{request.specialty} on {request.date} at {request.time}"
        ),
        "appointment": appointment
    }


@app.post("/appointments/reschedule")
def reschedule_appointment(request: RescheduleAppointmentRequest):
    """Reschedule an existing appointment."""
    appointment = get_appointment_or_404(request.appointment_id)

    if appointment["status"] == AppointmentStatus.CANCELLED:
        raise HTTPException(status_code=400, detail="Cannot reschedule a cancelled appointment.")

    new_date = request.new_date or appointment["date"]
    new_time = request.new_time or appointment["time"]

    validate_future_date(new_date)
    validate_time(new_time)

    if new_date != appointment["date"] or new_time != appointment["time"]:
        check_slot_available(appointment["hospital"], new_date, new_time, exclude_id=request.appointment_id)

    old_date, old_time = appointment["date"], appointment["time"]
    appointment["date"] = new_date
    appointment["time"] = new_time
    appointment["status"] = AppointmentStatus.RESCHEDULED
    appointment["rescheduled_at"] = datetime.now().isoformat()

    return {
        "message": f"📅 Appointment rescheduled from {old_date} {old_time} to {new_date} {new_time}",
        "appointment": appointment
    }


@app.post("/appointments/cancel")
def cancel_appointment(request: CancelAppointmentRequest):
    """Cancel an existing appointment."""
    appointment = get_appointment_or_404(request.appointment_id)

    if appointment["status"] == AppointmentStatus.CANCELLED:
        raise HTTPException(status_code=400, detail="Appointment already cancelled.")

    appointment["status"] = AppointmentStatus.CANCELLED
    appointment["cancelled_at"] = datetime.now().isoformat()

    return {
        "message": f"❌ Appointment {request.appointment_id} cancelled.",
        "appointment": appointment
    }


@app.get("/appointments/upcoming")
def get_upcoming_appointments(
    patient_email: Optional[str] = Query(None, description="Filter by patient email"),
    days_ahead: int = Query(30, ge=1, le=365, description="Days ahead to include")
):
    """List upcoming confirmed appointments. Optionally filter by patient email."""
    now = datetime.now()
    cutoff = now + timedelta(days=days_ahead)

    upcoming = []
    for appt in appointments_store.values():
        if appt["status"] != AppointmentStatus.CONFIRMED:
            continue
        appt_dt = datetime.strptime(f"{appt['date']} {appt['time']}", "%Y-%m-%d %H:%M")
        if not (now <= appt_dt <= cutoff):
            continue
        if patient_email and appt["patient_email"].lower() != patient_email.lower():
            continue
        upcoming.append(appt)

    upcoming.sort(key=lambda a: datetime.strptime(f"{a['date']} {a['time']}", "%Y-%m-%d %H:%M"))
    return {"appointments": upcoming, "count": len(upcoming)}


@app.get("/appointments/{appointment_id}")
def get_appointment(appointment_id: str):
    """Get details of a specific appointment."""
    return {"appointment": get_appointment_or_404(appointment_id)}


@app.get("/hospitals")
def get_all_hospitals():
    """Return all hospitals."""
    return {"hospitals": HOSPITALS}


@app.get("/")
def root():
    return {"status": "ok", "message": "Booking API is running 🩺"}