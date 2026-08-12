"""
appointment_agent.py — LangGraph appointment booking agent
Uses create_react_agent with tools wrapping booking_api endpoints.
Integrates with existing intent router alongside lab_explainer.
"""

from __future__ import annotations

import json
import os
from typing import Annotated, Any, Literal

import httpx
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field


# ── Configuration ──────────────────────────────────────────────────────────────
BOOKING_API_BASE = os.getenv("BOOKING_API_BASE", "http://localhost:8001")
TIMEOUT = 30.0


# ── HTTP Client ────────────────────────────────────────────────────────────────
class BookingAPIClient:
    """Thin wrapper around booking_api HTTP endpoints."""

    def __init__(self, base_url: str = BOOKING_API_BASE, timeout: float = TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def search_hospitals(self, specialty: str | None = None, city: str | None = None) -> dict:
        params = {}
        if specialty:
            params["specialty"] = specialty
        if city:
            params["city"] = city
        client = await self._get_client()
        resp = await client.get(f"{self.base_url}/hospitals/search", params=params)
        resp.raise_for_status()
        return resp.json()

    async def get_available_slots(self, hospital_name: str, date: str) -> dict:
        client = await self._get_client()
        resp = await client.get(
            f"{self.base_url}/hospitals/{hospital_name}/slots",
            params={"date": date}
        )
        resp.raise_for_status()
        return resp.json()

    async def book_appointment(
        self,
        hospital_name: str,
        specialty: str,
        date: str,
        time: str,
        patient_name: str,
        patient_email: str
    ) -> dict:
        client = await self._get_client()
        payload = {
            "hospital_name": hospital_name,
            "specialty": specialty,
            "date": date,
            "time": time,
            "patient_name": patient_name,
            "patient_email": patient_email,
        }
        resp = await client.post(f"{self.base_url}/appointments/book", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def reschedule_appointment(
        self,
        appointment_id: str,
        new_date: str | None = None,
        new_time: str | None = None
    ) -> dict:
        client = await self._get_client()
        payload = {"appointment_id": appointment_id}
        if new_date:
            payload["new_date"] = new_date
        if new_time:
            payload["new_time"] = new_time
        resp = await client.post(f"{self.base_url}/appointments/reschedule", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def cancel_appointment(self, appointment_id: str) -> dict:
        client = await self._get_client()
        resp = await client.post(
            f"{self.base_url}/appointments/cancel",
            json={"appointment_id": appointment_id}
        )
        resp.raise_for_status()
        return resp.json()

    async def get_upcoming_appointments(
        self,
        patient_email: str | None = None,
        days_ahead: int = 30
    ) -> dict:
        client = await self._get_client()
        params = {"days_ahead": days_ahead}
        if patient_email:
            params["patient_email"] = patient_email
        resp = await client.get(f"{self.base_url}/appointments/upcoming", params=params)
        resp.raise_for_status()
        return resp.json()


# Global client instance (reused across tool calls)
_api_client = BookingAPIClient()


# ── Tool Definitions ───────────────────────────────────────────────────────────
@tool
async def search_hospitals(
    specialty: Annotated[str | None, "Medical specialty to search for (e.g., Endocrinologist)"] = None,
    city: Annotated[str | None, "City to search in (e.g., Delhi)"] = None,
) -> str:
    """Search hospitals by specialty and/or city. Returns list of matching hospitals with details."""
    try:
        result = await _api_client.search_hospitals(specialty, city)
        hospitals = result.get("hospitals", [])
        if not hospitals:
            return "No hospitals found matching your criteria."
        lines = [f"Found {len(hospitals)} hospital(s):"]
        for h in hospitals:
            specs = ", ".join(h.get("specialties", []))
            lines.append(f"  • {h['name']} ({h['city']}) ⭐{h['rating']} — {specs} — ₹{h['consultation_fee']}")
        return "\n".join(lines)
    except httpx.HTTPStatusError as e:
        return f"API error: {e.response.status_code} - {e.response.text}"
    except Exception as e:
        return f"Error searching hospitals: {e}"


@tool
async def get_available_slots(
    hospital_name: Annotated[str, "Exact hospital name from search results"],
    date: Annotated[str, "Date in YYYY-MM-DD format (e.g., 2026-08-15)"],
) -> str:
    """Get available appointment slots for a hospital on a specific date."""
    try:
        result = await _api_client.get_available_slots(hospital_name, date)
        slots = result.get("available_slots", [])
        if not slots:
            return f"No available slots at {hospital_name} on {date}."
        return f"Available slots at {hospital_name} on {date}:\n" + "\n".join(f"  • {s}" for s in slots)
    except httpx.HTTPStatusError as e:
        return f"API error: {e.response.status_code} - {e.response.text}"
    except Exception as e:
        return f"Error getting slots: {e}"


@tool
async def book_appointment(
    hospital_name: Annotated[str, "Exact hospital name from search results"],
    specialty: Annotated[str, "Medical specialty (e.g., Endocrinologist)"],
    date: Annotated[str, "Date in YYYY-MM-DD format"],
    time: Annotated[str, "Time in HH:MM format (24-hour, e.g., 10:00)"],
    patient_name: Annotated[str, "Patient's full name"],
    patient_email: Annotated[str, "Patient's email for confirmation"],
) -> str:
    """Book an appointment. Returns confirmation with appointment ID."""
    try:
        result = await _api_client.book_appointment(
            hospital_name, specialty, date, time, patient_name, patient_email
        )
        appt = result.get("appointment", {})
        return (
            f"✅ {result.get('message', 'Appointment booked')}\n"
            f"Appointment ID: {appt.get('id', 'N/A')}\n"
            f"Details: {appt.get('hospital')} — {appt.get('specialty')} — {appt.get('date')} {appt.get('time')}"
        )
    except httpx.HTTPStatusError as e:
        return f"Booking failed: {e.response.status_code} - {e.response.text}"
    except Exception as e:
        return f"Error booking appointment: {e}"


@tool
async def reschedule_appointment(
    appointment_id: Annotated[str, "Appointment ID from booking confirmation"],
    new_date: Annotated[str | None, "New date in YYYY-MM-DD format (optional)"] = None,
    new_time: Annotated[str | None, "New time in HH:MM format (optional)"] = None,
) -> str:
    """Reschedule an existing appointment to a new date and/or time."""
    if not new_date and not new_time:
        return "Error: Must provide at least new_date or new_time."
    try:
        result = await _api_client.reschedule_appointment(appointment_id, new_date, new_time)
        appt = result.get("appointment", {})
        return (
            f"📅 {result.get('message', 'Appointment rescheduled')}\n"
            f"New Details: {appt.get('hospital')} — {appt.get('specialty')} — {appt.get('date')} {appt.get('time')}"
        )
    except httpx.HTTPStatusError as e:
        return f"Reschedule failed: {e.response.status_code} - {e.response.text}"
    except Exception as e:
        return f"Error rescheduling appointment: {e}"


@tool
async def cancel_appointment(
    appointment_id: Annotated[str, "Appointment ID from booking confirmation"],
) -> str:
    """Cancel an existing appointment."""
    try:
        result = await _api_client.cancel_appointment(appointment_id)
        return f"❌ {result.get('message', 'Appointment cancelled')}"
    except httpx.HTTPStatusError as e:
        return f"Cancellation failed: {e.response.status_code} - {e.response.text}"
    except Exception as e:
        return f"Error cancelling appointment: {e}"


@tool
async def list_upcoming_appointments(
    patient_email: Annotated[str | None, "Filter by patient email (optional)"] = None,
    days_ahead: Annotated[int, "Days ahead to include (default 30)"] = 30,
) -> str:
    """List upcoming confirmed appointments. Optionally filter by patient email."""
    try:
        result = await _api_client.get_upcoming_appointments(patient_email, days_ahead)
        appointments = result.get("appointments", [])
        if not appointments:
            return "No upcoming appointments found."
        lines = [f"Upcoming appointments ({len(appointments)}):"]
        for a in appointments:
            lines.append(
                f"  • {a['id']} — {a['hospital']} — {a['specialty']} — "
                f"{a['date']} {a['time']} — {a['patient_name']} ({a['patient_email']})"
            )
        return "\n".join(lines)
    except httpx.HTTPStatusError as e:
        return f"API error: {e.response.status_code} - {e.response.text}"
    except Exception as e:
        return f"Error listing appointments: {e}"


# ── Agent Creation ─────────────────────────────────────────────────────────────
TOOLS = [
    search_hospitals,
    get_available_slots,
    book_appointment,
    reschedule_appointment,
    cancel_appointment,
    list_upcoming_appointments,
]


SYSTEM_PROMPT = """You are an appointment booking assistant for a healthcare system.

Your capabilities:
- Search hospitals by specialty and city
- Check available appointment slots
- Book, reschedule, and cancel appointments
- List upcoming appointments

Guidelines:
- Always confirm hospital name, specialty, date, time, patient name, and email before booking
- If user mentions "tomorrow", "next week", etc., convert to YYYY-MM-DD (today is {today})
- For rescheduling/cancellation, ask for appointment ID if not provided
- Be concise and helpful
- If a tool returns an error, explain it to the user and suggest next steps
- Call ONLY ONE tool per response. Wait for the result before calling another.
- After getting tool results, respond to the user directly without calling more tools unless the user asks for something new.
"""

def create_appointment_agent(model_name: str = "llama-3.1-8b-instant", temperature: float = 0):
    """Create the LangGraph react agent for appointment booking."""
    llm = ChatGroq(
        model_name=model_name,
        temperature=temperature,
    )

    # Inject current date into system prompt
    from datetime import date
    today = date.today().isoformat()
    system_prompt = SYSTEM_PROMPT.format(today=today)

    agent = create_react_agent(
        model=llm,
        tools=TOOLS,
        prompt=system_prompt,
    )
    return agent


# ── Standalone Runner (for testing) ────────────────────────────────────────────
async def run_agent(message: str, agent=None) -> str:
    """Run the agent with a single message and return the response."""
    if agent is None:
        agent = create_appointment_agent()

    result = await agent.ainvoke({"messages": [("user", message)]})
    # Extract the last AI message
    for msg in reversed(result["messages"]):
        if msg.type == "ai":
            return msg.content
    return "No response generated"


if __name__ == "__main__":
    import asyncio

    async def test():
        agent = create_appointment_agent()
        print("Appointment Agent ready. Type 'quit' to exit.\n")
        while True:
            user_input = input("You: ")
            if user_input.lower() in ("quit", "exit", "q"):
                break
            response = await run_agent(user_input, agent)
            print(f"Agent: {response}\n")

    asyncio.run(test())