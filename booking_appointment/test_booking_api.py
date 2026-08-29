"""
test_booking_api.py — pytest suite for booking API endpoints and LangGraph tools
Following test_verifier.py pattern: mocked HTTP calls, deterministic assertions.
"""

from __future__ import annotations

import json
import pytest
from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi.testclient import TestClient
from booking_appointment.booking_api import app, appointments_store, HOSPITAL_NAMES, HOSPITAL_BY_NAME
from booking_appointment.appointment_agent import (
    search_hospitals,
    get_available_slots,
    book_appointment,
    reschedule_appointment,
    cancel_appointment,
    list_upcoming_appointments,
    BookingAPIClient,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_appointments():
    """Clear in-memory store before each test."""
    appointments_store.clear()
    yield
    appointments_store.clear()


@pytest.fixture
def tomorrow():
    return (date.today() + timedelta(days=1)).isoformat()


@pytest.fixture
def day_after():
    return (date.today() + timedelta(days=2)).isoformat()


@pytest.fixture
def mock_api_client():
    """Mock BookingAPIClient for tool testing."""
    with patch("appointment_agent._api_client", new_callable=AsyncMock) as mock:
        yield mock


# ── Helper: Create test appointment ───────────────────────────────────────────
def create_appointment(
    appointment_id: str = "test123",
    hospital: str = "AIIMS Delhi",
    specialty: str = "Endocrinologist",
    date_str: str = None,
    time_str: str = "10:00",
    status: str = "confirmed",
) -> dict:
    if date_str is None:
        date_str = (date.today() + timedelta(days=1)).isoformat()
    return {
        "id": appointment_id,
        "hospital": hospital,
        "specialty": specialty,
        "date": date_str,
        "time": time_str,
        "patient_name": "John Doe",
        "patient_email": "john@example.com",
        "status": status,
        "created_at": "2026-01-01T10:00:00",
    }


# ── Booking API Endpoint Tests ────────────────────────────────────────────────

class TestSearchHospitals:
    def test_search_by_specialty_and_city(self, client):
        resp = client.get("/hospitals/search?specialty=Endocrinologist&city=Delhi")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 3  # AIIMS, Apollo, Fortis
        for h in data["hospitals"]:
            assert "Endocrinologist" in h["specialties"]
            assert h["city"] == "Delhi"

    def test_search_by_city_only(self, client):
        resp = client.get("/hospitals/search?city=Gurgaon")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["hospitals"][0]["name"] == "Medanta Hospital Gurgaon"

    def test_search_by_specialty_only(self, client):
        resp = client.get("/hospitals/search?specialty=Hematologist")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2  # AIIMS, Medanta

    def test_search_no_filters(self, client):
        resp = client.get("/hospitals/search")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 7

    def test_search_no_results(self, client):
        resp = client.get("/hospitals/search?specialty=Cardiologist&city=Mumbai")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0
        assert data["hospitals"] == []


class TestGetAvailableSlots:
    def test_get_slots_success(self, client, tomorrow):
        resp = client.get(f"/hospitals/AIIMS%20Delhi/slots?date={tomorrow}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["hospital"] == "AIIMS Delhi"
        assert data["date"] == tomorrow
        assert len(data["available_slots"]) == 17  # All DEFAULT_SLOTS free

    def test_get_slots_hospital_not_found(self, client, tomorrow):
        resp = client.get(f"/hospitals/Fake%20Hospital/slots?date={tomorrow}")
        assert resp.status_code == 404

    def test_get_slots_past_date_rejected(self, client):
        past = (date.today() - timedelta(days=1)).isoformat()
        resp = client.get(f"/hospitals/AIIMS%20Delhi/slots?date={past}")
        assert resp.status_code == 400

    def test_get_slots_invalid_date_format(self, client):
        resp = client.get("/hospitals/AIIMS%20Delhi/slots?date=invalid")
        assert resp.status_code == 400


class TestBookAppointment:
    def test_book_success(self, client, tomorrow):
        payload = {
            "hospital_name": "AIIMS Delhi",
            "specialty": "Endocrinologist",
            "date": tomorrow,
            "time": "10:00",
            "patient_name": "John Doe",
            "patient_email": "john@example.com"
        }
        resp = client.post("/appointments/book", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "✅" in data["message"]
        appt = data["appointment"]
        assert appt["hospital"] == "AIIMS Delhi"
        assert appt["specialty"] == "Endocrinologist"
        assert appt["status"] == "confirmed"
        assert len(appointments_store) == 1

    def test_book_hospital_not_found(self, client, tomorrow):
        payload = {
            "hospital_name": "Fake Hospital",
            "specialty": "Endocrinologist",
            "date": tomorrow,
            "time": "10:00",
            "patient_name": "John Doe",
            "patient_email": "john@example.com"
        }
        resp = client.post("/appointments/book", json=payload)
        assert resp.status_code == 404

    def test_book_specialty_not_offered(self, client, tomorrow):
        payload = {
            "hospital_name": "AIIMS Delhi",
            "specialty": "Cardiologist",  # Not offered by AIIMS
            "date": tomorrow,
            "time": "10:00",
            "patient_name": "John Doe",
            "patient_email": "john@example.com"
        }
        resp = client.post("/appointments/book", json=payload)
        assert resp.status_code == 400

    def test_book_slot_conflict(self, client, tomorrow):
        # Book first appointment
        payload = {
            "hospital_name": "AIIMS Delhi",
            "specialty": "Endocrinologist",
            "date": tomorrow,
            "time": "10:00",
            "patient_name": "John Doe",
            "patient_email": "john@example.com"
        }
        client.post("/appointments/book", json=payload)

        # Try to book same slot
        payload["patient_email"] = "jane@example.com"
        resp = client.post("/appointments/book", json=payload)
        assert resp.status_code == 409
        assert "not available" in resp.json()["detail"]

    def test_book_past_date_rejected(self, client):
        past = (date.today() - timedelta(days=1)).isoformat()
        payload = {
            "hospital_name": "AIIMS Delhi",
            "specialty": "Endocrinologist",
            "date": past,
            "time": "10:00",
            "patient_name": "John Doe",
            "patient_email": "john@example.com"
        }
        resp = client.post("/appointments/book", json=payload)
        assert resp.status_code == 400

    def test_book_invalid_time_format(self, client, tomorrow):
        payload = {
            "hospital_name": "AIIMS Delhi",
            "specialty": "Endocrinologist",
            "date": tomorrow,
            "time": "25:00",  # Invalid hour
            "patient_name": "John Doe",
            "patient_email": "john@example.com"
        }
        resp = client.post("/appointments/book", json=payload)
        assert resp.status_code == 400

    def test_book_invalid_email(self, client, tomorrow):
        payload = {
            "hospital_name": "AIIMS Delhi",
            "specialty": "Endocrinologist",
            "date": tomorrow,
            "time": "10:00",
            "patient_name": "John Doe",
            "patient_email": "not-an-email"
        }
        resp = client.post("/appointments/book", json=payload)
        assert resp.status_code == 422  # Pydantic validation


class TestRescheduleAppointment:
    def test_reschedule_success(self, client, tomorrow, day_after):
        # Create appointment
        appt = create_appointment(date_str=tomorrow)
        appointments_store[appt["id"]] = appt

        resp = client.post("/appointments/reschedule", json={
            "appointment_id": appt["id"],
            "new_date": day_after,
            "new_time": "11:00"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "📅" in data["message"]
        assert data["appointment"]["date"] == day_after
        assert data["appointment"]["time"] == "11:00"
        assert data["appointment"]["status"] == "rescheduled"

    def test_reschedule_time_only(self, client, tomorrow):
        appt = create_appointment(date_str=tomorrow)
        appointments_store[appt["id"]] = appt

        resp = client.post("/appointments/reschedule", json={
            "appointment_id": appt["id"],
            "new_time": "15:30"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["appointment"]["date"] == tomorrow  # unchanged
        assert data["appointment"]["time"] == "15:30"

    def test_reschedule_date_only(self, client, tomorrow, day_after):
        appt = create_appointment(date_str=tomorrow)
        appointments_store[appt["id"]] = appt

        resp = client.post("/appointments/reschedule", json={
            "appointment_id": appt["id"],
            "new_date": day_after
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["appointment"]["date"] == day_after
        assert data["appointment"]["time"] == "10:00"  # unchanged

    def test_reschedule_not_found(self, client, tomorrow):
        resp = client.post("/appointments/reschedule", json={
            "appointment_id": "nonexistent",
            "new_date": tomorrow
        })
        assert resp.status_code == 404

    def test_reschedule_cancelled_appointment(self, client, tomorrow):
        appt = create_appointment(date_str=tomorrow, status="cancelled")
        appointments_store[appt["id"]] = appt

        resp = client.post("/appointments/reschedule", json={
            "appointment_id": appt["id"],
            "new_date": tomorrow
        })
        assert resp.status_code == 400
        assert "cancelled" in resp.json()["detail"]

    def test_reschedule_slot_conflict(self, client, tomorrow, day_after):
        # Existing appointment at 11:00 on day_after
        existing = create_appointment(appointment_id="existing1", date_str=day_after, time_str="11:00")
        appointments_store[existing["id"]] = existing

        # Try to reschedule another to same slot
        appt = create_appointment(appointment_id="move_me", date_str=tomorrow)
        appointments_store[appt["id"]] = appt

        resp = client.post("/appointments/reschedule", json={
            "appointment_id": appt["id"],
            "new_date": day_after,
            "new_time": "11:00"
        })
        assert resp.status_code == 409


class TestCancelAppointment:
    def test_cancel_success(self, client, tomorrow):
        appt = create_appointment(date_str=tomorrow)
        appointments_store[appt["id"]] = appt

        resp = client.post("/appointments/cancel", json={"appointment_id": appt["id"]})
        assert resp.status_code == 200
        data = resp.json()
        assert "❌" in data["message"]
        assert data["appointment"]["status"] == "cancelled"

    def test_cancel_not_found(self, client):
        resp = client.post("/appointments/cancel", json={"appointment_id": "nonexistent"})
        assert resp.status_code == 404

    def test_cancel_already_cancelled(self, client, tomorrow):
        appt = create_appointment(date_str=tomorrow, status="cancelled")
        appointments_store[appt["id"]] = appt

        resp = client.post("/appointments/cancel", json={"appointment_id": appt["id"]})
        assert resp.status_code == 400
        assert "already cancelled" in resp.json()["detail"]


class TestGetUpcomingAppointments:
    def test_list_upcoming(self, client, tomorrow):
        appt1 = create_appointment(appointment_id="a1", date_str=tomorrow, time_str="10:00")
        appt2 = create_appointment(appointment_id="a2", date_str=tomorrow, time_str="14:00")
        appointments_store[appt1["id"]] = appt1
        appointments_store[appt2["id"]] = appt2

        resp = client.get("/appointments/upcoming")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        # Sorted by time
        assert data["appointments"][0]["time"] == "10:00"
        assert data["appointments"][1]["time"] == "14:00"

    def test_list_upcoming_filter_email(self, client, tomorrow):
        appt1 = create_appointment(appointment_id="a1", patient_email="john@example.com")
        appt2 = create_appointment(appointment_id="a2", patient_email="jane@example.com")
        appointments_store[appt1["id"]] = appt1
        appointments_store[appt2["id"]] = appt2

        resp = client.get("/appointments/upcoming?patient_email=john@example.com")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["appointments"][0]["patient_email"] == "john@example.com"

    def test_list_excludes_cancelled(self, client, tomorrow):
        appt1 = create_appointment(appointment_id="a1", status="confirmed")
        appt2 = create_appointment(appointment_id="a2", status="cancelled")
        appointments_store[appt1["id"]] = appt1
        appointments_store[appt2["id"]] = appt2

        resp = client.get("/appointments/upcoming")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["appointments"][0]["id"] == "a1"

    def test_list_excludes_rescheduled(self, client, tomorrow):
        appt1 = create_appointment(appointment_id="a1", status="confirmed")
        appt2 = create_appointment(appointment_id="a2", status="rescheduled")
        appointments_store[appt1["id"]] = appt1
        appointments_store[appt2["id"]] = appt2

        resp = client.get("/appointments/upcoming")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1

    def test_list_days_ahead_limit(self, client, tomorrow):
        far_future = (date.today() + timedelta(days=60)).isoformat()
        appt1 = create_appointment(appointment_id="a1", date_str=tomorrow)
        appt2 = create_appointment(appointment_id="a2", date_str=far_future)
        appointments_store[appt1["id"]] = appt1
        appointments_store[appt2["id"]] = appt2

        resp = client.get("/appointments/upcoming?days_ahead=7")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1


class TestGetAppointment:
    def test_get_appointment(self, client, tomorrow):
        appt = create_appointment(appointment_id="test123", date_str=tomorrow)
        appointments_store[appt["id"]] = appt

        resp = client.get("/appointments/test123")
        assert resp.status_code == 200
        data = resp.json()
        assert data["appointment"]["id"] == "test123"

    def test_get_appointment_not_found(self, client):
        resp = client.get("/appointments/nonexistent")
        assert resp.status_code == 404


# ── LangGraph Tool Tests (with mocked HTTP) ───────────────────────────────────

class TestSearchHospitalsTool:
    @pytest.mark.asyncio
    async def test_search_hospitals_success(self, mock_api_client):
        mock_api_client.search_hospitals.return_value = {
            "hospitals": [
                {"name": "AIIMS Delhi", "city": "Delhi", "specialties": ["Endocrinologist"], "rating": 4.8}
            ],
            "count": 1
        }
        result = await search_hospitals.ainvoke({"specialty": "Endocrinologist", "city": "Delhi"})
        assert "AIIMS Delhi" in result
        assert "Endocrinologist" in result
        mock_api_client.search_hospitals.assert_called_once_with("Endocrinologist", "Delhi")

    @pytest.mark.asyncio
    async def test_search_hospitals_empty(self, mock_api_client):
        mock_api_client.search_hospitals.return_value = {"hospitals": [], "count": 0}
        result = await search_hospitals.ainvoke({"specialty": "Cardiologist", "city": "Mumbai"})
        assert "No hospitals found" in result

    @pytest.mark.asyncio
    async def test_search_hospitals_error(self, mock_api_client):
        import httpx
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_api_client.search_hospitals.side_effect = httpx.HTTPStatusError(
            "Server Error", request=MagicMock(), response=mock_response
        )
        result = await search_hospitals.ainvoke({"specialty": "Endocrinologist"})
        assert "API error" in result


class TestGetAvailableSlotsTool:
    @pytest.mark.asyncio
    async def test_get_slots_success(self, mock_api_client, tomorrow):
        mock_api_client.get_available_slots.return_value = {
            "hospital": "AIIMS Delhi",
            "date": tomorrow,
            "available_slots": ["09:00", "10:00", "11:00"]
        }
        result = await get_available_slots.ainvoke({"hospital_name": "AIIMS Delhi", "date": tomorrow})
        assert "AIIMS Delhi" in result
        assert "09:00" in result
        assert "10:00" in result

    @pytest.mark.asyncio
    async def test_get_slots_empty(self, mock_api_client, tomorrow):
        mock_api_client.get_available_slots.return_value = {
            "hospital": "AIIMS Delhi",
            "date": tomorrow,
            "available_slots": []
        }
        result = await get_available_slots.ainvoke({"hospital_name": "AIIMS Delhi", "date": tomorrow})
        assert "No available slots" in result


class TestBookAppointmentTool:
    @pytest.mark.asyncio
    async def test_book_success(self, mock_api_client, tomorrow):
        mock_api_client.book_appointment.return_value = {
            "message": "✅ Appointment booked",
            "appointment": {
                "id": "abc12345",
                "hospital": "AIIMS Delhi",
                "specialty": "Endocrinologist",
                "date": tomorrow,
                "time": "10:00"
            }
        }
        result = await book_appointment.ainvoke({
            "hospital_name": "AIIMS Delhi",
            "specialty": "Endocrinologist",
            "date": tomorrow,
            "time": "10:00",
            "patient_name": "John Doe",
            "patient_email": "john@example.com"
        })
        assert "✅" in result
        assert "abc12345" in result

    @pytest.mark.asyncio
    async def test_book_error(self, mock_api_client, tomorrow):
        import httpx
        mock_response = MagicMock()
        mock_response.status_code = 409
        mock_response.text = "Slot not available"
        mock_api_client.book_appointment.side_effect = httpx.HTTPStatusError(
            "Conflict", request=MagicMock(), response=mock_response
        )
        result = await book_appointment.ainvoke({
            "hospital_name": "AIIMS Delhi",
            "specialty": "Endocrinologist",
            "date": tomorrow,
            "time": "10:00",
            "patient_name": "John Doe",
            "patient_email": "john@example.com"
        })
        assert "Booking failed" in result


class TestRescheduleAppointmentTool:
    @pytest.mark.asyncio
    async def test_reschedule_success(self, mock_api_client, tomorrow):
        mock_api_client.reschedule_appointment.return_value = {
            "message": "📅 Appointment rescheduled",
            "appointment": {
                "id": "abc12345",
                "hospital": "AIIMS Delhi",
                "specialty": "Endocrinologist",
                "date": tomorrow,
                "time": "11:00"
            }
        }
        result = await reschedule_appointment.ainvoke({
            "appointment_id": "abc12345",
            "new_date": tomorrow,
            "new_time": "11:00"
        })
        assert "📅" in result
        assert "11:00" in result

    @pytest.mark.asyncio
    async def test_reschedule_missing_params(self, mock_api_client):
        result = await reschedule_appointment.ainvoke({"appointment_id": "abc12345"})
        assert "Must provide at least new_date or new_time" in result


class TestCancelAppointmentTool:
    @pytest.mark.asyncio
    async def test_cancel_success(self, mock_api_client):
        mock_api_client.cancel_appointment.return_value = {
            "message": "❌ Appointment cancelled"
        }
        result = await cancel_appointment.ainvoke({"appointment_id": "abc12345"})
        assert "❌" in result
        assert "cancelled" in result.lower()


class TestListUpcomingAppointmentsTool:
    @pytest.mark.asyncio
    async def test_list_success(self, mock_api_client, tomorrow):
        mock_api_client.get_upcoming_appointments.return_value = {
            "appointments": [
                {
                    "id": "a1",
                    "hospital": "AIIMS Delhi",
                    "specialty": "Endocrinologist",
                    "date": tomorrow,
                    "time": "10:00",
                    "patient_name": "John Doe",
                    "patient_email": "john@example.com"
                }
            ],
            "count": 1
        }
        result = await list_upcoming_appointments.ainvoke({})
        assert "Upcoming appointments" in result
        assert "a1" in result
        assert "AIIMS Delhi" in result

    @pytest.mark.asyncio
    async def test_list_empty(self, mock_api_client):
        mock_api_client.get_upcoming_appointments.return_value = {"appointments": [], "count": 0}
        result = await list_upcoming_appointments.ainvoke({})
        assert "No upcoming appointments" in result


# ── BookingAPIClient Tests ────────────────────────────────────────────────────

class TestBookingAPIClient:
    @pytest.mark.asyncio
    async def test_client_search_hospitals(self):
        with patch("httpx.AsyncClient.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"hospitals": [], "count": 0}
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp

            client = BookingAPIClient("http://test-api")
            result = await client.search_hospitals("Endocrinologist", "Delhi")
            assert result == {"hospitals": [], "count": 0}

    @pytest.mark.asyncio
    async def test_client_book_appointment(self):
        with patch("httpx.AsyncClient.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"message": "ok", "appointment": {"id": "123"}}
            mock_resp.raise_for_status = MagicMock()
            mock_post.return_value = mock_resp

            client = BookingAPIClient("http://test-api")
            result = await client.book_appointment(
                "Hospital", "Specialty", "2026-01-01", "10:00", "Patient", "email@test.com"
            )
            assert result["appointment"]["id"] == "123"


# ── Integration-style: Full flow ──────────────────────────────────────────────

class TestFullBookingFlow:
    def test_search_slots_book_cancel(self, client, tomorrow):
        # 1. Search
        resp = client.get("/hospitals/search?specialty=Endocrinologist&city=Delhi")
        assert resp.status_code == 200
        hospitals = resp.json()["hospitals"]
        assert len(hospitals) >= 1

        # 2. Get slots
        hospital_name = hospitals[0]["name"]
        resp = client.get(f"/hospitals/{hospital_name}/slots?date={tomorrow}")
        assert resp.status_code == 200
        slots = resp.json()["available_slots"]
        assert len(slots) > 0

        # 3. Book
        slot = slots[0]
        payload = {
            "hospital_name": hospital_name,
            "specialty": "Endocrinologist",
            "date": tomorrow,
            "time": slot,
            "patient_name": "Integration Test",
            "patient_email": "integration@test.com"
        }
        resp = client.post("/appointments/book", json=payload)
        assert resp.status_code == 200
        appt_id = resp.json()["appointment"]["id"]

        # 4. Verify in upcoming
        resp = client.get("/appointments/upcoming")
        assert resp.status_code == 200
        assert resp.json()["count"] == 1

        # 5. Cancel
        resp = client.post("/appointments/cancel", json={"appointment_id": appt_id})
        assert resp.status_code == 200

        # 6. Verify cancelled
        resp = client.get("/appointments/upcoming")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0


# ── Run with: python -m pytest test_booking_api.py -v ─────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])