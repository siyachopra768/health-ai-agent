"""
booking_handlers.py — Deterministic booking flow executors
Pure Python, zero LLM calls. Orchestrates API calls for booking flows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from .intent_router import Intent
from .appointment_agent import BookingAPIClient

# ── Logging ────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ── Data Classes ───────────────────────────────────────────────────────────────
@dataclass(slots=True)
class HandlerResult:
    """Result of a handler execution."""
    success: bool
    message: str
    data: dict | None = None
    requires_clarification: bool = False
    clarification_prompt: str | None = None


def _snapshot_intent(intent: Intent, context: dict) -> None:
    """Save intent fields into context so a later turn can rebuild an
    equivalent Intent. Intent is frozen (and slotted), so it can't carry
    state across turns itself -- context (backed by st.session_state) is
    where multi-turn booking state actually lives.
    """
    context["pending_intent"] = {
        "specialty": intent.specialty,
        "city": intent.city,
        "date": intent.date,
        "time": intent.time,
        "appointment_id": intent.appointment_id,
        "patient_email": intent.patient_email,
    }


# ── Handler Class ──────────────────────────────────────────────────────────────
class BookingHandler:
    """
    Deterministic booking flow executor.
    Handles: search, book, cancel, reschedule, list.
    """

    def __init__(self, api_client: BookingAPIClient | None = None):
        self.api = api_client or BookingAPIClient()

    # ── Search ────────────────────────────────────────────────────────────────
    async def handle_search(self, intent: Intent, context: dict) -> HandlerResult:
        """Search hospitals by specialty and/or city."""
        try:
            result = await self.api.search_hospitals(intent.specialty, intent.city)
            hospitals = result.get("hospitals", [])

            if not hospitals:
                return HandlerResult(
                    success=True,
                    message="No hospitals found matching your criteria.",
                    data={"hospitals": [], "count": 0}
                )

            lines = [f"Found {len(hospitals)} hospital(s):"]
            for h in hospitals:
                specs = ", ".join(h.get("specialties", []))
                lines.append(
                    f"  • {h['name']} ({h['city']}) ⭐{h['rating']} — "
                    f"{specs} — ₹{h['consultation_fee']}"
                )

            return HandlerResult(
                success=True,
                message="\n".join(lines),
                data={"hospitals": hospitals, "count": len(hospitals)}
            )

        except Exception as e:
            logger.exception("Search failed")
            return HandlerResult(
                success=False,
                message=f"Search error: {e}"
            )

    # ── Book ──────────────────────────────────────────────────────────────────
    async def handle_book(self, intent: Intent, context: dict) -> HandlerResult:
        """
        Execute booking flow.

        `context` is a mutable dict (backed by st.session_state in app.py)
        that accumulates hospital_name / patient_name / patient_email across
        turns. Intent itself is frozen and rebuilt fresh from each raw
        message, so it can never hold state gathered mid-flow -- context is
        the only thing that persists between this call and the next one.
        """
        missing = intent.missing_required
        if missing:
            # Persist the partial intent so the router can resume the booking
            # across turns even if the user's follow-up isn't keyword-rich.
            _snapshot_intent(intent, context)
            return HandlerResult(
                success=False,
                message=f"To book, I need: {', '.join(missing)}.",
                requires_clarification=True,
                clarification_prompt=f"Please provide: {', '.join(missing)}"
            )

        hospital_name = context.get("hospital_name")

        # If specialty provided but no hospital chosen yet, search first
        if intent.specialty and not hospital_name:
            search_result = await self.api.search_hospitals(intent.specialty, intent.city)
            hospitals = search_result.get("hospitals", [])

            if not hospitals:
                return HandlerResult(
                    success=False,
                    message=f"No hospitals found for {intent.specialty}"
                    + (f" in {intent.city}" if intent.city else "")
                )

            if len(hospitals) == 1:
                hospital_name = hospitals[0]["name"]
                context["hospital_name"] = hospital_name
            else:
                # Multiple matches - need user to choose. Remember both the
                # candidate list and the intent so the router can resolve
                # the next reply ("2" / a hospital name) without having to
                # re-derive specialty/date/time from scratch.
                context["pending_hospital_choices"] = hospitals
                _snapshot_intent(intent, context)

                lines = ["Multiple hospitals found. Please choose one:"]
                for i, h in enumerate(hospitals, 1):
                    specs = ", ".join(h.get("specialties", []))
                    lines.append(f"  {i}. {h['name']} ({h['city']}) ⭐{h['rating']} — {specs}")

                return HandlerResult(
                    success=True,
                    message="\n".join(lines),
                    requires_clarification=True,
                    clarification_prompt="Reply with hospital number or name",
                    data={"hospitals": hospitals}
                )

        if not hospital_name:
            _snapshot_intent(intent, context)
            return HandlerResult(
                success=False,
                message="Please specify a hospital.",
                requires_clarification=True,
                clarification_prompt="Which hospital would you like to book at?"
            )

        # Check slots
        try:
            slots_result = await self.api.get_available_slots(hospital_name, intent.date)
            available_slots = slots_result.get("available_slots", [])

            if intent.time not in available_slots:
                context["hospital_name"] = hospital_name
                _snapshot_intent(intent, context)
                return HandlerResult(
                    success=False,
                    message=f"Slot {intent.time} on {intent.date} not available at {hospital_name}.",
                    data={"available_slots": available_slots},
                    requires_clarification=True,
                    clarification_prompt=f"Available slots: {', '.join(available_slots)}. Choose one."
                )

        except Exception as e:
            logger.warning("Could not check slots: %s", e)
            # Proceed anyway - API will validate

        # Need patient name and email -- neither lives on Intent, so pull
        # from context (patient_email may also come from the intent itself,
        # since the email regex extractor runs on every message).
        patient_name = context.get("patient_name")
        patient_email = context.get("patient_email") or intent.patient_email

        if not patient_name or not patient_email:
            missing_fields = []
            if not patient_name:
                missing_fields.append("name")
            if not patient_email:
                missing_fields.append("email")

            context["hospital_name"] = hospital_name
            context["awaiting_patient_info"] = True
            _snapshot_intent(intent, context)

            return HandlerResult(
                success=False,
                message=f"Need your {' and '.join(missing_fields)} to complete booking.",
                requires_clarification=True,
                clarification_prompt=f"Please share your {' and '.join(missing_fields)}."
            )

        # Execute booking
        try:
            result = await self.api.book_appointment(
                hospital_name=hospital_name,
                specialty=intent.specialty,
                date=intent.date,
                time=intent.time,
                patient_name=patient_name,
                patient_email=patient_email
            )

            appt = result.get("appointment", {})
            context.clear()  # booking complete -- drop accumulated state

            return HandlerResult(
                success=True,
                message=(
                    f"✅ Appointment booked!\n"
                    f"Appointment ID: {appt.get('id', 'N/A')}\n"
                    f"{appt.get('hospital')} — {appt.get('specialty')} — "
                    f"{appt.get('date')} {appt.get('time')}"
                ),
                data={"appointment": appt}
            )

        except Exception as e:
            logger.exception("Booking failed")
            return HandlerResult(
                success=False,
                message=f"Booking failed: {e}"
            )

    # ── Cancel ────────────────────────────────────────────────────────────────
    async def handle_cancel(self, intent: Intent, context: dict) -> HandlerResult:
        """Cancel an appointment."""
        if not intent.appointment_id:
            return HandlerResult(
                success=False,
                message="Please provide the appointment ID to cancel.",
                requires_clarification=True,
                clarification_prompt="What is the appointment ID?"
            )

        try:
            result = await self.api.cancel_appointment(intent.appointment_id)
            return HandlerResult(
                success=True,
                message=f"❌ {result.get('message', 'Appointment cancelled')}",
                data=result.get("appointment")
            )
        except Exception as e:
            logger.exception("Cancel failed")
            return HandlerResult(
                success=False,
                message=f"Cancellation failed: {e}"
            )

    # ── Reschedule ────────────────────────────────────────────────────────────
    async def handle_reschedule(self, intent: Intent, context: dict) -> HandlerResult:
        """Reschedule an appointment."""
        if not intent.appointment_id:
            return HandlerResult(
                success=False,
                message="Please provide the appointment ID to reschedule.",
                requires_clarification=True,
                clarification_prompt="What is the appointment ID?"
            )

        if not intent.date and not intent.time:
            return HandlerResult(
                success=False,
                message="Please provide new date and/or time.",
                requires_clarification=True,
                clarification_prompt="What is the new date and/or time?"
            )

        try:
            result = await self.api.reschedule_appointment(
                intent.appointment_id,
                intent.date,
                intent.time
            )
            appt = result.get("appointment", {})
            return HandlerResult(
                success=True,
                message=(
                    f"📅 Appointment rescheduled!\n"
                    f"New: {appt.get('hospital')} — {appt.get('specialty')} — "
                    f"{appt.get('date')} {appt.get('time')}"
                ),
                data={"appointment": appt}
            )
        except Exception as e:
            logger.exception("Reschedule failed")
            return HandlerResult(
                success=False,
                message=f"Reschedule failed: {e}"
            )

    # ── List ──────────────────────────────────────────────────────────────────
    async def handle_list(self, intent: Intent, context: dict) -> HandlerResult:
        """List upcoming appointments."""
        try:
            result = await self.api.get_upcoming_appointments(
                patient_email=intent.patient_email,
                days_ahead=30
            )
            appointments = result.get("appointments", [])

            if not appointments:
                return HandlerResult(
                    success=True,
                    message="No upcoming appointments found.",
                    data={"appointments": [], "count": 0}
                )

            lines = [f"Upcoming appointments ({len(appointments)}):"]
            for a in appointments:
                lines.append(
                    f"  • {a['id']} — {a['hospital']} — {a['specialty']} — "
                    f"{a['date']} {a['time']} — {a['patient_name']} ({a['patient_email']})"
                )

            return HandlerResult(
                success=True,
                message="\n".join(lines),
                data={"appointments": appointments, "count": len(appointments)}
            )
        except Exception as e:
            logger.exception("List failed")
            return HandlerResult(
                success=False,
                message=f"Failed to list appointments: {e}"
            )

    # ── Main Dispatch ─────────────────────────────────────────────────────────
    async def handle(self, intent: Intent, context: dict) -> HandlerResult:
        """Route intent to appropriate handler."""
        if not intent.is_actionable():
            return HandlerResult(
                success=False,
                message="I couldn't understand that request clearly.",
                requires_clarification=True
            )

        handlers = {
            "search": self.handle_search,
            "book": self.handle_book,
            "cancel": self.handle_cancel,
            "reschedule": self.handle_reschedule,
            "list": self.handle_list,
        }

        handler = handlers.get(intent.action)
        if not handler:
            return HandlerResult(
                success=False,
                message=f"Unknown action: {intent.action}"
            )

        return await handler(intent, context)


# ── Convenience Function ──────────────────────────────────────────────────────
async def execute_booking_flow(
    intent: Intent,
    context: dict | None = None,
    api_client: BookingAPIClient | None = None
) -> HandlerResult:
    """Standalone function to execute a booking flow."""
    handler = BookingHandler(api_client)
    return await handler.handle(intent, context if context is not None else {})