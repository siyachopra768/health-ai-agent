"""
reminder_scheduler.py — APScheduler-based appointment reminder service
Checks /appointments/upcoming every 30 min, sends email reminders via SMTP.
Deduplicates sent reminders using in-memory set (replace with Redis/DB for production).
"""

from __future__ import annotations

import os
import smtplib
import logging
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional

import httpx
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger


# ── Configuration ──────────────────────────────────────────────────────────────
BOOKING_API_BASE = os.getenv("BOOKING_API_BASE", "http://localhost:8001")
CHECK_INTERVAL_MINUTES = int(os.getenv("REMINDER_CHECK_INTERVAL_MINUTES", "30"))
REMINDER_HOURS_BEFORE = int(os.getenv("REMINDER_HOURS_BEFORE", "24"))  # send reminder 24h before

# SMTP config (never hardcoded — all from env)
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
SMTP_FROM_EMAIL = os.getenv("SMTP_FROM_EMAIL", SMTP_USER)
SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "true").lower() == "true"

# In-memory deduplication (replace with Redis/DB for production)
# Key: (appointment_id, reminder_type) where reminder_type in {"24h", "1h"}
_sent_reminders: set[tuple[str, str]] = set()


# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# ── SMTP Validation ────────────────────────────────────────────────────────────
def validate_smtp_config() -> bool:
    """Check all required SMTP env vars are set."""
    required = ["SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD"]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        logger.error("Missing SMTP env vars: %s", missing)
        return False
    return True


# ── Email Sending ──────────────────────────────────────────────────────────────
def send_email(to_email: str, subject: str, body: str) -> bool:
    """Send email via SMTP. Returns True on success."""
    if not validate_smtp_config():
        return False

    msg = MIMEMultipart()
    msg["From"] = SMTP_FROM_EMAIL
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            if SMTP_USE_TLS:
                server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
        logger.info("Email sent to %s: %s", to_email, subject)
        return True
    except Exception as e:
        logger.exception("Failed to send email to %s: %s", to_email, e)
        return False


def build_reminder_email(appointment: dict, hours_before: int) -> tuple[str, str]:
    """Build subject and body for reminder email."""
    subject = f"⏰ Appointment Reminder: {appointment['specialty']} at {appointment['hospital']}"

    body = f"""Dear {appointment['patient_name']},

This is a reminder for your upcoming appointment:

🏥 Hospital: {appointment['hospital']}
🩺 Specialty: {appointment['specialty']}
📅 Date: {appointment['date']}
🕐 Time: {appointment['time']}
🆔 Appointment ID: {appointment['id']}

This reminder is sent {hours_before} hour(s) before your appointment.

Please arrive 15 minutes early. Bring any relevant medical reports.

If you need to reschedule or cancel, please contact us.

Best regards,
Health AI Agent
"""
    return subject, body


# ── Reminder Logic ─────────────────────────────────────────────────────────────
def should_send_reminder(appointment: dict, hours_before: int) -> Optional[str]:
    """
    Determine if a reminder should be sent for this appointment.
    Returns reminder_type ("24h" or "1h") if should send, None otherwise.
    """
    try:
        appt_dt = datetime.strptime(f"{appointment['date']} {appointment['time']}", "%Y-%m-%d %H:%M")
    except (KeyError, ValueError):
        return None

    now = datetime.now()
    hours_until = (appt_dt - now).total_seconds() / 3600

    # Check if within the reminder window (±30 min tolerance for scheduler interval)
    tolerance = 0.5  # 30 minutes
    if hours_before == 24:
        if 24 - tolerance <= hours_until <= 24 + tolerance:
            return "24h"
    elif hours_before == 1:
        if 1 - tolerance <= hours_until <= 1 + tolerance:
            return "1h"
    return None


def is_reminder_sent(appointment_id: str, reminder_type: str) -> bool:
    """Check if reminder already sent (deduplication)."""
    return (appointment_id, reminder_type) in _sent_reminders


def mark_reminder_sent(appointment_id: str, reminder_type: str) -> None:
    """Mark reminder as sent."""
    _sent_reminders.add((appointment_id, reminder_type))


async def fetch_upcoming_appointments() -> list[dict]:
    """Fetch upcoming appointments from booking API."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Fetch next 2 days to catch 24h and 1h reminders
        resp = await client.get(
            f"{BOOKING_API_BASE}/appointments/upcoming",
            params={"days_ahead": 2}
        )
        resp.raise_for_status()
        return resp.json().get("appointments", [])


async def check_and_send_reminders() -> None:
    """Main job: fetch appointments, check for due reminders, send emails."""
    logger.info("Running reminder check...")

    if not validate_smtp_config():
        logger.warning("SMTP not configured, skipping reminder check")
        return

    try:
        appointments = await fetch_upcoming_appointments()
    except Exception as e:
        logger.exception("Failed to fetch appointments: %s", e)
        return

    sent_count = 0
    for appt in appointments:
        if appt.get("status") != "confirmed":
            continue

        for hours_before in (24, 1):
            reminder_type = should_send_reminder(appt, hours_before)
            if not reminder_type:
                continue

            if is_reminder_sent(appt["id"], reminder_type):
                logger.debug("Reminder %s already sent for %s", reminder_type, appt["id"])
                continue

            subject, body = build_reminder_email(appt, hours_before)
            if send_email(appt["patient_email"], subject, body):
                mark_reminder_sent(appt["id"], reminder_type)
                sent_count += 1

    logger.info("Reminder check complete. Sent %d reminder(s).", sent_count)


# ── Scheduler Setup ────────────────────────────────────────────────────────────
def create_scheduler() -> BackgroundScheduler:
    """Create and configure the background scheduler."""
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        check_and_send_reminders,
        trigger=IntervalTrigger(minutes=CHECK_INTERVAL_MINUTES),
        id="reminder_check",
        name="Check and send appointment reminders",
        replace_existing=True,
        max_instances=1,  # prevent overlapping runs
    )
    return scheduler


# ── Entry Point ────────────────────────────────────────────────────────────────
def main() -> None:
    """Run the reminder scheduler as a standalone service."""
    if not validate_smtp_config():
        logger.error("SMTP configuration incomplete. Set SMTP_HOST, SMTP_USER, SMTP_PASSWORD.")
        logger.info("Example: SMTP_HOST=smtp.gmail.com SMTP_PORT=587 SMTP_USER=me@gmail.com SMTP_PASSWORD=app_password")
        return

    logger.info("Starting reminder scheduler (check every %d min)...", CHECK_INTERVAL_MINUTES)
    logger.info("Reminder windows: 24h and 1h before appointment")

    scheduler = create_scheduler()
    scheduler.start()

    try:
        # Keep alive
        import signal
        import sys

        def shutdown(signum, frame):
            logger.info("Shutting down...")
            scheduler.shutdown()
            sys.exit(0)

        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)

        # Run once immediately on startup, then on interval
        import asyncio
        asyncio.run(check_and_send_reminders())

        # Block forever
        signal.pause()
    except KeyboardInterrupt:
        scheduler.shutdown()
        logger.info("Scheduler stopped.")


if __name__ == "__main__":
    main()