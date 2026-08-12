"""
hybrid_router.py — Orchestrates deterministic routing with LLM fallback
Combines intent classifier, booking handlers, RAG retrieval, and the
LangGraph booking agent.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Optional

from .intent_router import Intent, classify_intent, extract_email
from .booking_handlers import BookingHandler, HandlerResult
from .appointment_agent import create_appointment_agent, run_agent

# HybridRetriever lives in the `rag` package at the project root, imported
# absolutely -- same convention app.py uses for `booking_appointment`
# (both are top-level dirs next to app.py, not nested in a shared package).
from rag.hybrid_retrieval import HybridRetriever

# ── Logging ────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# Messages containing any of these are treated as booking-flow-ish and skip
# RAG entirely, going straight to the deterministic handler / booking agent.
BOOKING_KEYWORDS = (
    "book", "appointment", "schedule", "cancel", "reschedule",
    "hospital", "doctor", "slot", "specialist", "clinic",
)

CHOICE_NUMBER_RE = re.compile(r"\b(\d+)\b")


def _looks_like_booking(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in BOOKING_KEYWORDS)


# ── Orchestrator ───────────────────────────────────────────────────────────────
@dataclass
class HybridRouter:
    """
    Main routing logic:
    1. Resolve any pending multi-turn booking clarification (hospital
       choice / patient info) before touching the classifier at all.
    2. Classify intent with deterministic classifier.
    3. If high confidence and actionable -> use deterministic handlers.
    4. Else, if it doesn't look like a booking message -> try RAG.
    5. Else -> fall back to the LangGraph booking agent.
    """
    api_base: str = "http://localhost:8001"
    llm_model: str = "llama-3.1-8b-instant"
    llm_temperature: float = 0.0

    def __post_init__(self):
        self.handler = BookingHandler()  # Will create its own API client
        self.retriever = HybridRetriever()
        self.llm_agent = create_appointment_agent(
            model_name=self.llm_model,
            temperature=self.llm_temperature
        )
        logger.info("HybridRouter initialized with LLM agent")

    # ── Multi-turn booking resolution ───────────────────────────────────────
    def _reconstruct_intent(self, context: dict) -> Optional[Intent]:
        """Rebuild an Intent from the snapshot booking_handlers.py stores in
        context. Intent is frozen, so this is the only way to "continue" a
        booking across turns instead of re-deriving everything from a reply
        that may only contain a hospital number or a bare name/email.
        """
        snapshot = context.get("pending_intent")
        if not snapshot:
            return None
        return Intent(
            action="book",
            specialty=snapshot.get("specialty"),
            city=snapshot.get("city"),
            date=snapshot.get("date"),
            time=snapshot.get("time"),
            appointment_id=snapshot.get("appointment_id"),
            patient_email=snapshot.get("patient_email"),
            confidence=1.0,
            missing_required=[],
        )

    async def _continue_booking(self, context: dict) -> str:
        """Re-invoke handle_book with the remembered intent now that context
        has been updated (hospital chosen / patient info supplied)."""
        intent = self._reconstruct_intent(context)
        if intent is None:
            context.clear()
            return "Something went wrong tracking your booking details — could you restart your booking request?"

        result = await self.handler.handle_book(intent, context)
        if result.success:
            return result.message
        if result.requires_clarification:
            return result.clarification_prompt or result.message
        return result.message

    async def _resolve_hospital_choice(self, message: str, context: dict) -> str:
        hospitals = context.get("pending_hospital_choices", [])

        chosen = None
        m = CHOICE_NUMBER_RE.search(message)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(hospitals):
                chosen = hospitals[idx]["name"]
        if not chosen:
            for h in hospitals:
                if h["name"].lower() in message.lower():
                    chosen = h["name"]
                    break

        if not chosen:
            return "I didn't catch that — please reply with the hospital number or name from the list above."

        context["hospital_name"] = chosen
        context.pop("pending_hospital_choices", None)
        return await self._continue_booking(context)

    async def _resolve_patient_info(self, message: str, context: dict) -> str:
        email = extract_email(message)
        name = message.replace(email, "").strip(" ,.-") if email else message.strip()

        if email:
            context["patient_email"] = email
        if name:
            context["patient_name"] = name

        if not context.get("patient_name") or not context.get("patient_email"):
            missing = []
            if not context.get("patient_name"):
                missing.append("name")
            if not context.get("patient_email"):
                missing.append("email")
            return f"I still need your {' and '.join(missing)} to complete the booking."

        context["awaiting_patient_info"] = False
        return await self._continue_booking(context)

    # ── RAG ──────────────────────────────────────────────────────────────────
    async def _answer_from_guidelines(self, query: str) -> Optional[str]:
        """Run hybrid retrieval in a worker thread (it's synchronous /
        blocking) and format retrieved chunks into a plain excerpt-based
        answer. No extra LLM synthesis call for now -- deliberately the
        simplest version that works; a prose-synthesis pass over these
        excerpts is a natural next step, not a requirement to ship this.
        """
        results = await asyncio.to_thread(self.retriever.search, query, 3, 20, True)
        if not results:
            return None

        lines = ["Here's what the guidelines say:"]
        for r in results:
            title = r["metadata"].get("title", r["metadata"].get("id", "Guideline"))
            excerpt = r["content"][:400].strip()
            lines.append(f"\n**{title}**\n{excerpt}")
        return "\n".join(lines)

    # ── Main entry point ────────────────────────────────────────────────────
    async def route(self, message: str, context: Optional[dict] = None) -> str:
        """
        Route user message to appropriate handler.
        Returns response string to send to user.

        `context` is a mutable dict the caller should persist across turns
        (e.g. st.session_state["booking_context"] in a Streamlit app) --
        it's how multi-turn booking state (chosen hospital, patient info)
        survives between messages, since Intent itself is stateless.
        """
        if context is None:
            context = {}

        if not message or not message.strip():
            return "I didn't receive a message. How can I help you?"

        logger.info("Routing message: %s", message[:100])

        # Step 0: resolve any pending multi-turn booking clarification
        # before reclassifying -- a reply like "2" or "Siya, siya@x.com"
        # would not classify as a booking intent on its own.
        if context.get("pending_hospital_choices"):
            return await self._resolve_hospital_choice(message, context)
        if context.get("awaiting_patient_info"):
            return await self._resolve_patient_info(message, context)

        # Step 1: classify intent (deterministic, zero LLM)
        intent = classify_intent(message)
        logger.info("Classifier result: %s", intent)

        # Step 2: try deterministic handler
        if intent.is_actionable():
            logger.info("Using deterministic handler for: %s", intent.action)
            try:
                result = await self.handler.handle(intent, context)

                if result.success:
                    logger.info("Deterministic handler succeeded")
                    return result.message
                elif result.requires_clarification:
                    logger.info("Deterministic handler needs clarification")
                    return result.clarification_prompt or result.message
                else:
                    logger.warning("Deterministic handler failed: %s", result.message)
                    # Fall through to LLM for complex error cases

            except Exception:
                logger.exception("Deterministic handler error")
                # Fall through to LLM

        # Step 3: for non-booking-looking messages, try RAG before handing
        # the question to the booking-only LLM agent (which has no medical
        # knowledge -- only hospital/slot/booking tools).
        if not _looks_like_booking(message):
            try:
                rag_answer = await self._answer_from_guidelines(message)
                if rag_answer:
                    logger.info("Answered from guideline RAG")
                    return rag_answer
            except Exception:
                logger.exception("RAG retrieval failed, falling back to LLM agent")

        # Step 4: fall back to the booking LLM agent
        logger.info("Falling back to LangGraph agent")
        try:
            result = await asyncio.wait_for(
                run_agent(message, self.llm_agent),
                timeout=30.0
            )
            logger.info("LLM agent succeeded")
            return result
        except asyncio.TimeoutError:
            logger.error("LLM agent timed out")
            return "I'm taking too long to process that. Please try a simpler request or try again later."
        except Exception as e:
            logger.exception("LLM agent failed")
            return f"I encountered an error processing your request: {str(e)}"


# ── Singleton ──────────────────────────────────────────────────────────────────
# A HybridRouter is expensive to build (spins up an httpx client, a LangGraph
# agent, and a Chroma/BM25-backed retriever). Building one per message -- as
# the old route_message() did -- adds real latency/cost per turn and leaks
# an unclosed httpx.AsyncClient every time. Build it once, reuse it.
_router_singleton: Optional[HybridRouter] = None


def get_router() -> HybridRouter:
    global _router_singleton
    if _router_singleton is None:
        _router_singleton = HybridRouter()
    return _router_singleton


async def route_message(message: str, context: Optional[dict] = None) -> str:
    """Standalone function to route a message using the shared router."""
    router = get_router()
    return await router.route(message, context)


# ── Self-Test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    async def test():
        router = get_router()

        test_cases = [
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

        for msg in test_cases:
            print(f"\nInput: {msg}")
            context: dict = {}
            try:
                response = await router.route(msg, context)
                print(f"Response: {response[:200]}...")
            except Exception as e:
                print(f"Error: {e}")

    asyncio.run(test())