import logging
import re
import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, List, Set
from sqlalchemy import or_, and_

# Database and Model Imports
from receptionist.database import (
    SessionLocal,
    get_company_details,
    get_available_slots,
    schedule_meeting,
    get_employee_by_name,
)
from receptionist.models import Employee, Visitor, Meeting, ReceptionLog
from models.groq_processor import BASE_SYSTEM_PROMPT, GroqProcessor
from services.notify_slack import send_slack_arrival, clear_session as clear_slack_cache
from services.calendar_service import schedule_google_meeting_background
from services.slack_reply_store import pop_reply

# Logger Configuration
logger = logging.getLogger(__name__)

# Constants - STRICTLY PRESERVED
AI_NAME = "Jarvis"
COMPANY_NAME = "Sharp Software Development India Private Limited."
SESSION_TIMEOUT_SECONDS = 300

NAME_BLACKLIST = {
    "jarvis",
    "davis",
    "darwis",
    "darvis",
    "jarves",
    "dervis",
    "bruce",
    "chalves",
    "travis",
    "unknown",
    "none",
    "null",
    "it",
    "alexa",
    "dadfish",
    "jadfish",
}

WAKE_WORDS = [
    "hey jarvis",
    "hi jarvis",
    "wake_word_triggered",
    "hey charles",
    "hey elvis",
    "hey jadfish",
    "hey dadfish",
    "hey travis",
]

PRONOUNS = {
    "him",
    "her",
    "them",
    "he",
    "she",
    "they",
    "it",
    "that person",
    "someone",
    "this guy",
    "this person",
}

FOOD_DELIVERY_KEYWORDS = {"zomato", "swiggy", "food", "bistro", "blinkit", "danzo"}
PACKAGE_DELIVERY_KEYWORDS = {
    "amazon",
    "flipkart",
    "ajio",
    "savana",
    "delivery",
    "parcel",
    "courier",
    "toing",
}

# Semantic Role-Group Mapping
SEMANTIC_MAP = {
    "fix": "Operations",
    "leak": "Operations",
    "broken": "Operations",
    "maintenance": "Operations",
    "ac": "Operations",
    "repair": "Operations",
    "hiring": "HR",
    "recruit": "HR",
    "interview": "HR",
    "money": "Finance",
    "account": "Finance",
    "invoice": "Finance",
    "bill": "Finance",
    "payment": "Finance",
    "tech": "IT Support",
    "computer": "IT Support",
    "system": "IT Support",
    "internet": "IT Support",
    "software": "IT Support",
}


class State:
    INIT = "INIT"
    COLLECTING = "COLLECTING"
    COMPLETED = "COMPLETED"
    TERMINATED = "TERMINATED"


_client_sessions: Dict[str, Dict[str, Any]] = {}

# ─────────────────────────────────────────────────────────────────────────────
# SESSION MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────


def get_session_state(client_id: str) -> Dict[str, Any]:
    now = datetime.utcnow()
    if client_id in _client_sessions:
        last_active = _client_sessions[client_id].get("last_active")
        if (
            last_active
            and (now - last_active).total_seconds() > SESSION_TIMEOUT_SECONDS
        ):
            clear_session_state(client_id)
    if client_id not in _client_sessions:
        _client_sessions[client_id] = _fresh_state()
    _client_sessions[client_id]["last_active"] = now
    return _client_sessions[client_id]


def clear_session_state(client_id: str) -> None:
    if client_id in _client_sessions:
        session_id = _client_sessions[client_id].get("session_id")
        if session_id:
            clear_slack_cache(session_id)
        del _client_sessions[client_id]
    try:
        GroqProcessor.get_instance().reset_history(client_id)
    except Exception as e:
        logger.error(f"Hardware reset failed for {client_id}: {e}")


def _fresh_state() -> Dict[str, Any]:
    return {
        "session_id": str(__import__("uuid").uuid4()),
        "conv_state": "INIT",
        "last_active": __import__("datetime").datetime.utcnow(),
        "visitor_name": None,
        "visitor_email": None,
        "visitor_type": "Visitor/Guest",
        "greeting_sent": False,
        "meeting_with_raw": None,
        "meeting_with_resolved": None,
        "host_details": None,
        "is_employee": False,
        "purpose": None,
        "scheduling_active": False,
        "sched_employee_name": None,
        "sched_employee_email": None,
        "sched_date": None,
        "sched_time": None,
        "sched_purpose": None,
        "identity_updated": False,
        "notified_hosts": set(),
        "greeted": False,
        "force_admin": False,
        "awaiting_slack_reply": False,  # ← NEW
    }


# ─────────────────────────────────────────────────────────────────────────────
# NORMALIZATION & LOOKUP HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def _get_time_greeting() -> str:
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "Good Morning"
    elif 12 <= hour < 17:
        return "Good Afternoon"
    else:
        return "Good Evening"


def _is_jarvis(name: str) -> bool:
    if not name:
        return False
    return name.lower().strip().replace(".", "") in NAME_BLACKLIST


def _determine_visitor_type(text: str, purpose: str, current_type: str) -> str:
    combined = f"{text} {purpose}".lower()
    if "intern" in combined:
        return "Intern"
    if re.search(r"\b(interview|candidate)\b", combined):
        return "Interviewee"
    if any(k in combined for k in FOOD_DELIVERY_KEYWORDS):
        return "Food Delivery"
    if any(k in combined for k in PACKAGE_DELIVERY_KEYWORDS):
        return "Package Delivery"
    if re.search(
        r"\b(vendor|electrician|plumber|maintenance|ac|fix|leak|broken)\b", combined
    ):
        return "Contractor/Vendor"
    if re.search(r"\b(client|customer|demo)\b", combined):
        return "Client"
    return current_type or "Visitor/Guest"


def _format_descriptive_purpose(
    state: Dict[str, Any], event_type: str = "VISIT"
) -> str:
    v_type = state.get("visitor_type", "Visitor")
    host = (
        state.get("meeting_with_resolved")
        or state.get("sched_employee_name")
        or "Administration Team"
    )
    base_purpose = (
        state.get("purpose") or state.get("sched_purpose") or "General Business"
    )
    if event_type == "SCHEDULED":
        return f"SCHEDULED: {base_purpose} with {host} (Set for {state.get('sched_date')} @ {state.get('sched_time')})"
    return f"{v_type.upper()}: {base_purpose} for {host}"


def _normalize_date(raw: str) -> Optional[str]:
    if not raw:
        return None
    s = str(raw).lower()
    today = datetime.now().date()
    match = re.search(
        r"next\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", s
    )
    if match:
        weekday_map = {
            "mon": 0,
            "tue": 1,
            "wed": 2,
            "thu": 3,
            "fri": 4,
            "sat": 5,
            "sun": 6,
        }
        target_day = weekday_map[match.group(1)[:3]]
        days_ahead = (target_day - today.weekday() + 7) % 7
        if days_ahead == 0:
            days_ahead = 7
        return (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    if "today" in s:
        return today.strftime("%Y-%m-%d")
    if "tomorrow" in s:
        return (today + timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        return datetime.strptime(s, "%Y-%m-%d").date().strftime("%Y-%m-%d")
    except:
        return None


def _normalize_time(raw: str) -> Optional[str]:
    if not raw:
        return None
    s = (
        str(raw)
        .lower()
        .replace("p.m.", "pm")
        .replace("a.m.", "am")
        .replace(".", "")
        .replace(" ", "")
    )
    m = re.match(r"^(\d{1,2})(?::(\d{2}))?(am|pm)$", s)
    if m:
        h, mn, mer = int(m.group(1)), int(m.group(2) or 0), m.group(3)
        if mer == "pm" and h != 12:
            h += 12
        if mer == "am" and h == 12:
            h = 0
        return f"{h:02d}:{mn:02d}"
    return s if re.match(r"^\d{2}:\d{2}$", s) else None


def _lookup_employee(search_term: str) -> Optional[Employee]:
    if not search_term or len(str(search_term)) < 2:
        return None
    clean = re.sub(
        r"\b(the|is|who|of|this|company|with|for|at|his|her|name|cabin|room|manager|engineer|lead|ceo)\b",
        "",
        str(search_term).lower(),
    ).strip()
    for key, dept in SEMANTIC_MAP.items():
        if key in clean:
            clean = dept
            break
    if clean in ["admin", "administration", "front desk", "anyone"]:
        return Employee(
            name="Administration Team", role="Support", location="Reception"
        )
    db = SessionLocal()
    try:
        emp = (
            db.query(Employee)
            .filter(
                or_(
                    Employee.role.ilike(f"%{clean}%"),
                    Employee.name.ilike(f"%{clean}%"),
                    Employee.department.ilike(f"%{clean}%"),
                )
            )
            .first()
        )
        return emp
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────────────────────
# PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────


def _finalize_meeting_and_log(state: Dict[str, Any]) -> bool:
    db = SessionLocal()
    try:
        v_name = state.get("visitor_name") or "Guest"
        host_name = state["sched_employee_name"]
        emp = _lookup_employee(host_name)
        narrative = _format_descriptive_purpose(state, "SCHEDULED")
        mid = schedule_meeting(
            v_name,
            "Visitor",
            host_name,
            state["sched_date"],
            state["sched_time"],
            narrative,
        )
        if not mid:
            return False
        visitor = db.query(Visitor).filter(Visitor.name.ilike(v_name)).first()
        if not visitor:
            visitor = Visitor(name=v_name)
            db.add(visitor)
            db.flush()
        log = ReceptionLog(
            visitor_id=visitor.id,
            employee_id=emp.id if emp and emp.id else None,
            person_type=state["visitor_type"],
            purpose=narrative,
            check_in_time=datetime.utcnow(),
        )
        db.add(log)
        db.commit()
        send_slack_arrival(
            host_name, v_name, state["visitor_type"], narrative, state["session_id"]
        )
        state["notified_hosts"].add(host_name)
        if state.get("sched_employee_email"):
            schedule_google_meeting_background(
                v_name,
                state["sched_employee_email"],
                state["sched_date"],
                state["sched_time"],
            )
        return True
    except Exception as e:
        logger.error(f"Finalization failed: {e}")
        return False
    finally:
        db.close()


def _commit_checkin(state: Dict[str, Any]) -> bool:
    db = SessionLocal()
    try:
        v_name = state.get("visitor_name") or "Guest"
        visitor = db.query(Visitor).filter(Visitor.name.ilike(v_name)).first()
        if not visitor:
            visitor = Visitor(name=v_name)
            db.add(visitor)
            db.flush()
        host_raw = state.get("meeting_with_resolved") or state.get("meeting_with_raw")
        host_emp = _lookup_employee(host_raw)
        log = ReceptionLog(
            visitor_id=visitor.id,
            employee_id=host_emp.id if host_emp and host_emp.id else None,
            person_type=state["visitor_type"],
            purpose=_format_descriptive_purpose(state),
            check_in_time=datetime.utcnow(),
        )
        db.add(log)
        db.commit()
        return True
    except Exception as e:
        logger.error(f"Commit error: {e}")
        return False
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────────────────────
# HANDLERS
# ─────────────────────────────────────────────────────────────────────────────


def _merge_checkin_entities(
    state: Dict[str, Any], entities: Dict[str, Any], user_query: str
) -> None:
    query_low = user_query.lower()
    v_name = entities.get("visitor_name")
    if v_name and not _is_jarvis(v_name):
        new_name = v_name.capitalize()
        emp_record = get_employee_by_name(new_name)
        if emp_record or "i am an employee" in query_low:
            state["is_employee"], state["visitor_type"] = True, "Employee"
        if state.get("visitor_name") and state["visitor_name"] != new_name:
            state["visitor_name"], state["identity_updated"] = new_name, True
        elif not state.get("visitor_name"):
            state["visitor_name"] = new_name

    state["visitor_type"] = _determine_visitor_type(
        user_query, entities.get("purpose", ""), state["visitor_type"]
    )
    target = (
        entities.get("employee_name")
        or entities.get("employee_role")
        or state.get("meeting_with_raw")
    )
    if target and not _is_jarvis(target):
        state["meeting_with_raw"] = target
        emp = _lookup_employee(target)
        if emp:
            state["meeting_with_resolved"] = state["sched_employee_name"] = emp.name
            state["sched_employee_email"] = emp.email
        else:
            state["meeting_with_resolved"] = target

    if entities.get("date"):
        state["sched_date"] = _normalize_date(str(entities["date"]))
    if entities.get("time"):
        state["sched_time"] = _normalize_time(str(entities["time"]))
    if entities.get("purpose"):
        state["purpose"] = state["sched_purpose"] = entities["purpose"]


async def _handle_availability_check(
    state: Dict[str, Any], query: str, client_id: str
) -> str:
    target = state.get("meeting_with_resolved") or query
    emp = _lookup_employee(target)
    if emp:
        slots = get_available_slots(emp.name, datetime.now().strftime("%Y-%m-%d"))
        slot_str = ", ".join(slots[:3]) if slots else "no more slots today"
        return await _llm_reply(
            f"Inform them {emp.name} is the {emp.role}. Free slots: {slot_str}.",
            state,
            query,
            client_id,
        )
    return await _llm_reply(
        "Apologize you couldn't check availability.", state, query, client_id
    )


async def _handle_directory_lookup(
    state: Dict[str, Any], query: str, client_id: str
) -> str:
    emp = _lookup_employee(query)
    if emp:
        state["meeting_with_resolved"] = emp.name
        return await _llm_reply(
            f"Tell them {emp.name} is the {emp.role} at {emp.location}.",
            state,
            query,
            client_id,
        )
    return await _llm_reply(
        "Apologize politely that you couldn't find them.", state, query, client_id
    )


async def _finalize_checkin_and_respond_UPDATED(state, query, client_id):
    if not state.get("visitor_name"):
        return await _llm_reply("Ask for their name politely.", state, query, client_id)

    current_host = state["meeting_with_resolved"] or "Administration Team"

    if current_host not in state["notified_hosts"]:
        from services.notify_slack import send_slack_arrival

        send_slack_arrival(
            current_host,
            state["visitor_name"],
            state["visitor_type"],
            state.get("purpose", "Arrival"),
            state["session_id"],
        )
        state["notified_hosts"].add(current_host)
        state["awaiting_slack_reply"] = True  # ← NEW: start waiting
        _commit_checkin(state)

    state["conv_state"] = "COMPLETED"
    situation = f"Confirm {current_host} is notified."
    situation += (
        " Tell them to leave the item."
        if "Delivery" in state["visitor_type"]
        else " Ask to wait in lobby."
    )
    return await _llm_reply(situation, state, query, client_id)


async def _handle_scheduling(
    client_id: str, query: str, state: Dict[str, Any], intent: str
) -> str:
    if not state.get("visitor_name"):
        return await _llm_reply("Ask for their name.", state, query, client_id)
    if not state.get("sched_employee_name"):
        return await _llm_reply("Ask who they want to meet.", state, query, client_id)
    if not state.get("sched_date"):
        return await _llm_reply("Ask for the date.", state, query, client_id)
    if not state.get("sched_time"):
        return await _llm_reply("Ask for the time.", state, query, client_id)
    slots = get_available_slots(state["sched_employee_name"], state["sched_date"])
    if state["sched_time"] not in slots:
        return await _llm_reply(
            f"Suggest alternate slots: {', '.join(slots[:3])}.", state, query, client_id
        )
    if intent == "confirm" or any(
        x in query.lower() for x in ["okay", "yes", "correct", "book"]
    ):
        if _finalize_meeting_and_log(state):
            state["scheduling_active"], state["conv_state"] = False, State.COMPLETED
            return await _llm_reply(
                "Confirm booking successful.", state, query, client_id
            )
    return await _llm_reply(
        f"Verify meeting with {state['sched_employee_name']} at {state['sched_time']}. Proceed?",
        state,
        query,
        client_id,
    )


async def route_query(client_id: str, user_query: str) -> str:
    from receptionist.database import get_company_details
    from models.groq_processor import GroqProcessor, BASE_SYSTEM_PROMPT

    state = get_session_state(client_id)
    llm = GroqProcessor.get_instance()
    query_low = user_query.lower().strip()

    # Wake word → fresh session
    if any(x in query_low for x in WAKE_WORDS):
        clear_session_state(client_id)
        state = get_session_state(client_id)
        state["greeted"] = True
        return (
            f"{_get_time_greeting()}! Welcome to {COMPANY_NAME}. "
            f"I am {AI_NAME}, how can I assist you today?"
        )

    extracted = await llm.extract_intent_and_entities(user_query)
    intent = extracted.get("intent", "general")
    entities = extracted.get("entities", {})
    _merge_checkin_entities(state, entities, user_query)

    # ── CHECK FOR PENDING SLACK REPLY ──────────────────────────────────────
    if state.get("awaiting_slack_reply"):
        host_name = state.get("meeting_with_resolved") or state.get(
            "sched_employee_name"
        )
        if host_name:
            slack_reply = pop_reply(host_name)  # None if not arrived yet
            if slack_reply:
                state["awaiting_slack_reply"] = False
                situation = (
                    f"{host_name} replied via Slack with the following message: "
                    f'"{slack_reply}". '
                    f"Relay this message naturally and helpfully to the visitor."
                )
                return await _llm_reply(situation, state, user_query, client_id)
    # ─────────────────────────────────────────────────────────────────────────

    # ── rest of the existing routing logic (unchanged) ────────────────────
    if any(
        x in query_low for x in ["don't know", "do not know", "anyone", "notify admin"]
    ):
        state["force_admin"] = True
        state["meeting_with_resolved"] = "Administration Team"

    if any(w in query_low for w in ["thank you", "thanks", "bye", "goodbye"]):
        reply = await llm.get_raw_response(
            f"Warm closing for: {user_query}", client_id=client_id
        )
        clear_session_state(client_id)
        return reply

    if any(x in query_low for x in ["free time", "available", "is he free"]):
        return await _handle_availability_check(state, user_query, client_id)

    if intent == "employee_lookup" or any(
        x in query_low for x in ["who is", "director", "ceo"]
    ):
        return await _handle_directory_lookup(state, user_query, client_id)

    if intent == "schedule_meeting" or state["scheduling_active"]:
        state["scheduling_active"] = True
        if (
            state["visitor_name"]
            and state["sched_employee_name"]
            and state["sched_date"]
            and state["sched_time"]
        ):
            if _finalize_meeting_and_log(state):
                state["scheduling_active"], state["conv_state"] = False, State.COMPLETED
                return await _llm_reply(
                    f"Confirmed meeting with {state['sched_employee_name']}.",
                    state,
                    user_query,
                    client_id,
                )
        return await _handle_scheduling(client_id, user_query, state, intent)

    if intent == "check_in" or state["meeting_with_resolved"]:
        if state.get("is_employee"):
            return await _llm_reply(
                "Wish staff a great day.", state, user_query, client_id
            )
        return await _finalize_checkin_and_respond(state, user_query, client_id)

    return await llm.get_response(
        client_id, user_query, company_info={"visitor_name": state["visitor_name"]}
    )
    state["greeting_sent"] = True
    return reply


async def _llm_reply(
    situation: str, state: Dict[str, Any], user_query: str, client_id: str
) -> str:
    llm = GroqProcessor.get_instance()
    visitor = state.get("visitor_name") or "Visitor"
    host = (
        state.get("meeting_with_resolved")
        or state.get("sched_employee_name")
        or "Admin"
    )
    info = f"TALKING TO: {visitor} | HOST: {host} | STATUS: {state['conv_state']}"
    is_first = not state.get("greeted", False)
    prompt = f"{BASE_SYSTEM_PROMPT}\nKB: {info}\nRULES: {'Greet warmly' if is_first else 'DO NOT greet again'}. GOAL: {situation}\nUSER: {user_query}"
    resp = await llm.get_raw_response(prompt, client_id=client_id)
    state["greeted"] = True
    return resp
