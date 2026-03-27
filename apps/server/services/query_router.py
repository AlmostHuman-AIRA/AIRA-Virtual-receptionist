import logging
import re
from datetime import datetime
from typing import Any, Dict, Optional
from sqlalchemy import or_
from sqlalchemy.orm import Session

from receptionist.database import SessionLocal
from receptionist.models import Employee, Visitor, Meeting
from models.ollama_processor import OllamaProcessor

logger = logging.getLogger(__name__)

# State to track the interaction across turns
session_state: Dict[str, Any] = {
    "visitor_id": None,
    "visitor_name": None,
    "status": None,  # <--- ADDED THIS
    "employee_name": None,
    "time": None,
}


def _clear_state():
    logger.info("Cleaning session state...")
    for key in session_state:
        session_state[key] = None


def _parse_meeting_time(time_str: str) -> Optional[datetime]:
    """Robust parsing for messy STT output."""
    if not time_str:
        return None
    s = str(time_str).lower().strip()
    s = (
        s.replace("p.m.", "pm")
        .replace("a.m.", "am")
        .replace("to", "2")
        .replace(".", ":")
    )
    s = re.sub(r"[^0-9ap: ]", "", s)

    try:
        m = re.search(r"(\d{1,2}):?(\d{2})?\s*(am|pm)?", s)
        if m:
            hour = int(m.group(1))
            minute = int(m.group(2) or 0)
            meridiem = m.group(3)
            if meridiem == "pm" and hour < 12:
                hour += 12
            if meridiem == "am" and hour == 12:
                hour = 0
            if not meridiem and 1 <= hour <= 7:
                hour += 12
            return datetime.now().replace(
                hour=hour, minute=minute, second=0, microsecond=0
            )
    except Exception as e:
        logger.error(f"Time parse error: {e}")
    return None


def _merge_entities(entities: Dict[str, Any], raw_query: str) -> None:
    raw_lower = raw_query.lower()
    PRONOUNS = [
        "i",
        "me",
        "you",
        "myself",
        "someone",
        "receptionist",
        "user",
        "here",
        "software",
        "developer",
    ]

    # 1. VISITOR NAME EXTRACTION (With Regex Fallback)
    ent_visitor = entities.get("visitor_name") or entities.get("name")

    # If LLM failed to extract name, use Regex to find "myself [Name]" or "I am [Name]"
    if not ent_visitor:
        name_match = re.search(
            r"(?:myself|i am|i'm|name is|this is)\s+([a-z]+)", raw_lower
        )
        if name_match:
            ent_visitor = name_match.group(1)

    if ent_visitor:
        # Clean common prefixes if LLM included them
        clean_name = re.sub(
            r"^(myself|i am|i'm|this is)\s+", "", str(ent_visitor), flags=re.IGNORECASE
        ).strip()
        if clean_name.lower() not in PRONOUNS and len(clean_name) > 2:
            session_state["visitor_name"] = clean_name.capitalize()

    # 2. EMPLOYEE NAME EXTRACTION
    ent_employee = entities.get("employee_name") or entities.get("role")
    is_target_id = any(
        k in raw_lower
        for k in ["meet", "see", "looking for", "visit", "appointment with", "to see"]
    )

    if ent_employee and ent_employee.lower() not in PRONOUNS:
        session_state["employee_name"] = ent_employee
    elif is_target_id:
        # Fallback: if they say "meet Priya", and LLM missed it, try to find the word after "meet"
        emp_match = re.search(r"(?:meet|see|visit|with)\s+([a-z]+)", raw_lower)
        if emp_match:
            candidate = emp_match.group(1)
            if candidate not in PRONOUNS:
                session_state["employee_name"] = candidate

    # 3. TIME EXTRACTION (With Regex Fallback)
    new_time = entities.get("time")
    if not new_time:
        time_match = re.search(
            r"(\d{1,2}[:.]?\d{0,2}\s*(?:am|pm|p\.m\.|a\.m\.))", raw_lower
        )
        if time_match:
            new_time = time_match.group(1)

    if new_time and new_time.lower() not in ["today", "now", "soon"]:
        session_state["time"] = new_time


def log_initial_visitor(name: str, status: str = "Arrived") -> Optional[int]:
    session = SessionLocal()
    try:
        new_v = Visitor(name=name, status=status, checkin_time=datetime.now())
        session.add(new_v)
        session.commit()
        session.refresh(new_v)
        logger.info(f"Successfully logged visitor {name} to DB.")
        return new_v.id
    except Exception as e:
        logger.error(f"Visitor log error: {e}")
        return None
    finally:
        session.close()


def schedule_meeting_record() -> Optional[Dict]:
    v_name = session_state["visitor_name"]
    e_name = session_state["employee_name"]
    t_raw = session_state["time"]

    dt = _parse_meeting_time(t_raw)
    if not dt:
        return {"error": "invalid_time"}

    session = SessionLocal()
    try:
        emp = (
            session.query(Employee)
            .filter(
                or_(
                    Employee.name.ilike(f"%{e_name}%"),
                    Employee.role.ilike(f"%{e_name}%"),
                )
            )
            .first()
        )

        if not emp:
            return {"error": "employee_not_found"}

        new_meeting = Meeting(
            visitor_name=v_name, employee_name=emp.name, scheduled_time=dt
        )
        session.add(new_meeting)
        session.commit()

        logger.info(f"Meeting RECORDED: {v_name} with {emp.name} at {dt}")

        # --- Google Calendar Hook ---
        if getattr(emp, "email", None):
            try:
                from services.calendar_service import send_calendar_invite

                send_calendar_invite(v_name, emp.email, dt)
                logger.info("Successfully pushed invite wrapper request.")
            except Exception as e:
                logger.error(f"Calendar invite wrapper failed: {e}")
        # ----------------------------

        return {
            "employee": emp.name,
            "time": dt.strftime("%I:%M %p"),
            "cabin": emp.cabin_number,
        }
    except Exception as e:
        logger.error(f"Database error: {e}")
        return {"error": "db_error"}
    finally:
        session.close()


async def route_query(user_query: str) -> str:
    ollama = OllamaProcessor.get_instance()

    extracted = await ollama.extract_intent_and_entities(user_query)
    entities = extracted.get("entities") or {}
    intent = extracted.get("intent")

    # 1. Update State (Now with Regex Fallbacks)
    _merge_entities(entities, user_query)

    # 2. PRIORITY: EMPLOYEE LOOKUP
    if intent in ["employee_lookup", "role_lookup"] or any(
        k in user_query.lower() for k in ["who is", "where is", "cabin"]
    ):
        session = SessionLocal()
        search_term = (
            session_state["employee_name"]
            or entities.get("role")
            or entities.get("name")
        )
        if search_term:
            emp = (
                session.query(Employee)
                .filter(
                    or_(
                        Employee.name.ilike(f"%{search_term}%"),
                        Employee.role.ilike(f"%{search_term}%"),
                    )
                )
                .first()
            )
            if emp:
                session.close()
                return await ollama.generate_grounded_response(
                    context={
                        "intent": "lookup",
                        "employee": {
                            "name": emp.name,
                            "role": emp.role,
                            "cabin_number": emp.cabin_number,
                            "department": emp.department,  # Added missing field for prompt
                        },
                    },
                    question=user_query,
                )
        session.close()

    # 3. CHECK-IN LOGIC (Triggers as soon as we have a name)
    if session_state["visitor_name"] and not session_state["visitor_id"]:
        session = SessionLocal()
        is_employee = (
            session.query(Employee)
            .filter(Employee.name.ilike(f"%{session_state['visitor_name']}%"))
            .first()
        )
        session.close()

        query_lower = user_query.lower()
        if is_employee:
            status = "Employee"
        elif "intern" in query_lower:
            status = "Intern Visit"
        elif "interview" in query_lower or "candidate" in query_lower:
            status = "Candidate"
        elif "guest" in query_lower:
            status = "Guest"
        else:
            status = "Arrived"

        # SAVE STATUS TO STATE SO LLM KNOWS WHO THEY ARE TALKING TO
        session_state["status"] = status  # <--- ADDED THIS

        session_state["visitor_id"] = log_initial_visitor(
            session_state["visitor_name"], status
        )

    # 4. MEETING SCHEDULING LOGIC
    if intent == "schedule_meeting" or (
        session_state["employee_name"] and session_state["time"]
    ):
        if not session_state["visitor_name"]:
            return "I'd be happy to schedule that. May I have your name first?"
        if not session_state["employee_name"]:
            return (
                f"Certainly {session_state['visitor_name']}. Who are you here to see?"
            )
        if not session_state["time"]:
            return f"Understood. What time is your meeting with {session_state['employee_name']}?"

        res = schedule_meeting_record()
        if "error" in res:
            if res["error"] == "employee_not_found":
                session_state["employee_name"] = None
                return f"I'm sorry, I couldn't find '{entities.get('employee_name', 'that person')}' in our staff directory."
            return "I had trouble saving the meeting. Could you repeat the time?"

        v_save = session_state["visitor_name"]
        _clear_state()
        return f"Perfect {v_save}. I've scheduled your meeting with {res['employee']} for {res['time']}. You can head to cabin {res['cabin']}."

    # 5. FALLBACK
    return await ollama.get_response(
        f"Context: {session_state}. User says: {user_query}"
    )
