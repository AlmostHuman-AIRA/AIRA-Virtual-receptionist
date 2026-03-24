import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from models.ollama_processor import OllamaProcessor
from receptionist.database import SessionLocal
from receptionist.models import Employee, Visitor, Meeting

logger = logging.getLogger(__name__)

SCHEDULE_INTENT = "schedule_meeting"

# State for multi-turn scheduling
meeting_state: Dict[str, Optional[str]] = {
    "visitor_name": None,
    "employee_name": None,
    "time": None,
}


def _serialize_employee(emp: Employee) -> Dict[str, Any]:
    """Converts DB object to dictionary."""
    return {
        "id": emp.id,
        "name": emp.name,
        "role": emp.role,
        "department": emp.department,
        "cabin_number": emp.cabin_number,
    }


def _parse_meeting_time(time_str: str) -> Optional[datetime]:
    """Parses time strings like '2 PM'."""
    if not time_str or not str(time_str).strip():
        return None
    s = str(time_str).strip()
    try:
        m = re.match(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?$", s)
        if m:
            h, minute = int(m.group(1)), int(m.group(2))
            return datetime.now().replace(
                hour=h, minute=minute, second=0, microsecond=0
            )

        m = re.match(r"^(\d{1,2})(?::(\d{2}))?\s*(AM|PM)$", s, re.IGNORECASE)
        if m:
            h, minute = int(m.group(1)), int(m.group(2) or 0)
            if m.group(3).upper() == "PM" and h != 12:
                h += 12
            elif m.group(3).upper() == "AM" and h == 12:
                h = 0
            return datetime.now().replace(
                hour=h, minute=minute, second=0, microsecond=0
            )
    except Exception:
        pass
    return None


def _log_visitor_history(name: str, status: str):
    """Saves the person to the general visitors history table."""
    session = SessionLocal()
    try:
        new_v = Visitor(name=name, status=status, checkin_time=datetime.now())
        session.add(new_v)
        session.commit()
        logger.info(f"Logged to Visitor History: {name}")
    except Exception as e:
        logger.error(f"Visitor log error: {e}")
        session.rollback()
    finally:
        session.close()


# Inside query_router.py


def handle_db_query(
    llm_entities: Dict[str, Any], raw_query: str = None
) -> Optional[Dict[str, Any]]:
    session = SessionLocal()
    try:
        # Combine all possible search terms
        search_term = (
            llm_entities.get("employee_name")
            or llm_entities.get("role")
            or llm_entities.get("name")
        )

        # If AI failed, use the raw query keywords
        if not search_term and raw_query:
            # Simple logic: find the most likely noun in the query
            words = raw_query.split()
            search_term = words[-1].strip("?")  # Take the last word as a guess

        if not search_term:
            return None

        print(f"[DEBUG] Searching DB for term: '{search_term}'")

        # BROAD SEARCH: Check Name, Role, AND Department
        emp = (
            session.query(Employee)
            .filter(
                (Employee.name.ilike(f"%{search_term}%"))
                | (Employee.role.ilike(f"%{search_term}%"))
                | (Employee.department.ilike(f"%{search_term}%"))
                |
                # Special case for "HR" -> "Human Resources"
                (
                    Employee.department.ilike("%HR%")
                    if "human" in search_term.lower()
                    else False
                )
            )
            .first()
        )

        if emp:
            return {
                "intent": "employee_lookup",
                "employee": _serialize_employee(emp),
            }

        # If no single employee, check if it's a department list
        dept_emps = (
            session.query(Employee)
            .filter(Employee.department.ilike(f"%{search_term}%"))
            .all()
        )
        if dept_emps:
            return {
                "intent": "role_lookup",
                "department": search_term,
                "employees": [_serialize_employee(e) for e in dept_emps],
            }

        return None
    finally:
        session.close()


def handle_schedule_meeting() -> Optional[Dict[str, Any]]:
    """Saves the appointment to the meetings table."""
    v_name = meeting_state.get("visitor_name")
    e_name = meeting_state.get("employee_name")
    t_raw = meeting_state.get("time")

    dt = _parse_meeting_time(t_raw)
    if not dt:
        return None

    session = SessionLocal()
    try:
        emp = session.query(Employee).filter(Employee.name.ilike(f"%{e_name}%")).first()
        if not emp:
            return {"error": "employee_not_found"}

        new_meeting = Meeting(
            visitor_name=v_name, employee_name=emp.name, scheduled_time=dt
        )
        session.add(new_meeting)
        session.commit()

        # Generate Google Calendar Invite!
        if getattr(emp, "email", None):
            try:
                from services.calendar_service import send_calendar_invite

                send_calendar_invite(v_name, emp.email, dt)
            except Exception as e:
                logger.error(f"Calendar invite failed: {e}")

        meeting_state.update(
            {"visitor_name": None, "employee_name": None, "time": None}
        )
        return {
            "visitor": v_name,
            "employee": emp.name,
            "time": dt.strftime("%I:%M %p"),
        }
    except Exception as e:
        session.rollback()
        return {"error": "db_error"}
    finally:
        session.close()


async def route_query(user_query: str) -> str:
    ollama = OllamaProcessor.get_instance()
    print(f"\n--- NEW QUERY: '{user_query}' ---")

    extracted_data = await ollama.extract_intent_and_entities(user_query)
    llm_entities = extracted_data.get("entities") or {}
    intent = extracted_data.get("intent", "general_conversation")

    # 1. LOG VISITOR HISTORY
    v_name = llm_entities.get("visitor_name") or llm_entities.get("name")
    if v_name:
        status = "Intern" if "intern" in user_query.lower() else "Visitor"
        _log_visitor_history(v_name, status)

    # 2. HANDLE MEETINGS
    if intent == SCHEDULE_INTENT:
        for key in ["visitor_name", "employee_name", "time"]:
            if llm_entities.get(key):
                meeting_state[key] = llm_entities[key]

        if not meeting_state["visitor_name"]:
            return "May I have your name, please?"
        if not meeting_state["employee_name"]:
            return "Who would you like to meet?"
        if not meeting_state["time"]:
            return "What time should I schedule this for?"

        res = handle_schedule_meeting()
        if res:
            if res.get("error") == "employee_not_found":
                return "I couldn't find that employee."
            return f"Perfect. I've scheduled a meeting for {res['visitor']} with {res['employee']} at {res['time']}."

    # 3. HANDLE EMPLOYEE LOOKUP
    lookup_keywords = ["who", "where", "manager", "cabin", "office", "department"]
    is_asking_lookup = intent in ["employee_lookup", "role_lookup"] or any(
        w in user_query.lower() for w in lookup_keywords
    )

    if is_asking_lookup:
        db_result = handle_db_query(llm_entities, raw_query=user_query)
        if db_result:
            return await ollama.generate_grounded_response(
                context=db_result, question=user_query
            )
        else:
            return "I'm sorry, I couldn't find any employee matching that description in our records."

    # 4. FALLBACK TO CHAT
    return await ollama.get_response(user_query)
