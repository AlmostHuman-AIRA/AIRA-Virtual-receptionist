import logging
import re
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from models.ollama_processor import OllamaProcessor
from receptionist.database import SessionLocal
from receptionist.models import Employee

logger = logging.getLogger(__name__)


def detect_intent(user_query: str) -> str:
    """
    Very simple rule-based intent detection.

    Returns one of:
        - "employee_lookup"
        - "department_lookup"
        - "cabin_lookup"
        - "non_db"
    """
    if not user_query:
        return "non_db"

    q = user_query.lower()

    # Cabin-related queries take precedence
    if "cabin" in q or "desk" in q or "seat" in q or re.search(r"\bcabin\s+\w+", q):
        return "cabin_lookup"

    # Department/team queries
    if "department" in q or "team" in q or "which dept" in q or "which department" in q:
        return "department_lookup"

    # Employee lookup (where / who is NAME)
    if (
        "where is" in q
        or "who is" in q
        or "employee" in q
        or "colleague" in q
        or "manager" in q
    ):
        return "employee_lookup"

    return "non_db"


def _extract_person_name(user_query: str) -> Optional[str]:
    """
    Best-effort name extraction for patterns like:
        - "Where is Rohit?"
        - "Where is Rohit Sharma's cabin?"
        - "Who is Priya from HR?"
    """
    q = user_query.strip()

    # Normalize whitespace
    q = re.sub(r"\s+", " ", q)

    # Common pattern: "where is NAME"
    m = re.search(r"where is (.+)", q, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"who is (.+)", q, flags=re.IGNORECASE)

    if m:
        candidate = m.group(1)
        # Remove trailing question marks and possessives
        candidate = re.sub(r"[?!.]+$", "", candidate).strip()
        candidate = re.sub(r"'s\b", "", candidate).strip()

        # Remove trailing phrases that are clearly not part of the name
        stop_words = [
            "from hr",
            "from it",
            "from finance",
            "from engineering",
            "from sales",
            "from marketing",
            "from support",
            "from the hr department",
            "from the it department",
        ]
        for stop in stop_words:
            if candidate.lower().endswith(stop):
                candidate = candidate[: -len(stop)].strip()
                break

        # Heuristic: limit to a reasonable name length
        parts = candidate.split()
        if 0 < len(parts) <= 4:
            return candidate

    return None


def _extract_department(user_query: str) -> Optional[str]:
    """
    Extract probable department name for queries like:
        - "Who is in the HR department?"
        - "Show all employees in engineering."
    """
    q = user_query.lower()

    # Simple keyword-based departments
    known_departments = [
        "hr",
        "human resources",
        "it",
        "engineering",
        "sales",
        "marketing",
        "finance",
        "support",
    ]

    for dept in known_departments:
        if dept in q:
            return dept

    # Fallback: capture word(s) before "department"
    m = re.search(r"in the ([a-z ]+)\s+department", q)
    if m:
        return m.group(1).strip()

    return None


def _extract_cabin_number(user_query: str) -> Optional[str]:
    """
    Extract cabin identifier for queries like:
        - "Who sits in cabin 12?"
        - "Who is in cabin A3?"
    """
    m = re.search(r"cabin\s+([A-Za-z0-9\-]+)", user_query, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def _serialize_employee(emp: Employee) -> Dict[str, Any]:
    return {
        "id": emp.id,
        "name": emp.name,
        "department": emp.department,
        "cabin_number": emp.cabin_number,
    }


def handle_db_query(user_query: str) -> Optional[Dict[str, Any]]:
    """
    Extracts relevant entity from the query, runs the appropriate DB query,
    and returns a structured dictionary result or None if no match.
    """
    intent = detect_intent(user_query)

    if intent == "non_db":
        return None

    session: Session = SessionLocal()
    try:
        if intent == "employee_lookup":
            name = _extract_person_name(user_query)
            if not name:
                logger.info("Could not extract employee name from query")
                return None

            employee = (
                session.query(Employee).filter(Employee.name.ilike(f"%{name}%")).first()
            )
            if not employee:
                return None

            return {
                "intent": intent,
                "employee": _serialize_employee(employee),
                "search_name": name,
            }

        if intent == "department_lookup":
            department = _extract_department(user_query)
            if not department:
                logger.info("Could not extract department from query")
                return None

            employees: List[Employee] = (
                session.query(Employee)
                .filter(Employee.department.ilike(f"%{department}%"))
                .all()
            )
            if not employees:
                return None

            return {
                "intent": intent,
                "department": department,
                "employees": [_serialize_employee(e) for e in employees],
            }

        if intent == "cabin_lookup":
            cabin = _extract_cabin_number(user_query)
            if not cabin:
                logger.info("Could not extract cabin number from query")
                return None

            employee = (
                session.query(Employee).filter(Employee.cabin_number == cabin).first()
            )
            if not employee:
                return None

            return {
                "intent": intent,
                "cabin_number": cabin,
                "employee": _serialize_employee(employee),
            }

        # Fallback
        return None

    finally:
        session.close()


async def route_query(user_query: str) -> str:
    """
    Main entry point for handling a user query.

    - Detect intent
    - If DB intent: query SQLite and, if found, call the LLM with grounded context
    - Otherwise: call the LLM normally
    """
    intent = detect_intent(user_query)
    ollama = OllamaProcessor.get_instance()

    if intent != "non_db":
        db_result = handle_db_query(user_query)
        if db_result:
            logger.info(f"DB hit for intent '{intent}' with result: {db_result}")
            response = await ollama.generate_grounded_response(
                context=db_result,
                question=user_query,
            )
            return response

        logger.info(f"No matching DB record found for intent '{intent}'")
        return "No matching record found in the office database."

    # Non-DB query: fall back to normal conversational LLM
    logger.info("Non-DB query, routing directly to LLM")
    return await ollama.get_response(user_query)
