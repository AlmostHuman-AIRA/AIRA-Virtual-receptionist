import logging
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

# Use absolute imports based on your project structure
from models.ollama_processor import OllamaProcessor
from receptionist.database import SessionLocal
from receptionist.models import Employee

logger = logging.getLogger(__name__)

DB_INTENTS = {"employee_lookup", "department_lookup", "cabin_lookup"}


def _serialize_employee(emp: Employee) -> Dict[str, Any]:
    return {
        "id": emp.id,
        "name": emp.name,
        "department": emp.department,
        "cabin_number": emp.cabin_number,
    }


def handle_db_query(extracted_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    intent = extracted_data.get("intent", "general_conversation")
    entities = extracted_data.get("entities") or {}

    session: Session = SessionLocal()
    try:
        # 1. PRIORITY: If a NAME is present, always try employee lookup first
        name = entities.get("name")
        if name and str(name).strip():
            employee = (
                session.query(Employee).filter(Employee.name.ilike(f"%{name}%")).first()
            )
            if employee:
                return {
                    "intent": "employee_lookup",
                    "employee": _serialize_employee(employee),
                    "search_name": name,
                }

        # 2. DEPARTMENT LOOKUP
        if intent == "department_lookup" or entities.get("department"):
            dept = entities.get("department")
            if dept:
                employees = (
                    session.query(Employee)
                    .filter(Employee.department.ilike(f"%{dept}%"))
                    .all()
                )
                if employees:
                    return {
                        "intent": "department_lookup",
                        "department": dept,
                        "employees": [_serialize_employee(e) for e in employees],
                    }

        # 3. CABIN LOOKUP (by number)
        if intent == "cabin_lookup":
            cabin = entities.get("cabin_number")
            if cabin:
                employee = (
                    session.query(Employee)
                    .filter(Employee.cabin_number.ilike(f"%{cabin}%"))
                    .first()
                )
                if employee:
                    return {
                        "intent": "cabin_lookup",
                        "cabin_number": cabin,
                        "employee": _serialize_employee(employee),
                    }
        return None
    finally:
        session.close()


async def route_query(user_query: str) -> str:
    if not user_query or not user_query.strip():
        return "How can I help you today?"

    ollama = OllamaProcessor.get_instance()

    # Step 1: Extract Intent/Entities
    extracted_data = await ollama.extract_intent_and_entities(user_query)

    # Step 2: Try Database
    db_result = handle_db_query(extracted_data)

    if db_result:
        # Step 3: Generate Grounded Response (The "Human" answer)
        return await ollama.generate_grounded_response(
            context=db_result,
            question=user_query,
        )

    # Step 4: Fallback to General Conversation
    if extracted_data.get("intent") == "general_conversation":
        return await ollama.get_response(user_query)

    return "I'm sorry, I couldn't find any matching records in our office database."
