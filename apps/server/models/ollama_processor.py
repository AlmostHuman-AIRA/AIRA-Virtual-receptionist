import json
import logging
import re
from typing import List, Dict, Any

import ollama

logger = logging.getLogger(__name__)

# Cleaned up: Removed the confusing empty JSON Data block
SYSTEM_PROMPT = """
You are AlmostHuman, the receptionist at Sharp Software Technology.
Greet visitors politely.
Identify if they are employee, intern, guest, or candidate.
Confirm details, then guide them to HR, meeting room, or team.
Never mention being an AI.
Keep replies short and professional.
NEVER make up a name, cabin number, or department.
Do not guess the name , role, or department of an employee if not explicitly mentioned by the visitor. If the visitor does not provide this information, respond with a polite request for more details.
"""


class OllamaProcessor:
    """Handles text generation using an Ollama-served LLM."""

    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.client = ollama.AsyncClient()
        self.model_name = "llama3.2:3b"
        self.history: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        logger.info(f"OllamaProcessor initialized with model '{self.model_name}'")

    def reset_history(self):
        """Clear the conversation history (preserving system prompt)."""
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
        logger.info("OllamaProcessor conversation history reset")

    async def get_response(self, prompt: str) -> str:
        """Standard conversational response (Maintains History)."""
        if not prompt:
            return ""

        # Keep only last 6 messages + system prompt to prevent memory bloat
        self.history = [self.history[0]] + self.history[-6:]
        self.history.append({"role": "user", "content": prompt})

        # Keep history short for Phi
        if len(self.history) > 7:
            self.history = [self.history[0]] + self.history[-6:]

        try:
            response = await self.client.chat(
                model=self.model_name,
                messages=self.history,
                stream=False,
            )
            content = response.message.content.strip()

            if not content:
                content = "Welcome to Sharp Software Technology! How can I help you?"

            self.history.append({"role": "assistant", "content": content})
            return content

        except Exception as e:
            logger.error(f"Ollama inference error: {e}")
            return "I'm having trouble connecting to the system right now."

    async def extract_intent_and_entities(self, user_query: str) -> Dict[str, Any]:
        """Stateless extraction: Does NOT pollute conversation history."""

        EXTRACT_SYSTEM = """
    You are a technical data extractor. Output ONLY valid JSON.
    INTENTS: 
    - "visitor_checkin": User introduces themselves (e.g., "I am Rinko").
    - "employee_lookup": User asks for a person, role, or department.
    - "schedule_meeting": User wants to book a time with someone.

    ENTITIES:
    - "visitor_name": Name of the person speaking.
    - "visitor_status": "New Intern", "Guest", "Candidate".
    - "employee_name": Name of the staff member mentioned.
    - "role": Job title (e.g., "Marketing Manager").
    - "time": Meeting time (e.g., "4:30 PM").
    """
        context_msgs = self.history[-3:] if len(self.history) > 1 else []

        messages = [{"role": "system", "content": EXTRACT_SYSTEM}]
        messages.extend(context_msgs)  # Add history to the extraction call
        messages.append({"role": "user", "content": f"Extract from: '{user_query}'"})

        try:
            response = await self.client.chat(
                model=self.model_name,
                format="json",
                messages=messages,  # Now includes history!
                options={"temperature": 0},
            )
            raw = response.message.content.strip()

            # Robust JSON cleaning: find the first '{' and last '}'
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                raw = match.group(0)

            parsed = json.loads(raw)
            return {
                "intent": parsed.get("intent", "general_conversation"),
                "entities": (
                    parsed.get("entities", {})
                    if isinstance(parsed.get("entities"), dict)
                    else {}
                ),
            }
        except Exception as e:
            logger.warning(f"Extraction failed for query '{user_query}': {e}")
            return {"intent": "general_conversation", "entities": {}}

    async def generate_grounded_response(self, context: dict, question: str) -> str:
        """Stateless grounded response: Does NOT pollute conversation history."""
        if "employee" in context:
            e = context["employee"]
            info = f"Name: {e['name']}, Role: {e['role']}, Cabin: {e['cabin_number']}, Department: {e['department']}"
        elif "employees" in context:
            dept = context.get("department", "the requested")
            people = ", ".join(
                [f"{emp['name']} ({emp['role']})" for emp in context["employees"]]
            )
            info = f"Department: {dept}, Staff: {people}"
        else:
            info = "No records found."

        # Your exact prompt text preserved
        prompt_text = f"""u are a professional and polite office receptionist assisting visitors inside a company office.
    A visitor asked: "{question}"
    The system searched internal records and returned: {info}
    
    Your task:
    - Respond like a real office receptionist.
    - Use the information in {info} to guide the visitor.
    - Keep the response short, natural, and conversational (1–3 sentences).
    - Tone: Friendly, helpful, and professional.
    Now generate the receptionist response."""

        try:
            # Direct call to avoid history poisoning
            response = await self.client.chat(
                model=self.model_name,
                messages=[{"role": "system", "content": prompt_text}],
                stream=False,
            )
            return response.message.content.strip()
        except Exception as e:
            logger.error(f"Grounded response error: {e}")
            return "I found the information, but I'm having trouble explaining it. Please wait a moment."
