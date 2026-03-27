import json
import logging
import re
from typing import List, Dict, Any
import ollama

logger = logging.getLogger(__name__)

# 1. CONVERSATIONAL PROMPT (For general chat)
SYSTEM_PROMPT = """
You are AlmostHuman, the receptionist at Sharp Software Technology.
Greet visitors politely.
Identify if they are employee, intern, guest, or candidate.

### CONSTRAINTS & GUARDRAILS:
1. NEVER ask for or mention internal database IDs or ID numbers.
2. NEVER ask for department names or employee codes.
3. NEVER make up meeting availability or check calendars.
4. If a user wants a meeting, ONLY collect: Name, Person to see, and Time.
5. If the 'Context' provided to you already contains a name, do not ask for it again.
6. Keep replies very short (1-2 sentences).
7. Never mention being an AI.
"""

# 2. EXTRACTION PROMPT (For structured data)
EXTRACT_SYSTEM = """
Extract entities from the user's input. Return ONLY a JSON object.
Rules:
1. 'visitor_name': The person speaking.
2. 'employee_name': The person they want to meet.
3. 'role': Job title mentioned (e.g. HR, Manager).
4. 'time': Specific time (ignore 'today', 'now', 'soon').
5. 'intent': 'check_in', 'schedule_meeting', or 'employee_lookup'.

Example: "I am Jim, I want to see Priya at 4pm"
Output: {"intent": "schedule_meeting", "entities": {"visitor_name": "Jim", "employee_name": "Priya", "time": "4:00 PM"}}
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
        """Clear the conversation history."""
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
        logger.info("OllamaProcessor conversation history reset")

    async def get_response(self, prompt: str) -> str:
        """Standard conversational response (Maintains History)."""
        if not prompt:
            return ""

        # Keep history manageable
        self.history = [self.history[0]] + self.history[-6:]
        self.history.append({"role": "user", "content": prompt})

        try:
            response = await self.client.chat(
                model=self.model_name,
                messages=self.history,
                stream=False,
            )
            content = response.message.content.strip()
            self.history.append({"role": "assistant", "content": content})
            return content
        except Exception as e:
            logger.error(f"Ollama inference error: {e}")
            return "Welcome to Sharp Software Technology. How can I help you?"

    async def extract_intent_and_entities(self, user_query: str) -> Dict[str, Any]:
        """Stateless extraction: Uses EXTRACT_SYSTEM to get JSON data."""
        try:
            response = await self.client.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": EXTRACT_SYSTEM},
                    {"role": "user", "content": user_query.strip()},
                ],
                stream=False,
                options={"temperature": 0},  # Keep it deterministic
            )
            raw = response.message.content.strip()

            # Clean JSON: Remove markdown code blocks if present
            if "```" in raw:
                raw = re.sub(r"```(?:json)?", "", raw).strip()

            # Find the first '{' and last '}' to handle extra text
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start != -1 and end != 0:
                raw = raw[start:end]

            parsed = json.loads(raw)

            # Handle cases where LLM returns flat JSON or nested entities
            entities = parsed.get("entities", parsed)
            if not isinstance(entities, dict):
                entities = parsed

            return {
                "intent": parsed.get("intent", "general_conversation"),
                "entities": entities,
            }
        except Exception as e:
            logger.error(
                f"Extraction failed: {e} | Raw output was: {raw if 'raw' in locals() else 'None'}"
            )
            return {"intent": "general_conversation", "entities": {}}

    async def generate_grounded_response(self, context: dict, question: str) -> str:
        """Stateless grounded response for database lookups."""
        if "employee" in context:
            e = context["employee"]
            info = f"Name: {e['name']}, Role: {e['role']}, Cabin: {e['cabin_number']}, Department: {e['department']}"
        else:
            info = "No records found."

        # UPDATED PROMPT TO BE NEUTRAL (Changed 'A visitor' to 'The person')
        prompt_text = f"""You are a professional office receptionist.
    The person asked: "{question}"
    Internal records show: {info}
    
    Task:
    - Use the info to guide them.
    - Keep it to 1-2 sentences.
    - Tone: Friendly and professional.
    """

        try:
            response = await self.client.chat(
                model=self.model_name,
                messages=[{"role": "system", "content": prompt_text}],
                stream=False,
            )
            return response.message.content.strip()
        except Exception as e:
            logger.error(f"Grounded response error: {e}")
            return "I found the information. Please follow the directions to the cabin."
