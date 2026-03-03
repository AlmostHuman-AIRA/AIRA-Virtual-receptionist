import json
import logging
from typing import List, Dict, Any

import ollama

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are AlmostHuman, the receptionist at Sharp Software Technology.
Greet visitors politely.
Identify if they are employee, intern, guest, or candidate.
Confirm details, then guide them to HR, meeting room, or team.
Never mention being an AI.
Keep replies short and professional.

Response:
Natural reply.

Data:
{"t":"","n":"","p":"","m":"","a":"","s":""}
"""


class OllamaProcessor:
    """Handles text generation using an Ollama-served LLM (llama3.2)."""

    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.client = ollama.AsyncClient()
        self.model_name = "llama-reduced"
        self.history: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        logger.info(f"OllamaProcessor initialized with model '{self.model_name}'")

    def reset_history(self):
        """Clear the conversation history (preserving system prompt)."""
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
        logger.info("OllamaProcessor conversation history reset")

    async def get_response(self, prompt: str) -> str:
        if not prompt:
            return ""

        # Keep only last 6 messages + system prompt
        self.history = [self.history[0]] + self.history[-6:]

        self.history.append({"role": "user", "content": prompt})

        try:
            response = await self.client.chat(
                model=self.model_name,
                messages=self.history,
                stream=False,
            )

            if hasattr(response, "message"):
                content = response.message.content
            else:
                content = response.get("message", {}).get("content", "")

            content = (content or "").strip()

            if not content:
                logger.warning("Ollama returned empty response.")
                content = "I'm sorry, I couldn't process that. How can I help you?"

            self.history.append({"role": "assistant", "content": content})
            return content

        except Exception as e:
            logger.error(f"Ollama inference error: {e}")
            return "I'm having trouble thinking right now."

    async def generate_grounded_response(
        self, context: Dict[str, Any], question: str
    ) -> str:
        """
        Generate a response that is strictly grounded in the provided database context.

        The model is explicitly instructed to ONLY use the given structured data
        and to avoid guessing or hallucinating any information that is not present.
        """
        try:
            system_message = (
                "You are AlmostHuman, the receptionist at Sharp Software Technology. "
                "You will receive structured data from an office database and a visitor's question. "
                "Your job is to respond in a short, professional receptionist style.\n\n"
                "CRITICAL RULES:\n"
                "- Only use the provided database information.\n"
                "- If the answer is not clearly present in the data, say you don't know.\n"
                "- Do NOT invent names, departments, cabin numbers, or any other facts.\n"
                "- Do NOT reference being an AI or language model.\n"
            )

            context_pretty = json.dumps(context, indent=2, ensure_ascii=False)

            user_message = (
                f"Database result (structured JSON):\n{context_pretty}\n\n"
                f"Visitor question:\n{question}\n\n"
                "Using ONLY the information in the database result above, "
                "generate a concise, professional receptionist-style answer."
            )

            response = await self.client.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                stream=False,
            )

            if hasattr(response, "message"):
                content = response.message.content
            else:
                content = response.get("message", {}).get("content", "")

            content = (content or "").strip()

            if not content:
                logger.warning("Ollama returned empty grounded response.")
                content = (
                    "I'm sorry, I could not derive an answer from the office database."
                )

            return content

        except Exception as e:
            logger.error(f"Ollama grounded inference error: {e}")
            return "I'm having trouble accessing the office database information right now."
