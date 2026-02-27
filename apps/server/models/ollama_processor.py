import logging
from typing import List, Dict
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
    """Handles text generation using an Ollama-served LLM (llama3.1)."""

    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.client = ollama.AsyncClient()
        self.model_name = "llama3.2:1b"
        self.history: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        logger.info(f"OllamaProcessor initialized with model '{self.model_name}'")

    def reset_history(self):
        """Clear the conversation history (preserving system prompt)."""
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
        logger.info("OllamaProcessor conversation history reset")

    # FIXED INDENTATION HERE:
    async def get_response(self, prompt: str) -> str:
        if not prompt:
            return ""

        self.history.append({"role": "user", "content": prompt})

        try:
            # 1. Add stream=False to ensure a complete response
            response = await self.client.chat(
                model=self.model_name, messages=self.history, stream=False
            )

            # 2. Correctly extract content from the Response object
            if hasattr(response, "message"):
                content = response.message.content
            else:
                content = response.get("message", {}).get("content", "")

            content = (content or "").strip()

            if not content:
                logger.warning("Ollama returned an empty response string")
                content = "I'm sorry, I couldn't process that. How can I help you?"

            self.history.append({"role": "assistant", "content": content})
            return content

        except Exception as e:
            logger.error(f"Ollama inference error: {e}")
            return "I'm having trouble thinking right now."
