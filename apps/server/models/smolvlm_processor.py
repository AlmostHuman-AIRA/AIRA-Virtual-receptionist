import torch
import asyncio
import time
import re
import io
import logging

from PIL import Image
from threading import Thread

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    TextIteratorStreamer,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are AlmostHuman AI, the official holographic office receptionist.

You exist inside the lobby hologram display.
You are not an AI model. You are the digital front-desk assistant.

Rules:
- Never say you are an AI.
- Never explain limitations about being non-human.
- Never discuss your abilities.
- Do not apologize unless there is an actual issue.
- Keep responses under 15 words.
- Be professional, confident, and direct.

Assist visitors with directions, check-ins, HR, meetings, and team locations.
"""


class SmolVLMProcessor:
    """Handles image + text processing using SmolVLM2 model"""

    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device for SmolVLM2: {self.device}")

        # Load SmolVLM2 model
        model_path = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
        logger.info(f"Loading {model_path}...")

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            attn_implementation="eager",
        )

        logger.info("SmolVLM2 model ready for multimodal generation")

        # Cache for most recent image
        self.last_image = None
        self.last_image_timestamp = 0
        self.lock = asyncio.Lock()

        # Message history management
        self.message_history = []
        self.max_history_messages = 4  # Keep last 4 exchanges

        # Counter
        self.generation_count = 0

    async def set_image(self, image_data):
        """Cache the most recent image received"""
        async with self.lock:
            try:
                # Convert image data to PIL Image
                image = Image.open(io.BytesIO(image_data))

                # Resize to 75% of original size for efficiency
                new_size = (int(image.size[0] * 0.75), int(image.size[1] * 0.75))
                image = image.resize(new_size, Image.Resampling.LANCZOS)

                # Clear message history when new image is set
                self.message_history = []
                self.last_image = image
                self.last_image_timestamp = time.time()
                logger.info("Image cached successfully")
                return True
            except Exception as e:
                logger.error(f"Error processing image: {e}")
                return False

    async def process_text_with_image(self, text, initial_chunks=3):
        """Process text with image context using SmolVLM2"""
        async with self.lock:
            try:
                if not self.last_image:
                    messages = [
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": text},
                            ],
                        },
                    ]
                else:
                    messages = [
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": text},
                            ],
                        },
                    ]

                # Apply chat template
                inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(self.device, dtype=torch.bfloat16)

                # Create a streamer for token-by-token generation
                streamer = TextIteratorStreamer(
                    tokenizer=self.processor.tokenizer,
                    skip_special_tokens=True,
                    skip_prompt=True,
                    clean_up_tokenization_spaces=False,
                )

                # Configure generation parameters
                generation_kwargs = dict(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=1200,
                    streamer=streamer,
                )

                # Start generation in a separate thread
                thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
                thread.start()

                # Collect initial text until we have a complete sentence or enough content
                initial_text = ""
                min_chars = 50  # Minimum characters to collect for initial chunk
                sentence_end_pattern = re.compile(r"[.!?]")
                has_sentence_end = False
                initial_collection_stopped_early = False

                # Collect the first sentence or minimum character count
                for chunk in streamer:
                    initial_text += chunk
                    logger.info(f"Streaming chunk: '{chunk}'")

                    # Check if we have a sentence end
                    if sentence_end_pattern.search(chunk):
                        has_sentence_end = True
                        # If we have at least some content, break after sentence end
                        if len(initial_text) >= min_chars / 2:
                            initial_collection_stopped_early = True
                            break

                    # If we have enough content, break
                    if len(initial_text) >= min_chars and (
                        has_sentence_end or "," in initial_text
                    ):
                        initial_collection_stopped_early = True
                        break

                    # Safety check - if we've collected a lot of text without sentence end
                    if len(initial_text) >= min_chars * 2:
                        initial_collection_stopped_early = True
                        break

                # Return initial text and the streamer for continued generation
                self.generation_count += 1
                logger.info(
                    f"SmolVLM2 initial generation: '{initial_text}' ({len(initial_text)} chars)"
                )

                # Store user message and initial response
                self.pending_user_message = text
                self.pending_response = initial_text

                return streamer, initial_text, initial_collection_stopped_early

            except Exception as e:
                logger.error(f"SmolVLM2 streaming generation error: {e}")
                return None, f"Error processing: {text}", False

    def update_history_with_complete_response(
        self, user_text, initial_response, remaining_text=None
    ):
        """Update message history with complete response, including any remaining text"""
        # Combine initial and remaining text if available
        complete_response = initial_response
        if remaining_text:
            complete_response = initial_response + remaining_text

        # Add to history for context in future exchanges
        self.message_history.append({"role": "user", "text": user_text})

        self.message_history.append({"role": "assistant", "text": complete_response})

        # Trim history to keep only recent messages
        if len(self.message_history) > self.max_history_messages:
            self.message_history = self.message_history[-self.max_history_messages :]

        logger.info(
            f"Updated message history with complete response ({len(complete_response)} chars)"
        )
