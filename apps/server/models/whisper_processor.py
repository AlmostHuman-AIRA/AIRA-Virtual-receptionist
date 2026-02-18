import asyncio
import logging
import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

logger = logging.getLogger(__name__)


class WhisperProcessor:
    """Handles speech-to-text using Whisper model"""

    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        logger.info(f"Using device for Whisper: {self.device}")

        # Load Whisper model
        model_id = "openai/whisper-small"
        logger.info(f"Loading {model_id}...")

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_id)

        # Create pipeline
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

        logger.info("Whisper model ready for transcription")
        self.transcription_count = 0

    async def transcribe_audio(self, audio_bytes):
        """Transcribe audio bytes to text"""
        try:
            # Convert audio bytes to numpy array
            audio_array = (
                np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            )

            # Run transcription in executor to avoid blocking
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.pipe(audio_array)
            )

            transcribed_text = result["text"].strip()
            self.transcription_count += 1

            logger.info(
                f"Transcription #{self.transcription_count}: '{transcribed_text}'"
            )

            # Check for noise/empty transcription
            if not transcribed_text or len(transcribed_text) < 3:
                return "NO_SPEECH"

            # Check for common noise indicators
            noise_indicators = ["thank you", "thanks for watching", "you", ".", ""]
            if transcribed_text.lower().strip() in noise_indicators:
                return "NOISE_DETECTED"

            return transcribed_text

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None
