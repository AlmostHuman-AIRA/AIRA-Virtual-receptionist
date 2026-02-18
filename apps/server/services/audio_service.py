import asyncio
import base64
import json
import logging
import numpy as np
import traceback
from fastapi import WebSocket

from models.whisper_processor import WhisperProcessor
from models.smolvlm_processor import SmolVLMProcessor
from models.tts_processor import KokoroTTSProcessor
from services.streaming_service import collect_remaining_text
from utils.compatibility import anext

logger = logging.getLogger(__name__)


async def process_audio_segment(
    websocket: WebSocket,
    client_id: str,
    audio_data,
    image_data=None,
    manager=None,
    whisper_processor=None,
    smolvlm_processor=None,
    tts_processor=None,
):
    """Process a complete audio segment through the pipeline with optional image"""
    manager.client_state[client_id] = "THINKING"

    try:
        # Log what we received
        if image_data:
            logger.info(
                f"🎥 Processing audio+image segment: audio={len(audio_data)} bytes, image={len(image_data)} bytes"
            )
            manager.update_stats("audio_with_image_received")

            # Save the image for verification
            saved_path = manager.image_manager.save_image(
                image_data, client_id, "multimodal"
            )
            if saved_path:
                # Verify the saved image
                verification = manager.image_manager.verify_image(saved_path)
                if verification.get("valid"):
                    logger.info(
                        f"📸 Image verified successfully: {verification['size']} pixels"
                    )
                else:
                    logger.warning(f"⚠️ Image verification failed: {verification}")

        else:
            logger.info(f"🎤 Processing audio-only segment: {len(audio_data)} bytes")
            manager.update_stats("audio_segments_received")

        # Send interrupt immediately since frontend determined this is valid speech
        logger.info("Sending interrupt signal")
        interrupt_message = json.dumps({"interrupt": True})
        await websocket.send_text(interrupt_message)

        # Step 1: Transcribe audio with Whisper
        logger.info("Starting Whisper transcription")
        transcribed_text = await whisper_processor.transcribe_audio(audio_data)
        logger.info(f"Transcription result: '{transcribed_text}'")

        # Check if transcription indicates noise
        if transcribed_text in ["NOISE_DETECTED", "NO_SPEECH", None]:
            logger.info(
                f"Noise detected in transcription: '{transcribed_text}'. Skipping further processing."
            )
            return

        # Step 2: Set image if provided, then process text
        if image_data:
            await smolvlm_processor.set_image(image_data)
            logger.info("🖼️ Image set for multimodal processing")

        # Process transcribed text with image using SmolVLM2
        logger.info("Starting SmolVLM2 generation")
        streamer, initial_text, initial_collection_stopped_early = (
            await smolvlm_processor.process_text_with_image(transcribed_text)
        )
        logger.info(
            f"SmolVLM2 initial text: '{initial_text[:50]}...' ({len(initial_text)} chars)"
        )

        # Check if VLM response indicates noise
        if initial_text.startswith("NOISE:"):
            logger.info(
                f"Noise detected in VLM processing: '{initial_text}'. Skipping TTS."
            )
            return

        # Step 3: Generate TTS for initial text WITH NATIVE TIMING
        if initial_text:
            logger.info("Starting TTS for initial text")
            tts_task = asyncio.create_task(
                tts_processor.synthesize_initial_speech_with_timing(initial_text)
            )
            manager.set_task(client_id, "tts", tts_task)

            # FIXED: Properly unpack the tuple
            tts_result = await tts_task
            if isinstance(tts_result, tuple) and len(tts_result) == 2:
                initial_audio, initial_timings = tts_result
            else:
                # Fallback for legacy method
                initial_audio = tts_result
                initial_timings = []
                logger.warning(
                    "TTS returned single value instead of tuple - no timing data available"
                )

            logger.info(
                f"Initial TTS complete: {len(initial_audio) if initial_audio is not None else 0} samples, {len(initial_timings)} word timings"
            )

            if initial_audio is not None and len(initial_audio) > 0:
                # Convert to base64 and send to client WITH TIMING DATA
                audio_bytes = (initial_audio * 32767).astype(np.int16).tobytes()
                base64_audio = base64.b64encode(audio_bytes).decode("utf-8")

                # Send audio with native timing information
                audio_message = {
                    "audio": base64_audio,
                    "word_timings": initial_timings,  # 🎉 NATIVE TIMING DATA!
                    "sample_rate": 24000,
                    "method": "native_kokoro_timing",
                    "modality": "multimodal" if image_data else "audio_only",
                }
                manager.client_state[client_id] = "SPEAKING"

                await websocket.send_text(json.dumps(audio_message))
                logger.info(
                    f"✨ Initial audio sent to client with {len(initial_timings)} NATIVE word timings [{audio_message['modality']}]"
                )

                # Step 4: Process remaining text chunks if available
                if initial_collection_stopped_early:
                    logger.info("Processing remaining text chunks")
                    collected_chunks = []

                    try:
                        text_iterator = collect_remaining_text(streamer)

                        while True:
                            try:
                                text_chunk = await anext(text_iterator)
                                logger.info(
                                    f"Processing text chunk: '{text_chunk[:30]}...' ({len(text_chunk)} chars)"
                                )
                                collected_chunks.append(text_chunk)

                                # Generate TTS for this chunk WITH NATIVE TIMING
                                chunk_tts_task = asyncio.create_task(
                                    tts_processor.synthesize_remaining_speech_with_timing(
                                        text_chunk
                                    )
                                )
                                manager.set_task(client_id, "tts", chunk_tts_task)

                                # FIXED: Properly unpack the tuple for chunks too
                                chunk_tts_result = await chunk_tts_task
                                if (
                                    isinstance(chunk_tts_result, tuple)
                                    and len(chunk_tts_result) == 2
                                ):
                                    chunk_audio, chunk_timings = chunk_tts_result
                                else:
                                    # Fallback for legacy method
                                    chunk_audio = chunk_tts_result
                                    chunk_timings = []
                                    logger.warning(
                                        "Chunk TTS returned single value instead of tuple"
                                    )

                                logger.info(
                                    f"Chunk TTS complete: {len(chunk_audio) if chunk_audio is not None else 0} samples, {len(chunk_timings)} word timings"
                                )

                                if chunk_audio is not None and len(chunk_audio) > 0:
                                    # Convert to base64 and send to client WITH TIMING DATA
                                    audio_bytes = (
                                        (chunk_audio * 32767).astype(np.int16).tobytes()
                                    )
                                    base64_audio = base64.b64encode(audio_bytes).decode(
                                        "utf-8"
                                    )

                                    # Send chunk audio with native timing information
                                    chunk_audio_message = {
                                        "audio": base64_audio,
                                        "word_timings": chunk_timings,  # 🎉 NATIVE TIMING DATA!
                                        "sample_rate": 24000,
                                        "method": "native_kokoro_timing",
                                        "chunk": True,
                                        "modality": (
                                            "multimodal" if image_data else "audio_only"
                                        ),
                                    }
                                    manager.client_state[client_id] = "SPEAKING"
                                    await websocket.send_text(
                                        json.dumps(chunk_audio_message)
                                    )
                                    logger.info(
                                        f"✨ Chunk audio sent to client with {len(chunk_timings)} NATIVE word timings [{chunk_audio_message['modality']}]"
                                    )

                            except StopAsyncIteration:
                                logger.info("All text chunks processed")
                                break
                            except asyncio.CancelledError:
                                logger.info("Text chunk processing cancelled")
                                raise

                        # Update history with complete response
                        if collected_chunks:
                            complete_remaining_text = "".join(collected_chunks)
                            smolvlm_processor.update_history_with_complete_response(
                                transcribed_text,
                                initial_text,
                                complete_remaining_text,
                            )

                    except asyncio.CancelledError:
                        logger.info("Remaining text processing cancelled")
                        # Update history with partial response
                        if collected_chunks:
                            partial_remaining_text = "".join(collected_chunks)
                            smolvlm_processor.update_history_with_complete_response(
                                transcribed_text,
                                initial_text,
                                partial_remaining_text,
                            )
                        else:
                            smolvlm_processor.update_history_with_complete_response(
                                transcribed_text, initial_text
                            )
                        return
                else:
                    # No remaining text, just update history with initial response
                    smolvlm_processor.update_history_with_complete_response(
                        transcribed_text, initial_text
                    )

                # Signal end of audio stream
                await websocket.send_text(json.dumps({"audio_complete": True}))
                manager.client_state[client_id] = "WAITING_FOR_PLAYBACK"
                logger.info("Audio processing complete")

    except asyncio.CancelledError:
        logger.info("Audio processing cancelled")
        raise
    except Exception as e:
        logger.error(f"Error processing audio segment: {e}")
        # Add more detailed error info
        logger.error(f"Full traceback: {traceback.format_exc()}")
