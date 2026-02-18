import asyncio
import base64
import json
import logging
import time
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from managers.connection_manager import manager
from models.whisper_processor import WhisperProcessor
from models.smolvlm_processor import SmolVLMProcessor
from models.tts_processor import KokoroTTSProcessor
from services.audio_service import process_audio_segment

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time multimodal interaction"""
    await manager.connect(websocket, client_id)

    # Get instances of processors
    whisper_processor = WhisperProcessor.get_instance()
    smolvlm_processor = SmolVLMProcessor.get_instance()
    tts_processor = KokoroTTSProcessor.get_instance()

    try:
        # Send initial configuration confirmation
        await websocket.send_text(
            json.dumps({"status": "connected", "client_id": client_id})
        )

        async def send_keepalive():
            """Send periodic keepalive pings"""
            while True:
                try:
                    await websocket.send_text(
                        json.dumps({"type": "ping", "timestamp": time.time()})
                    )
                    await asyncio.sleep(10)  # Send ping every 10 seconds
                except Exception:
                    break

        async def receive_and_process():
            """Receive and process messages from the client"""
            try:
                while True:
                    data = await websocket.receive_text()
                    try:
                        message = json.loads(data)

                        # Handle complete audio segments from frontend
                        if "audio_segment" in message:
                            # Cancel any current processing
                            if manager.client_state.get(client_id) != "IDLE":
                                logger.info(
                                    f"Ignoring audio because state is {manager.client_state.get(client_id)}"
                                )
                                continue
                            await manager.cancel_current_tasks(client_id)

                            # Decode audio data
                            audio_data = base64.b64decode(message["audio_segment"])

                            # Check if image is also included
                            image_data = None
                            if "image" in message:
                                image_data = base64.b64decode(message["image"])
                                logger.info(
                                    f"Received audio+image: audio={len(audio_data)} bytes, image={len(image_data)} bytes"
                                )
                            else:
                                logger.info(
                                    f"Received audio-only: {len(audio_data)} bytes"
                                )

                            # Start processing the audio segment with optional image
                            processing_task = asyncio.create_task(
                                process_audio_segment(
                                    websocket,
                                    client_id,
                                    audio_data,
                                    image_data,
                                    manager,
                                    whisper_processor,
                                    smolvlm_processor,
                                    tts_processor,
                                )
                            )
                            manager.set_task(client_id, "processing", processing_task)

                        # Handle standalone images (only if not currently processing)
                        elif "image" in message:
                            if not (
                                client_id in manager.current_tasks
                                and manager.current_tasks[client_id]["processing"]
                                and not manager.current_tasks[client_id][
                                    "processing"
                                ].done()
                            ):
                                image_data = base64.b64decode(message["image"])
                                manager.update_stats("images_received")

                                # Save standalone image
                                saved_path = manager.image_manager.save_image(
                                    image_data, client_id, "standalone"
                                )
                                if saved_path:
                                    verification = manager.image_manager.verify_image(
                                        saved_path
                                    )
                                    logger.info(
                                        f"📸 Standalone image saved and verified: {verification}"
                                    )

                                await smolvlm_processor.set_image(image_data)
                                logger.info("Image updated")

                        # Handle realtime input (for backward compatibility)
                        elif "realtime_input" in message:
                            for chunk in message["realtime_input"]["media_chunks"]:
                                if chunk["mime_type"] == "audio/pcm":
                                    # Treat as complete audio segment
                                    await manager.cancel_current_tasks(client_id)

                                    audio_data = base64.b64decode(chunk["data"])
                                    processing_task = asyncio.create_task(
                                        process_audio_segment(
                                            websocket,
                                            client_id,
                                            audio_data,
                                            None,
                                            manager,
                                            whisper_processor,
                                            smolvlm_processor,
                                            tts_processor,
                                        )
                                    )
                                    manager.set_task(
                                        client_id, "processing", processing_task
                                    )

                                elif chunk["mime_type"] == "image/jpeg":
                                    # Only process image if not currently processing audio
                                    if not (
                                        client_id in manager.current_tasks
                                        and manager.current_tasks[client_id][
                                            "processing"
                                        ]
                                        and not manager.current_tasks[client_id][
                                            "processing"
                                        ].done()
                                    ):
                                        image_data = base64.b64decode(chunk["data"])
                                        manager.update_stats("images_received")

                                        # Save image from realtime input
                                        saved_path = manager.image_manager.save_image(
                                            image_data, client_id, "realtime"
                                        )
                                        if saved_path:
                                            verification = (
                                                manager.image_manager.verify_image(
                                                    saved_path
                                                )
                                            )
                                            logger.info(
                                                f"📸 Realtime image saved and verified: {verification}"
                                            )

                                        await smolvlm_processor.set_image(image_data)

                    except json.JSONDecodeError as e:
                        logger.error(f"Error decoding JSON: {e}")
                        await websocket.send_text(
                            json.dumps({"error": "Invalid JSON format"})
                        )
                    except KeyError as e:
                        logger.error(f"Missing key in message: {e}")
                        await websocket.send_text(
                            json.dumps({"error": f"Missing required field: {e}"})
                        )
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                        await websocket.send_text(
                            json.dumps({"error": f"Processing error: {str(e)}"})
                        )

            except WebSocketDisconnect:
                logger.info("WebSocket connection closed during receive loop")

        # Run tasks concurrently
        receive_task = asyncio.create_task(receive_and_process())
        keepalive_task = asyncio.create_task(send_keepalive())

        # Wait for any task to complete (usually due to disconnection or error)
        done, pending = await asyncio.wait(
            [receive_task, keepalive_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Cancel pending tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Log results of completed tasks
        for task in done:
            try:
                result = task.result()
            except Exception as e:
                logger.error(f"Task finished with error: {e}")

    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected normally")
    except Exception as e:
        logger.error(f"WebSocket session error for client {client_id}: {e}")
    finally:
        # Cleanup
        logger.info(f"Cleaning up resources for client {client_id}")
        await manager.cancel_current_tasks(client_id)
        manager.disconnect(client_id)
