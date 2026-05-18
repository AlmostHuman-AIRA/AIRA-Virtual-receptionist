# Camera-Based Presence Detection + Time Greeting — Task Tracker

## Server Tasks
- [x] Create `person_detection_service.py` (MediaPipe Face Detection singleton)
- [x] Modify `websocket_routes.py` (presence_frame handler, imports, executor, session state)
- [x] Add `mediapipe` to `pyproject.toml`
- [x] Add presence detection env vars to `.env`

## Client Tasks
- [x] Modify `WebSocketContext.tsx` (sendPresenceFrame + onPersonDetected)
- [x] Create `usePresenceDetection.ts` hook
- [x] Modify `page.tsx` (auto-camera + presence hook + green glow)
- [x] Modify `CameraStream.tsx` (auto-start camera on page load)

## Time-Based Greeting
- [x] `processor_service.py` — Good Morning/Afternoon/Evening greeting on WAKE_WORD_TRIGGERED
- [x] `query_router.py` — `_get_time_greeting()` helper for wake word path
- [x] Greeting fires for BOTH presence detection AND wake word triggers

## Verification
- [x] Code review — all paths verified end-to-end
