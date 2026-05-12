'use client';

/**
 * usePresenceDetection.ts
 * ─────────────────────────────────────────────────────────────────────────────
 * Polls the webcam every 1.5 seconds while AIRA is in "passive" state and
 * sends a JPEG frame to the server for MediaPipe face detection.
 *
 * When the server confirms a person is at the kiosk it replies with
 * { type: "person_detected" } — handled in WebSocketContext.tsx via the
 * onPersonDetected callback.
 *
 * The hook is a no-op when serverState !== "passive" so it never interferes
 * with an active conversation.
 */

import { useEffect, useRef } from 'react';
import { CameraStreamHandle } from '@/components/CameraStream';
import { useWebSocketContext } from '@/contexts/WebSocketContext';

/** How often (ms) to capture and send a frame while AIRA is passive. */
const PRESENCE_INTERVAL_MS = 1500;

export function usePresenceDetection(
  cameraRef: React.RefObject<CameraStreamHandle | null>,
  serverState: 'passive' | 'listening' | 'processing' | 'speaking'
) {
  const { sendPresenceFrame } = useWebSocketContext();

  // Keep a stable ref to sendPresenceFrame to avoid re-creating the interval
  // on every render (sendPresenceFrame is already memoised with useCallback
  // in WebSocketContext, but this is an extra safety net).
  const sendRef = useRef(sendPresenceFrame);
  useEffect(() => {
    sendRef.current = sendPresenceFrame;
  }, [sendPresenceFrame]);

  useEffect(() => {
    // Only send frames when AIRA is idle (waiting for someone to approach)
    if (serverState !== 'passive') return;

    const intervalId = setInterval(() => {
      // captureFrame() returns a base64 JPEG string, or null if camera is off
      const frame = cameraRef.current?.captureFrame();
      if (frame) {
        sendRef.current(frame);
      }
    }, PRESENCE_INTERVAL_MS);

    return () => clearInterval(intervalId);
  }, [serverState, cameraRef]);
}
