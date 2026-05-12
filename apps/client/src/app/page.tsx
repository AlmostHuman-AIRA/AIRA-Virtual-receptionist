'use client';

import { useRef, useState, useCallback } from 'react';
import Link from 'next/link';
import VoiceActivityDetector from '@/components/VoiceActivityDetector';
import TalkingHead from '@/components/TalkingHead';
import {
  CameraToggleButton,
  CameraToggleButtonHandle,
  CameraStreamHandle
} from '@/components/CameraStream';
import { useFaceVerification } from '@/hooks/useFaceVerification';
import { usePresenceDetection } from '@/hooks/usePresenceDetection';
import { useWebSocketContext } from '@/contexts/WebSocketContext';

export default function Home() {
  // ── Single camera ref shared between presence detection & face verification ──
  // CameraToggleButton owns the single CameraStream instance; it exposes
  // ensureCameraReady() AND forwards captureFrame() via the same cameraRef.
  const cameraRef = useRef<CameraStreamHandle | null>(null);
  const cameraToggleRef = useRef<CameraToggleButtonHandle | null>(null);

  // ── Server state tracking ─────────────────────────────────────────────────
  const [serverState, setServerState] = useState<
    'passive' | 'listening' | 'processing' | 'speaking'
  >('passive');

  // ── Person-detected glow indicator ────────────────────────────────────────
  const [personJustDetected, setPersonJustDetected] = useState(false);

  // ── WebSocket callbacks ───────────────────────────────────────────────────
  const { onServerState, onPersonDetected } = useWebSocketContext();

  onServerState(useCallback((state) => setServerState(state), []));

  onPersonDetected(
    useCallback(() => {
      setPersonJustDetected(true);
      setTimeout(() => setPersonJustDetected(false), 3000);
    }, [])
  );

  // ── Presence detection loop ───────────────────────────────────────────────
  usePresenceDetection(cameraRef, serverState);

  // ── Face verification ─────────────────────────────────────────────────────
  const { result, isVerifying, cameraStartupError } = useFaceVerification(
    cameraRef,
    {
      ensureCameraReady: async () =>
        (await cameraToggleRef.current?.ensureCameraReady()) ?? false
    }
  );

  return (
    <main className="relative min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8 text-center">
          <h1 className="mb-2 text-4xl font-bold text-gray-900">
            AlmostHuman AI
          </h1>
          <p className="text-lg text-gray-600">
            An AI-Based Virtual Receptionist System
          </p>
          <div className="mt-4">
            <Link
              href="/admin/employees"
              className="inline-flex rounded-md border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50"
            >
              Open Employee Photo Admin
            </Link>
          </div>
        </div>

        {/* Main Content Layout */}
        <div className="mb-8 grid grid-cols-1 gap-8 xl:grid-cols-2">
          {/* TalkingHead Component */}
          <div className="order-1">
            <div className="rounded-lg bg-white p-6 shadow-lg">
              <TalkingHead />
            </div>
          </div>

          {/* Voice Activity Detector */}
          <div className="order-2">
            <VoiceActivityDetector />
          </div>
        </div>
      </div>

      {/*
        Single floating camera widget.
        ─────────────────────────────────────────────────────────────────────
        CameraToggleButton owns the single CameraStream instance. It:
          1. Auto-starts the camera on page load (autoStart prop)
          2. Exposes ensureCameraReady() for face verification
          3. Forwards captureFrame() through cameraRef for presence detection
        This eliminates the duplicate camera window that appeared before.
      */}
      <CameraToggleButton
        ref={cameraToggleRef}
        cameraRef={cameraRef}
        autoStart
        glowActive={personJustDetected}
      />

      {/* Face verification badge */}
      {(isVerifying || result) && (
        <div className="fixed right-6 bottom-24 z-40 max-w-sm rounded-lg border bg-white p-3 shadow-xl">
          {isVerifying && (
            <div className="flex items-center gap-2 text-sm text-gray-700">
              <span className="h-2 w-2 animate-pulse rounded-full bg-amber-500" />
              Verifying identity...
            </div>
          )}

          {!isVerifying && result && (
            <div
              className={`text-sm font-medium ${
                result.verified ? 'text-green-700' : 'text-red-700'
              }`}
            >
              {result.verified
                ? `✅ Identity Confirmed — ${result.audioName}`
                : '⚠️ Identity Mismatch — please confirm'}
            </div>
          )}
        </div>
      )}

      {cameraStartupError && (
        <div className="fixed right-6 bottom-40 z-40 max-w-sm rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800 shadow-xl">
          {cameraStartupError}
        </div>
      )}
    </main>
  );
}
