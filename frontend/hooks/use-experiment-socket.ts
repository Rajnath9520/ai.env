"use client";

import { useEffect, useRef, useState } from "react";
import { wsUrlForExperiment } from "@/lib/api-client";
import type { TrainingTick } from "@/types/api";

const MAX_BUFFERED_TICKS = 500;

export interface ExperimentSocketState {
  latestTick: TrainingTick | null;
  ticks: TrainingTick[];
  connected: boolean;
  closed: boolean;
}

/** Subscribes to `/ws/experiments/{id}`, buffering the most recent ticks
 * (capped, per Section 25's "don't stream huge data" -- the payloads are
 * already small, this cap just bounds client-side memory for very long
 * runs). Reconnection is intentionally not automatic: once the server
 * closes the socket (training finished), `closed` flips true and stays
 * true -- there is nothing more to stream for a completed experiment.
 */
export function useExperimentSocket(experimentId: number | null, enabled: boolean): ExperimentSocketState {
  const [ticks, setTicks] = useState<TrainingTick[]>([]);
  const [connected, setConnected] = useState(false);
  const [closed, setClosed] = useState(false);
  const socketRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!enabled || experimentId == null) return;

    setTicks([]);
    setClosed(false);

    const socket = new WebSocket(wsUrlForExperiment(experimentId));
    socketRef.current = socket;

    socket.onopen = () => setConnected(true);
    socket.onclose = () => {
      setConnected(false);
      setClosed(true);
    };
    socket.onerror = () => setConnected(false);
    socket.onmessage = (event) => {
      try {
        const tick = JSON.parse(event.data) as TrainingTick;
        setTicks((prev) => {
          const next = [...prev, tick];
          return next.length > MAX_BUFFERED_TICKS ? next.slice(next.length - MAX_BUFFERED_TICKS) : next;
        });
      } catch {
        // ignore malformed frames
      }
    };

    return () => {
      socket.close();
      socketRef.current = null;
    };
  }, [experimentId, enabled]);

  return {
    latestTick: ticks.length ? ticks[ticks.length - 1] : null,
    ticks,
    connected,
    closed,
  };
}
