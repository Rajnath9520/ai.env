"use client";

import { useEffect, useRef, useState } from "react";
import type { Cell } from "@/components/grid-world/GridWorldRenderer";

/** Steps `agentPos` through `path` on an interval, looping. Kept as a
 * small standalone hook (rather than baked into GridWorldRenderer) so the
 * renderer itself stays a plain, stateless "render these props" component
 * -- reusable both for this demo animation and for a future real
 * per-step trajectory replay without changing the renderer at all.
 */
export function useAnimatedPath(path: Cell[] | null, intervalMs = 400) {
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(true);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    setIndex(0);
  }, [path]);

  useEffect(() => {
    if (!playing || !path || path.length === 0) return;
    timerRef.current = setInterval(() => {
      setIndex((prev) => (prev + 1 >= path.length ? 0 : prev + 1));
    }, intervalMs);
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [playing, path, intervalMs]);

  const agentPos = path && path.length > 0 ? path[Math.min(index, path.length - 1)] : undefined;
  const visited = path ? path.slice(0, index + 1) : [];

  return {
    agentPos,
    visited,
    playing,
    setPlaying,
    reset: () => setIndex(0),
    progress: path && path.length > 1 ? index / (path.length - 1) : 0,
  };
}
