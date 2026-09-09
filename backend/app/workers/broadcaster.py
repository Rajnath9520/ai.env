"""Phase 18 -- in-process progress broadcaster (Section 25).

Bridges the synchronous training thread (a FastAPI BackgroundTask running
`ExperimentManager.run()` in a plain Python thread) to any number of async
WebSocket connections subscribed to that experiment's updates.

Design: one bounded `queue.Queue` per (experiment_id, subscriber). Bounded
and drop-oldest-on-full rather than unbounded, per Section 25's "do not
stream huge trajectory data continuously -- stream summarized metrics":
ticks here are already small summarized dicts, but a slow/stalled
WebSocket client still shouldn't be able to make the queue grow without
bound.

Thread-safety: `publish()` is called from the training thread (sync code);
`subscribe()`/`unsubscribe()` and reading are called from the event loop
(async code, via `asyncio.to_thread` to avoid blocking on `queue.get()`).
A single `threading.Lock` guards the subscriber registry.
"""

from __future__ import annotations

import queue
import threading

MAX_QUEUE_SIZE = 200
_SENTINEL = object()  # signals "no more ticks are coming" to a subscriber


class ExperimentBroadcaster:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: dict[int, list[queue.Queue]] = {}

    def subscribe(self, experiment_id: int) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        with self._lock:
            self._subscribers.setdefault(experiment_id, []).append(q)
        return q

    def unsubscribe(self, experiment_id: int, q: queue.Queue) -> None:
        with self._lock:
            subscribers = self._subscribers.get(experiment_id, [])
            if q in subscribers:
                subscribers.remove(q)
            if not subscribers:
                self._subscribers.pop(experiment_id, None)

    def publish(self, experiment_id: int, tick: dict) -> None:
        with self._lock:
            subscribers = list(self._subscribers.get(experiment_id, []))
        for q in subscribers:
            self._put_dropping_oldest(q, tick)

    def close(self, experiment_id: int) -> None:
        """Signals every current subscriber that no more ticks are coming
        (called once the background training task finishes or fails)."""
        with self._lock:
            subscribers = list(self._subscribers.get(experiment_id, []))
        for q in subscribers:
            self._put_dropping_oldest(q, _SENTINEL)

    @staticmethod
    def _put_dropping_oldest(q: queue.Queue, item) -> None:
        try:
            q.put_nowait(item)
        except queue.Full:
            try:
                q.get_nowait()  # drop the oldest tick to make room
            except queue.Empty:
                pass
            try:
                q.put_nowait(item)
            except queue.Full:
                pass  # extremely unlikely race; drop this tick silently


# Single process-wide broadcaster instance -- matches the in-memory,
# single-process scope already documented for cancellation flags
# (app/services/experiment_service.py); a multi-worker deployment would
# need a shared pub/sub (e.g. Redis) instead.
broadcaster = ExperimentBroadcaster()
