"""Phase 18 -- WebSocket streaming route (Section 25).

`/ws/experiments/{id}` streams summarized training ticks as JSON, matching
Section 25's example fields: episode, reward, success_rate,
environment_id, primary_weakness, learning_progress, training_status.

Reads from the experiment's broadcaster queue via `asyncio.to_thread` so
the blocking `queue.get()` doesn't block the event loop, and exits cleanly
on client disconnect or once the broadcaster signals the run has ended.
"""

from __future__ import annotations

import asyncio
import queue

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.workers.broadcaster import _SENTINEL, broadcaster

router = APIRouter()

POLL_TIMEOUT_SECONDS = 1.0


@router.websocket("/ws/experiments/{experiment_id}")
async def stream_experiment_progress(websocket: WebSocket, experiment_id: int) -> None:
    await websocket.accept()
    q = broadcaster.subscribe(experiment_id)
    try:
        while True:
            try:
                item = await asyncio.to_thread(q.get, True, POLL_TIMEOUT_SECONDS)
            except queue.Empty:
                continue  # no tick yet; loop again so we keep noticing disconnects

            if item is _SENTINEL:
                await websocket.close()
                break
            await websocket.send_json(item)
    except WebSocketDisconnect:
        pass
    finally:
        broadcaster.unsubscribe(experiment_id, q)
