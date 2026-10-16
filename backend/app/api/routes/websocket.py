

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
