"""Section 20 -- the one implementation of "run a command, cancellably,
and kill the whole process group if it's cancelled or times out."
Originally lived inside `CommandAdapter`; extracted so `CustomAdapter`
(Section 11/12) can run its own subprocess -- the runner script that loads
`.ai-env/adapter.py` -- through the exact same real cancellation and
process-group-kill machinery, rather than a second, parallel
implementation of it.

Cancellation is real, not aspirational: `run_subprocess` polls the
subprocess via `Popen.communicate(timeout=...)` (the deadlock-safe idiom
the `subprocess` docs themselves recommend for pipes) rather than blocking
on a single `subprocess.run(timeout=...)`, so a caller-supplied
`threading.Event` can interrupt a run in flight. The process is started in
its own session (`start_new_session=True`, POSIX) so cancelling or timing
out kills the whole group -- everything the command itself spawned -- and
never anything else ("Never kill unrelated processes. Only manage
processes launched by AI.env").
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.ml.project.security import redact_secrets

_POLL_INTERVAL_SECONDS = 0.2
LOG_TAIL_CHARS = 4000


def tail(text: str, limit: int = LOG_TAIL_CHARS) -> str:
    return text if len(text) <= limit else text[-limit:]


@dataclass
class SubprocessResult:
    status: str  # "success" | "failed" | "timeout" | "cancelled"
    exit_code: int | None
    command: str
    started_at: str
    ended_at: str
    duration_seconds: float
    stdout: str
    stderr: str


def run_subprocess(
    command: str,
    working_directory: Path,
    timeout_seconds: float,
    cancel_event: threading.Event | None = None,
) -> SubprocessResult:
    started = datetime.now(timezone.utc)
    popen_kwargs: dict = {}
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True  # own process group -- see module docstring

    try:
        proc = subprocess.Popen(
            shlex.split(command),
            cwd=working_directory,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            **popen_kwargs,
        )
    except FileNotFoundError as exc:
        ended = datetime.now(timezone.utc)
        return SubprocessResult(
            status="failed",
            exit_code=None,
            command=command,
            started_at=started.isoformat(),
            ended_at=ended.isoformat(),
            duration_seconds=(ended - started).total_seconds(),
            stdout="",
            stderr=redact_secrets(str(exc)),
        )

    deadline = time.monotonic() + timeout_seconds
    outcome = "success"
    exited_naturally = False
    stdout = stderr = ""
    while True:
        try:
            stdout, stderr = proc.communicate(timeout=_POLL_INTERVAL_SECONDS)
            exited_naturally = True
            break
        except subprocess.TimeoutExpired:
            pass  # still running -- fall through to the cancel/deadline checks below

        if cancel_event is not None and cancel_event.is_set():
            outcome = "cancelled"
            break
        if time.monotonic() >= deadline:
            outcome = "timeout"
            break

    if not exited_naturally:
        _kill_process_group(proc)
        stdout, stderr = proc.communicate()

    ended = datetime.now(timezone.utc)
    exit_code = proc.returncode
    if outcome == "success":
        outcome = "success" if exit_code == 0 else "failed"

    return SubprocessResult(
        status=outcome,
        exit_code=exit_code if outcome != "timeout" else None,
        command=command,
        started_at=started.isoformat(),
        ended_at=ended.isoformat(),
        duration_seconds=(ended - started).total_seconds(),
        stdout=redact_secrets(stdout or ""),
        stderr=redact_secrets(stderr or ""),
    )


def _kill_process_group(proc: subprocess.Popen) -> None:
    """Escalates SIGTERM -> SIGKILL against the whole group `run_subprocess`
    started (POSIX), or just the one process on other platforms. Never
    touches a process this call didn't start (Section 20)."""
    if os.name == "posix":
        try:
            pgid = os.getpgid(proc.pid)
        except ProcessLookupError:
            return
        try:
            os.killpg(pgid, signal.SIGTERM)
            proc.wait(timeout=5)
            return
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    else:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
