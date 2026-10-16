import sys
import threading
import time
from pathlib import Path

from app.ml.project.adapters.subprocess_runner import run_subprocess, tail

PY = sys.executable


class TestRunSubprocess:
    def test_successful_command(self, tmp_path):
        result = run_subprocess(f"{PY} -c \"print('hi')\"", tmp_path, timeout_seconds=5)
        assert result.status == "success"
        assert result.exit_code == 0
        assert "hi" in result.stdout

    def test_nonzero_exit_is_failed(self, tmp_path):
        result = run_subprocess(f"{PY} -c \"import sys; sys.exit(7)\"", tmp_path, timeout_seconds=5)
        assert result.status == "failed"
        assert result.exit_code == 7

    def test_timeout(self, tmp_path):
        result = run_subprocess(f"{PY} -c \"import time; time.sleep(5)\"", tmp_path, timeout_seconds=1)
        assert result.status == "timeout"
        assert result.exit_code is None

    def test_missing_executable_is_failed_not_a_crash(self, tmp_path):
        result = run_subprocess("this-command-does-not-exist-anywhere", tmp_path, timeout_seconds=5)
        assert result.status == "failed"
        assert result.exit_code is None

    def test_cancellation_stops_a_slow_command_early(self, tmp_path):
        cancel_event = threading.Event()
        threading.Timer(0.3, cancel_event.set).start()

        started = time.monotonic()
        result = run_subprocess(f"{PY} -c \"import time; time.sleep(30)\"", tmp_path, timeout_seconds=60, cancel_event=cancel_event)
        elapsed = time.monotonic() - started

        assert result.status == "cancelled"
        assert elapsed < 10

    def test_secrets_are_redacted(self, tmp_path):
        result = run_subprocess(
            f"{PY} -c \"print('OPENAI_API_KEY=sk-shouldnotappear')\"", tmp_path, timeout_seconds=5
        )
        assert "sk-shouldnotappear" not in result.stdout
        assert "REDACTED" in result.stdout

    def test_runs_in_the_given_working_directory(self, tmp_path):
        marker = tmp_path / "marker.txt"
        marker.write_text("here")
        result = run_subprocess(f"{PY} -c \"import pathlib; print(pathlib.Path('marker.txt').read_text())\"", tmp_path, timeout_seconds=5)
        assert "here" in result.stdout


class TestTail:
    def test_short_text_is_unchanged(self):
        assert tail("hello") == "hello"

    def test_long_text_is_truncated_from_the_start(self):
        text = "x" * 5000 + "END"
        result = tail(text, limit=100)
        assert len(result) == 100
        assert result.endswith("END")
