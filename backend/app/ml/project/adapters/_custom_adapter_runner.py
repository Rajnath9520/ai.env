#!/usr/bin/env python3
"""Standalone runner for Section 11's ".ai-env/adapter.py" custom adapter
contract. Invoked as a subprocess (never imported) by
`app.ml.project.adapters.custom_adapter.CustomAdapter`, so a bug or crash
in the user's own adapter code can't take down the ai.env process that
launched it -- the same isolation `CommandAdapter` already gives a
project's own train.py/evaluate.py.

Deliberately stdlib-only, with no import of anything under `app.*`: this
file has to work when only it (not the rest of this repository) is next
to it on disk, and it is a hard requirement that the untrusted code it
loads runs in a process with as little of ai.env's own code mapped into
it as possible.

Usage:
    python _custom_adapter_runner.py <adapter_path> <action> <output_path> [<config_path>]

<action> is one of: describe, train, evaluate, predict.

Contract the user's `.ai-env/adapter.py` must satisfy: a module-level
`get_adapter()` function returning an object. Whichever of `train(config)`,
`evaluate(config)`, `predict(config)` methods that object defines are the
capabilities this adapter has -- `describe` reports exactly that, without
calling any of them.

Always writes exactly one JSON object to <output_path>, even on failure:
    {"status": "success", "result": <whatever the method returned>}
    {"status": "success", "capabilities": {"train": bool, "evaluate": bool, "predict": bool}}  # describe only
    {"status": "error", "message": "<what went wrong>"}
Never raises past its own top-level try/except -- a malformed adapter.py,
a missing get_adapter(), or a method that itself raises is reported in
the output file as data, not as an uncaught traceback the caller has to
tell apart from "the runner itself is broken."
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _write(output_path: Path, payload: dict) -> None:
    output_path.write_text(json.dumps(payload))


def main() -> None:
    if len(sys.argv) < 4:
        print(
            "usage: _custom_adapter_runner.py <adapter_path> <action> <output_path> [<config_path>]",
            file=sys.stderr,
        )
        sys.exit(2)

    adapter_path = Path(sys.argv[1])
    action = sys.argv[2]
    output_path = Path(sys.argv[3])
    config_path = Path(sys.argv[4]) if len(sys.argv) > 4 else None

    try:
        if not adapter_path.is_file():
            _write(output_path, {"status": "error", "message": f"{adapter_path} not found"})
            return

        spec = importlib.util.spec_from_file_location("_ai_env_custom_adapter", adapter_path)
        if spec is None or spec.loader is None:
            _write(
                output_path,
                {"status": "error", "message": f"could not load {adapter_path} as a Python module"},
            )
            return
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # the untrusted part -- everything above/below is just plumbing

        if not hasattr(module, "get_adapter"):
            _write(output_path, {"status": "error", "message": "adapter module must define get_adapter()"})
            return

        adapter = module.get_adapter()

        if action == "describe":
            _write(
                output_path,
                {
                    "status": "success",
                    "capabilities": {
                        "train": callable(getattr(adapter, "train", None)),
                        "evaluate": callable(getattr(adapter, "evaluate", None)),
                        "predict": callable(getattr(adapter, "predict", None)),
                    },
                },
            )
            return

        method = getattr(adapter, action, None)
        if not callable(method):
            _write(output_path, {"status": "error", "message": f"adapter has no {action}() method"})
            return

        config: dict = {}
        if config_path is not None and config_path.is_file():
            config = json.loads(config_path.read_text())

        result = method(config)
        try:
            json.dumps(result)  # fail here, explicitly, rather than write unparseable output
        except TypeError as exc:
            _write(
                output_path,
                {"status": "error", "message": f"{action}() returned something not JSON-serializable: {exc}"},
            )
            return

        _write(output_path, {"status": "success", "result": result})

    except Exception as exc:  # the user's adapter code (import or call) raised
        try:
            _write(output_path, {"status": "error", "message": f"{type(exc).__name__}: {exc}"})
        except Exception:
            print(f"_custom_adapter_runner: failed to write output: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
