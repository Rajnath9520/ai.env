"""Section 59/60 -- the minimum safeguards required before any Mode B
component touches a filesystem path or captures process output.

Kept as small, independently-testable functions (not folded into
`CommandAdapter`) since every future adapter -- and eventually the Local
Agent itself (Section 5's item 21: "Enforcing working-directory
restrictions") -- needs the exact same two checks.
"""

from __future__ import annotations

import re
from pathlib import Path

# Section 60's named examples plus the general shape of a secret-bearing
# line ("KEY=...", "token: ..."). Intentionally broad/generic rather than
# an exhaustive list of provider-specific names, since new providers (and
# new env var names) appear constantly and Section 60's requirement is
# "never expose secrets", not "never expose these six specific names".
_SECRET_PATTERN = re.compile(
    r"(?i)\b([A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD)[A-Z0-9_]*)\s*[:=]\s*\S+"
)


class PathTraversalError(ValueError):
    """Raised when a configured relative path would resolve outside its
    project root (Section 59, items 4/5)."""


def resolve_within(root: Path, relative: str) -> Path:
    """Resolve `relative` against `root` and guarantee the result is still
    inside `root`. This is the one place Mode B turns a user-configured
    path (working directory, artifact path, dataset path, metrics file)
    into something actually used -- every adapter should route through it
    rather than joining paths directly."""
    root = Path(root).resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise PathTraversalError(
            f"{relative!r} resolves to {candidate}, which is outside project root {root}"
        ) from exc
    return candidate


def redact_secrets(text: str) -> str:
    """Replace anything that looks like `SOME_KEY=value` or `token: value`
    with a redacted placeholder before captured stdout/stderr is stored or
    displayed (Section 60: "Redact secret-like values from stdout/stderr").
    """
    if not text:
        return text
    return _SECRET_PATTERN.sub(lambda m: f"{m.group(1)}=***REDACTED***", text)
