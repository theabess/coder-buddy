"""
Structured JSON logging utilities for Coder Buddy.

Every node emits two log entries (``event="start"`` and ``event="end"``)
via ``log_node_event``.  All entries are serialised to JSON and emitted
to the ``coder_buddy`` logger at ``INFO`` level so they can be captured
by any standard ``logging`` handler.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Literal

_logger = logging.getLogger("coder_buddy")


def log_node_event(
    node: str,
    event: Literal["start", "end"],
    retry_count: int,
    outcome: str | None = None,
    extra: dict | None = None,
) -> None:
    """
    Emit a structured JSON log entry for a node lifecycle event.

    The emitted entry always contains:
    - ``ts``          — Unix timestamp (float)
    - ``node``        — name of the node
    - ``event``       — ``"start"`` or ``"end"``
    - ``retry_count`` — current retry cycle count
    - ``outcome``     — optional routing outcome (e.g. ``"retry"``, ``"refactor"``)

    Any additional key/value pairs in *extra* are merged into the top-level
    entry dict.

    Args:
        node: Name of the node emitting the event.
        event: ``"start"`` when the node begins; ``"end"`` when it finishes.
        retry_count: Current value of ``AgentState["retry_count"]``.
        outcome: Optional routing outcome string.
        extra: Optional dict of additional fields to include in the log entry.
    """
    entry: dict = {
        **(extra or {}),
        "ts": time.time(),
        "node": node,
        "event": event,
        "retry_count": retry_count,
        "outcome": outcome,
    }
    _logger.info(json.dumps(entry))


def _extract_error_summary(logs: str) -> str:
    """
    Extract a concise error summary from sandbox execution logs.

    Returns the last traceback line if one is present (e.g.
    ``"NameError: name 'x' is not defined"``), otherwise the first
    non-empty line of *logs*.  The result is truncated to 500 characters.

    Args:
        logs: Combined stdout/stderr string from the sandbox.

    Returns:
        A short, human-readable error summary (≤ 500 characters).
    """
    lines = logs.splitlines()

    # Find the last traceback line: the line immediately following a
    # "Traceback (most recent call last):" block is the exception line,
    # which appears as the very last non-empty line after the header.
    last_exception_line: str | None = None
    in_traceback = False
    for line in lines:
        if line.strip().startswith("Traceback (most recent call last):"):
            in_traceback = True
            continue
        if in_traceback:
            stripped = line.strip()
            if stripped:
                # Keep updating — we want the *last* exception line
                last_exception_line = stripped

    if last_exception_line is not None:
        return last_exception_line[:500]

    # No traceback found — return the first non-empty line
    for line in lines:
        stripped = line.strip()
        if stripped:
            return stripped[:500]

    return ""[:500]
