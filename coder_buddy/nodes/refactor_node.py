"""
Refactor_Node — LLM-based code refactoring node.

Calls the LLM to produce a refactored version of the current code,
computes a unified diff between the old and new versions, and falls back
to the pre-refactor code if the LLM call times out (> 60 seconds).
"""

from __future__ import annotations

import difflib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from coder_buddy.state import AgentState


def refactor_node(state: "AgentState") -> dict:
    """
    Refactor ``state["current_code"]`` via the LLM.

    Steps:
    1. Save ``state["current_code"]`` as ``pre_refactor_code``.
    2. Call ``LLMClient.generate(CodeArtifact)`` with a 60-second timeout.
    3. On timeout, return pre-refactor code unchanged with a warning.
    4. Compute unified diff via ``compute_unified_diff``.

    Emits ``log_node_event`` at the start and end (including
    ``refactor_diff`` in the end entry).

    Returns:
        Partial ``AgentState`` dict with keys:
        ``current_code``, ``refactor_diff``, ``pre_refactor_code``,
        ``token_usage``.
    """
    ...


def compute_unified_diff(before: str, after: str, file_name: str) -> str:
    """
    Compute a unified diff between *before* and *after*.

    Uses ``difflib.unified_diff`` with ``fromfile="a/{file_name}"`` and
    ``tofile="b/{file_name}"``.

    Args:
        before: Original source code string.
        after: Refactored source code string.
        file_name: Filename used in the diff header.

    Returns:
        Unified diff string, or ``""`` if *before* and *after* are identical.
    """
    before_lines = before.splitlines(keepends=True)
    after_lines = after.splitlines(keepends=True)
    diff = difflib.unified_diff(
        before_lines,
        after_lines,
        fromfile=f"a/{file_name}",
        tofile=f"b/{file_name}",
    )
    return "".join(diff)
