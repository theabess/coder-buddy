"""
AgentState TypedDict definition for Coder Buddy.

The ``AgentState`` is the single shared data structure that flows through
every node in the LangGraph ``StateGraph``.  Each node receives the full
state and returns a *partial* dict containing only the fields it modifies;
LangGraph merges the partial dict back into the state automatically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from coder_buddy.models import HistoryEntry, TokenUsage

if TYPE_CHECKING:
    pass


class AgentState(TypedDict):
    """Shared state propagated through the LangGraph StateGraph."""

    # ------------------------------------------------------------------ #
    # Core workflow fields
    # ------------------------------------------------------------------ #
    user_prompt: str
    """Original natural language request from the user."""

    current_code: str
    """Most recently generated source code."""

    execution_logs: str
    """Most recent combined stdout/stderr from the sandbox."""

    error_status: bool
    """``True`` if the last sandbox execution produced errors."""

    retry_count: int
    """Number of write-execute-check cycles completed so far."""

    # ------------------------------------------------------------------ #
    # Code artifact metadata
    # ------------------------------------------------------------------ #
    dependencies: list[str]
    """Python package dependencies from the most recent ``CodeArtifact``."""

    file_name: str
    """Intended filename for the generated script (e.g. ``"main.py"``)."""

    language: str
    """Programming language — always ``"python"`` in V1."""

    # ------------------------------------------------------------------ #
    # Post-processing outputs
    # ------------------------------------------------------------------ #
    explanation: "str | None"
    """Plain-language explanation of the code (or ``None``)."""

    test_code: "str | None"
    """Generated pytest suite (or ``None``)."""

    test_logs: "str | None"
    """Test runner output (or ``None``)."""

    confidence_score: "int | None"
    """Self-rated confidence score in the range ``[1, 5]`` (or ``None``)."""

    refactor_diff: "str | None"
    """Unified diff produced by ``Refactor_Node`` (or ``None``)."""

    # ------------------------------------------------------------------ #
    # Observability
    # ------------------------------------------------------------------ #
    token_usage: "TokenUsage"
    """Accumulated token counts per node across the entire run."""

    session_history: "list[HistoryEntry]"
    """Bounded session history slice injected at the start of each run."""

    # ------------------------------------------------------------------ #
    # Control
    # ------------------------------------------------------------------ #
    max_retries: int
    """Configurable upper limit on retry cycles (default ``5``)."""

    pre_refactor_code: "str | None"
    """Saved pre-refactor source code used as a fallback."""

    warning: "str | None"
    """Optional warning message set when a fallback or degraded path is taken
    (e.g. refactored code failed re-execution, refactor timed out)."""

    _route: "str | None"
    """Internal routing decision set by the evaluator node.
    One of ``"retry"``, ``"refactor"``, or ``"fail"``."""
