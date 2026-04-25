"""
Evaluator — conditional edge router for the LangGraph StateGraph.

The evaluator is *not* a node; it is a conditional edge function that
inspects the current ``AgentState`` and returns a routing string that
LangGraph uses to select the next node.
"""

from __future__ import annotations

from typing import Literal

from coder_buddy.logging_utils import _extract_error_summary, log_node_event
from coder_buddy.state import AgentState


def evaluator(state: AgentState) -> Literal["retry", "refactor", "fail"]:
    """
    Determine the next step in the agent cycle.

    Routing logic:
    - Return ``"fail"``    if ``retry_count >= max_retries``.
    - Return ``"retry"``   if ``error_status`` is ``True``
      (and increments ``retry_count``).
    - Return ``"refactor"`` if ``error_status`` is ``False``.

    When routing to ``"retry"``, emits a structured log entry containing
    the error summary extracted from ``execution_logs`` via
    ``_extract_error_summary``.

    Args:
        state: Current ``AgentState``.

    Returns:
        One of ``"retry"``, ``"refactor"``, or ``"fail"``.
    """
    if state["retry_count"] >= state["max_retries"]:
        log_node_event(
            node="evaluator",
            event="end",
            retry_count=state["retry_count"],
            outcome="fail",
        )
        return "fail"

    if state["error_status"]:
        log_node_event(
            node="evaluator",
            event="end",
            retry_count=state["retry_count"],
            outcome="retry",
            extra={"error_summary": _extract_error_summary(state["execution_logs"])},
        )
        state["retry_count"] += 1
        return "retry"

    log_node_event(
        node="evaluator",
        event="end",
        retry_count=state["retry_count"],
        outcome="refactor",
    )
    return "refactor"
