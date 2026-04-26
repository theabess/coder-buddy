"""
Evaluator — node + conditional edge router for the LangGraph StateGraph.

The evaluator is implemented as a **proper node** (not a bare conditional
edge function) so that it can update ``retry_count`` in the LangGraph state.
Conditional edge functions in LangGraph receive a read-only snapshot of the
state; any mutations they make are discarded.  By making the evaluator a
node that returns a partial state dict, we ensure ``retry_count`` is
correctly incremented before the graph routes back to ``write_node``.

The companion ``evaluator_router`` function is the actual conditional edge
function — it reads the ``_route`` field set by the evaluator node and
returns the routing string.
"""

from __future__ import annotations

from typing import Literal

from coder_buddy.logging_utils import _extract_error_summary, log_node_event
from coder_buddy.state import AgentState


def evaluator(state: AgentState) -> dict:
    """
    Evaluator node — determines the next step and updates state.

    Routing logic:
    - Route to ``"fail"``    if ``retry_count >= max_retries``.
    - Route to ``"retry"``   if ``error_status`` is ``True``
      (and increments ``retry_count`` in the returned state update).
    - Route to ``"refactor"`` if ``error_status`` is ``False``.

    When routing to ``"retry"``, emits a structured log entry containing
    the error summary extracted from ``execution_logs`` via
    ``_extract_error_summary``.

    Args:
        state: Current ``AgentState``.

    Returns:
        Partial state dict with ``_route`` set to the routing decision,
        and ``retry_count`` incremented when routing to ``"retry"``.
    """
    if state["retry_count"] >= state["max_retries"]:
        log_node_event(
            node="evaluator",
            event="end",
            retry_count=state["retry_count"],
            outcome="fail",
        )
        return {"_route": "fail"}

    if state["error_status"]:
        log_node_event(
            node="evaluator",
            event="end",
            retry_count=state["retry_count"],
            outcome="retry",
            extra={"error_summary": _extract_error_summary(state["execution_logs"])},
        )
        return {
            "_route": "retry",
            "retry_count": state["retry_count"] + 1,
        }

    log_node_event(
        node="evaluator",
        event="end",
        retry_count=state["retry_count"],
        outcome="refactor",
    )
    return {"_route": "refactor"}


def evaluator_router(state: AgentState) -> Literal["retry", "refactor", "fail"]:
    """
    Conditional edge function that reads the routing decision from state.

    This function is used as the conditional edge function in the LangGraph
    ``StateGraph``.  It reads the ``_route`` field set by the ``evaluator``
    node and returns the routing string.

    Args:
        state: Current ``AgentState`` (after the evaluator node has run).

    Returns:
        One of ``"retry"``, ``"refactor"``, or ``"fail"``.
    """
    route = state.get("_route", "fail")  # type: ignore[call-overload]
    return route  # type: ignore[return-value]
