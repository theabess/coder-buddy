"""
LangGraph StateGraph construction for Coder Buddy.

This module contains the ``build_graph`` factory that wires all nodes
and conditional edges into a compiled ``StateGraph``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.graph import END, StateGraph

from coder_buddy.nodes.evaluator import evaluator, evaluator_router
from coder_buddy.nodes.execute_node import make_execute_node
from coder_buddy.nodes.post_process import make_post_process_node
from coder_buddy.nodes.refactor_node import make_refactor_node
from coder_buddy.nodes.test_node import make_test_node
from coder_buddy.nodes.write_node import make_write_node
from coder_buddy.state import AgentState

if TYPE_CHECKING:
    from coder_buddy.config import AgentConfig
    from coder_buddy.llm.client import LLMClient
    from coder_buddy.sandbox.base import SandboxBackend


def _make_re_execute_node(sandbox: "SandboxBackend", config: "AgentConfig"):
    """
    Build a re-execute node that wraps ``execute_node`` logic.

    After ``refactor_node`` produces a revised ``current_code``, this node
    re-runs the code in the sandbox.  If execution fails, it restores
    ``pre_refactor_code`` as ``current_code`` and sets a warning so the
    caller knows the refactored version was discarded.

    Args:
        sandbox: Concrete ``SandboxBackend`` instance.
        config:  ``AgentConfig`` providing ``sandbox_timeout_seconds``.

    Returns:
        A ``re_execute_node(state) -> dict`` function suitable for use as
        a LangGraph node.
    """
    # Reuse the same execute logic
    _execute = make_execute_node(sandbox, config)

    def re_execute_node(state: "AgentState") -> dict:
        """
        Re-execute the refactored code and fall back to pre-refactor on failure.

        Steps:
        1. Run the execute node logic on the current (refactored) code.
        2. If execution fails (``error_status=True``), restore
           ``pre_refactor_code`` as ``current_code`` and attach a warning.
        3. Return the merged partial state dict.

        Returns:
            Partial ``AgentState`` dict with keys:
            ``execution_logs``, ``error_status``, and optionally
            ``current_code`` and ``warning`` when the refactored code fails.
        """
        result = _execute(state)

        if result.get("error_status", False):
            # Refactored code failed — restore the last known-good code
            pre_refactor_code: str | None = state.get("pre_refactor_code")  # type: ignore[call-overload]
            if pre_refactor_code is not None:
                result["current_code"] = pre_refactor_code
            result["warning"] = (
                "Refactored code failed re-execution. "
                "Returning the last known-working version of the code."
            )

        return result

    return re_execute_node


def build_graph(
    sandbox: "SandboxBackend",
    llm_client: "LLMClient",
    config: "AgentConfig",
):
    """
    Construct and compile the LangGraph ``StateGraph``.

    Nodes wired:
        - ``write_node``       — LLM code generation
        - ``execute_node``     — sandbox execution
        - ``evaluator``        — conditional router (not a node)
        - ``refactor_node``    — LLM code refactoring
        - ``re_execute``       — re-runs sandbox after refactor
        - ``post_process``     — explanation + confidence scoring
        - ``test_node``        — pytest suite generation (when enabled)

    Conditional edges from ``evaluator``:
        - ``"retry"``    → ``write_node``
        - ``"refactor"`` → ``refactor_node``
        - ``"fail"``     → END

    Graph flow::

        write_node → execute_node → evaluator
            evaluator "retry"    → write_node
            evaluator "refactor" → refactor_node
            evaluator "fail"     → END
        refactor_node → re_execute
        re_execute → post_process
        post_process → test_node  (when test_generation_enabled)
        post_process → END        (when test_generation_enabled=False)
        test_node → END

    Args:
        sandbox:    Concrete ``SandboxBackend`` instance.
        llm_client: Configured ``LLMClient`` instance.
        config:     ``AgentConfig`` controlling feature flags and limits.

    Returns:
        A compiled LangGraph graph ready for ``graph.invoke()``.
    """
    # ------------------------------------------------------------------ #
    # Build node callables by injecting dependencies via factories
    # ------------------------------------------------------------------ #
    write_node_fn = make_write_node(llm_client, config)
    execute_node_fn = make_execute_node(sandbox, config)
    refactor_node_fn = make_refactor_node(llm_client, config)
    re_execute_node_fn = _make_re_execute_node(sandbox, config)
    post_process_node_fn = make_post_process_node(llm_client, config)
    test_node_fn = make_test_node(sandbox, llm_client, config)

    # ------------------------------------------------------------------ #
    # Construct the StateGraph
    # ------------------------------------------------------------------ #
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("write_node", write_node_fn)
    graph.add_node("execute_node", execute_node_fn)
    graph.add_node("evaluator", evaluator)
    graph.add_node("refactor_node", refactor_node_fn)
    graph.add_node("re_execute", re_execute_node_fn)
    graph.add_node("post_process", post_process_node_fn)
    graph.add_node("test_node", test_node_fn)

    # ------------------------------------------------------------------ #
    # Entry point
    # ------------------------------------------------------------------ #
    graph.set_entry_point("write_node")

    # ------------------------------------------------------------------ #
    # Linear edges
    # ------------------------------------------------------------------ #
    graph.add_edge("write_node", "execute_node")
    graph.add_edge("execute_node", "evaluator")
    graph.add_edge("refactor_node", "re_execute")
    graph.add_edge("re_execute", "post_process")

    # post_process → test_node → END  (when test generation is enabled)
    # post_process → END              (when test generation is disabled)
    if config.test_generation_enabled:
        graph.add_edge("post_process", "test_node")
        graph.add_edge("test_node", END)
    else:
        graph.add_edge("post_process", END)

    # ------------------------------------------------------------------ #
    # Conditional edges from evaluator node via evaluator_router
    # ------------------------------------------------------------------ #
    graph.add_conditional_edges(
        "evaluator",
        evaluator_router,
        {
            "retry": "write_node",
            "refactor": "refactor_node",
            "fail": END,
        },
    )

    return graph.compile()
