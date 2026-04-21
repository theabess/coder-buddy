"""
LangGraph StateGraph construction for Coder Buddy.

This module contains the ``build_graph`` factory that wires all nodes
and conditional edges into a compiled ``StateGraph``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from coder_buddy.config import AgentConfig
    from coder_buddy.llm.client import LLMClient
    from coder_buddy.sandbox.base import SandboxBackend


def build_graph(
    sandbox: "SandboxBackend",
    llm_client: "LLMClient",
    config: "AgentConfig",
):
    """
    Construct and compile the LangGraph ``StateGraph``.

    Nodes wired:
        - ``write_node``
        - ``execute_node``
        - ``evaluator`` (conditional router)
        - ``refactor_node``
        - ``re_execute_node``
        - ``post_process_node``
        - ``test_node`` (when enabled)

    Conditional edges from ``evaluator``:
        - ``"retry"``   → ``write_node``
        - ``"refactor"`` → ``refactor_node``
        - ``"fail"``    → END

    Args:
        sandbox: Concrete ``SandboxBackend`` instance.
        llm_client: Configured ``LLMClient`` instance.
        config: ``AgentConfig`` controlling feature flags and limits.

    Returns:
        A compiled LangGraph graph ready for ``graph.invoke()``.
    """
    ...
