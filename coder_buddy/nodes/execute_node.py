"""
Execute_Node — sandbox code execution node.

Installs dependencies (when present), executes the current source code
inside the configured sandbox backend, captures combined output, and
always calls ``sandbox.cleanup()`` in a ``finally`` block.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from coder_buddy.logging_utils import log_node_event

if TYPE_CHECKING:
    from coder_buddy.config import AgentConfig
    from coder_buddy.sandbox.base import SandboxBackend
    from coder_buddy.state import AgentState


def make_execute_node(sandbox: "SandboxBackend", config: "AgentConfig"):
    """
    Factory that returns an ``execute_node`` closure bound to *sandbox*
    and *config*.

    Args:
        sandbox: Configured ``SandboxBackend`` instance.
        config:  ``AgentConfig`` providing ``sandbox_timeout_seconds``.

    Returns:
        An ``execute_node(state) -> dict`` function suitable for use as a
        LangGraph node.
    """

    def execute_node(state: "AgentState") -> dict:
        """
        Execute ``state["current_code"]`` inside the sandbox.

        Steps:
        1. If ``state["dependencies"]`` is non-empty, call
           ``sandbox.install_dependencies(dependencies)``.
        2. Call ``sandbox.execute(current_code, timeout=sandbox_timeout_seconds)``.
        3. Store ``result.combined_output`` as ``execution_logs``.
        4. Set ``error_status`` from ``result.has_errors``.
        5. Call ``sandbox.cleanup()`` in a ``finally`` block.

        Returns:
            Partial ``AgentState`` dict with keys:
            ``execution_logs``, ``error_status``.
        """
        current_code: str = state["current_code"]
        dependencies: list[str] = state["dependencies"]
        timeout: float = config.sandbox_timeout_seconds

        log_node_event(node="execute_node", event="start", retry_count=state["retry_count"])

        try:
            if dependencies:
                sandbox.install_dependencies(dependencies)

            result = sandbox.execute(current_code, timeout)

            log_node_event(
                node="execute_node",
                event="end",
                retry_count=state["retry_count"],
                outcome="error" if result.has_errors else "success",
                extra={"timed_out": result.timed_out},
            )

            return {
                "execution_logs": result.combined_output,
                "error_status": result.has_errors,
            }
        finally:
            sandbox.cleanup()

    return execute_node
