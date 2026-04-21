"""
Execute_Node — sandbox code execution node.

Installs dependencies (when present), executes the current source code
inside the configured sandbox backend, captures combined output, and
always calls ``sandbox.cleanup()`` in a ``finally`` block.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from coder_buddy.state import AgentState


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

    Emits ``log_node_event`` at the start and end of execution.

    Returns:
        Partial ``AgentState`` dict with keys:
        ``execution_logs``, ``error_status``.
    """
    ...
