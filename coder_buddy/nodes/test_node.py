"""
Test_Node — LLM-based test generation and execution node.

Generates a pytest suite via the LLM, executes it in the sandbox, and
retries up to 3 times on failure.  If all retries fail, a warning is
stored in the state but ``source_code`` is left unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from coder_buddy.state import AgentState


def test_node(state: "AgentState") -> dict:
    """
    Generate and execute a pytest suite for ``state["current_code"]``.

    Steps:
    1. Call ``LLMClient.generate`` to produce a pytest suite.
    2. Execute the suite in the sandbox.
    3. On failure, retry up to 3 times.
    4. If all retries fail, store a warning without modifying ``source_code``.

    Emits ``log_node_event`` at the start and end of node execution.

    Returns:
        Partial ``AgentState`` dict with keys:
        ``test_code``, ``test_logs``.
        May also include ``warning`` if all test retries fail.
    """
    ...
