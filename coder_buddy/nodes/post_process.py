"""
Post-Process Node — explanation and confidence scoring.

Conditionally invokes the LLM for a plain-language explanation of the
final code (when ``explanation_enabled``) and always invokes the LLM for
a confidence score on successful runs.  Confidence scores outside ``[1, 5]``
are clamped; scores ≤ 2 trigger a warning in the response.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from coder_buddy.state import AgentState


def post_process_node(state: "AgentState") -> dict:
    """
    Generate explanation and confidence score for the final code.

    Steps:
    1. If ``state["explanation_enabled"]`` is ``True``, call the LLM for
       a plain-language explanation.
    2. Call the LLM for a confidence score (always on success).
    3. Clamp the confidence score to ``[1, 5]``.
    4. If ``confidence_score <= 2``, include a warning in the returned dict.

    Emits ``log_node_event`` at the start and end (including
    ``confidence_score`` in the end entry).

    Returns:
        Partial ``AgentState`` dict with keys:
        ``explanation``, ``confidence_score``, ``token_usage``.
        May also include ``warning`` when confidence is low.
    """
    ...
