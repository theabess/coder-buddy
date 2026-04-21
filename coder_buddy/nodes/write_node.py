"""
Write_Node — LLM-based code generation node.

Builds an LLM prompt from the user prompt, session history (last N
entries), and execution logs (when retrying), then calls
``LLMClient.generate(CodeArtifact)`` to produce the next code artifact.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from coder_buddy.state import AgentState


# Keywords that indicate the user is referencing prior work.
REFERENCE_KEYWORDS: list[str] = [
    "the script",
    "the code",
    "the function",
    "the program",
    "you just wrote",
    "from before",
    "previous",
    "last one",
    "above",
    "that script",
    "that code",
    "it faster",
    "it cleaner",
]


def write_node(state: "AgentState") -> dict:
    """
    Generate (or regenerate) source code via the LLM.

    Builds an LLM prompt from:
    - ``state["user_prompt"]``
    - ``state["session_history"]`` (last N entries, formatted as prior context)
    - ``state["execution_logs"]`` (only when ``state["retry_count"] > 0``)

    Calls ``LLMClient.generate(CodeArtifact)`` and returns a partial state
    dict with the updated code artifact fields and accumulated token usage.

    Returns:
        Partial ``AgentState`` dict with keys:
        ``current_code``, ``dependencies``, ``file_name``, ``language``,
        ``token_usage``.
    """
    ...


def _has_prior_reference(prompt: str) -> bool:
    """
    Return ``True`` if *prompt* contains a keyword that references prior work.

    Uses a lightweight keyword check against ``REFERENCE_KEYWORDS``.

    Args:
        prompt: The raw user prompt string.

    Returns:
        ``True`` if any reference keyword is found (case-insensitive).
    """
    lowered = prompt.lower()
    return any(kw in lowered for kw in REFERENCE_KEYWORDS)
