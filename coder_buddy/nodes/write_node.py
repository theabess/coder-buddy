"""
Write_Node — LLM-based code generation node.

Builds an LLM prompt from the user prompt, session history (last N
entries), and execution logs (when retrying), then calls
``LLMClient.generate(CodeArtifact)`` to produce the next code artifact.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from coder_buddy.logging_utils import log_node_event
from coder_buddy.models import CodeArtifact

if TYPE_CHECKING:
    from coder_buddy.config import AgentConfig
    from coder_buddy.llm.client import LLMClient
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


def make_write_node(llm_client: "LLMClient", config: "AgentConfig"):
    """
    Factory that returns a ``write_node`` closure bound to *llm_client*
    and *config*.

    Args:
        llm_client: Configured ``LLMClient`` instance.
        config:     ``AgentConfig`` controlling session history context size.

    Returns:
        A ``write_node(state) -> dict`` function suitable for use as a
        LangGraph node.
    """

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
        retry_count: int = state["retry_count"]
        user_prompt: str = state["user_prompt"]
        session_history = state["session_history"]
        execution_logs: str = state.get("execution_logs", "")  # type: ignore[call-overload]
        token_usage = state["token_usage"]

        log_node_event(
            node="write_node",
            event="start",
            retry_count=retry_count,
        )

        # Build the LLM prompt
        prompt = _build_prompt(
            user_prompt=user_prompt,
            session_history=session_history,
            execution_logs=execution_logs,
            retry_count=retry_count,
            session_history_context_n=config.session_history_context_n,
        )

        # Call the LLM
        artifact, token_record = llm_client.generate(prompt, CodeArtifact)

        # Accumulate token usage
        new_token_usage = token_usage.model_copy(update={"write_node": token_record})

        log_node_event(
            node="write_node",
            event="end",
            retry_count=retry_count,
            outcome="generated",
            extra={"file_name": artifact.file_name, "language": artifact.language},
        )

        return {
            "current_code": artifact.source_code,
            "dependencies": artifact.dependencies,
            "file_name": artifact.file_name,
            "language": artifact.language,
            "token_usage": new_token_usage,
        }

    return write_node


def _build_prompt(
    user_prompt: str,
    session_history: list,
    execution_logs: str,
    retry_count: int,
    session_history_context_n: int,
) -> str:
    """
    Construct the full LLM prompt string.

    Includes:
    - A "prior context" block with the last *session_history_context_n*
      history entries (when history is non-empty).
    - A "[Reference code]" block when the user prompt references prior work
      (detected via ``_has_prior_reference``) and ``session_history`` is
      non-empty.
    - The current user request.
    - Execution logs appended when *retry_count* > 0.

    Args:
        user_prompt:               The raw user request string.
        session_history:           Full session history list (``HistoryEntry`` objects).
        execution_logs:            Combined stdout/stderr from the last sandbox run.
        retry_count:               Current retry cycle count.
        session_history_context_n: Number of history entries to include.

    Returns:
        The complete prompt string to send to the LLM.
    """
    parts: list[str] = []

    # --- Prior context block ---
    n = session_history_context_n
    # Guard: -0 == 0 in Python, so list[-0:] returns the full list.
    # When n == 0 we explicitly want no history injected.
    if n > 0 and session_history:
        recent_history = session_history[-n:]
    else:
        recent_history = []

    if recent_history:
        parts.append(f"[Prior context — last {len(recent_history)} interactions]")
        for entry in recent_history:
            parts.append("---")
            parts.append(f"Prompt: {entry.prompt}")
            parts.append("Code:")
            parts.append(f"```python\n{entry.source_code}\n```")
        parts.append("---")
        parts.append("[End prior context]")
        parts.append("")

    # --- Reference code block ---
    # When the user prompt references prior work and history is available,
    # inject the most recent source_code as an explicit reference block so
    # the LLM can resolve the reference unambiguously.
    if _has_prior_reference(user_prompt) and session_history:
        most_recent_code = session_history[-1].source_code
        parts.append("[Reference code — most recent script]")
        parts.append(f"```python\n{most_recent_code}\n```")
        parts.append("[End reference code]")
        parts.append("")

    # --- Current request ---
    parts.append(f"Current request: {user_prompt}")

    # --- Retry: append execution logs ---
    if retry_count > 0 and execution_logs:
        parts.append("")
        parts.append(
            "The previous attempt produced the following output/errors. "
            "Please fix the issues:"
        )
        parts.append("")
        parts.append(execution_logs)

    return "\n".join(parts)


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
