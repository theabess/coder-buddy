"""
Refactor_Node — LLM-based code refactoring node.

Calls the LLM to produce a refactored version of the current code,
computes a unified diff between the old and new versions, and falls back
to the pre-refactor code if the LLM call times out (> 60 seconds).
"""

from __future__ import annotations

import concurrent.futures
from typing import TYPE_CHECKING

from coder_buddy.logging_utils import log_node_event
from coder_buddy.models import CodeArtifact
from coder_buddy.utils import compute_unified_diff

if TYPE_CHECKING:
    from coder_buddy.config import AgentConfig
    from coder_buddy.llm.client import LLMClient
    from coder_buddy.state import AgentState

# Timeout in seconds for the refactor LLM call (Requirement 5.5)
_REFACTOR_TIMEOUT_SECONDS = 60


def make_refactor_node(llm_client: "LLMClient", config: "AgentConfig"):
    """
    Factory that returns a ``refactor_node`` closure bound to *llm_client*
    and *config*.

    Args:
        llm_client: Configured ``LLMClient`` instance.
        config:     ``AgentConfig`` instance (reserved for future use).

    Returns:
        A ``refactor_node(state) -> dict`` function suitable for use as a
        LangGraph node.
    """

    def refactor_node(state: "AgentState") -> dict:
        """
        Refactor ``state["current_code"]`` via the LLM.

        Steps:
        1. Save ``state["current_code"]`` as ``pre_refactor_code``.
        2. Build a refactoring prompt from the current code.
        3. Call ``LLMClient.generate(CodeArtifact)`` with a 60-second timeout
           using ``concurrent.futures.ThreadPoolExecutor``.
        4. On timeout, return pre-refactor code unchanged.
        5. Compute unified diff via ``compute_unified_diff``.

        Emits ``log_node_event`` at the start and end (including
        ``refactor_diff`` in the end entry).

        Returns:
            Partial ``AgentState`` dict with keys:
            ``current_code``, ``refactor_diff``, ``pre_refactor_code``,
            ``token_usage``.
        """
        retry_count: int = state["retry_count"]
        pre_refactor_code: str = state["current_code"]
        file_name: str = state["file_name"]
        token_usage = state["token_usage"]

        log_node_event(
            node="refactor_node",
            event="start",
            retry_count=retry_count,
        )

        prompt = _build_refactor_prompt(pre_refactor_code, file_name)

        # Run the LLM call in a thread so we can enforce a hard timeout
        timed_out = False
        artifact = None
        token_record = None

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(llm_client.generate, prompt, CodeArtifact)
            try:
                artifact, token_record = future.result(
                    timeout=_REFACTOR_TIMEOUT_SECONDS
                )
            except concurrent.futures.TimeoutError:
                timed_out = True
                # Cancel is best-effort; the thread may still be running
                future.cancel()

        if timed_out or artifact is None:
            # Timeout path: return pre-refactor code unchanged (Req 5.5)
            log_node_event(
                node="refactor_node",
                event="end",
                retry_count=retry_count,
                outcome="timeout",
                extra={"refactor_diff": ""},
            )
            return {
                "current_code": pre_refactor_code,
                "refactor_diff": "",
                "pre_refactor_code": pre_refactor_code,
                "token_usage": token_usage,
            }

        # Compute unified diff between old and new source code (Req 15.1 / 15.3)
        refactor_diff = compute_unified_diff(
            before=pre_refactor_code,
            after=artifact.source_code,
            file_name=file_name,
        )

        # Accumulate token usage for this node
        new_token_usage = token_usage.model_copy(
            update={"refactor_node": token_record}
        )

        log_node_event(
            node="refactor_node",
            event="end",
            retry_count=retry_count,
            outcome="success",
            extra={"refactor_diff": refactor_diff},
        )

        return {
            "current_code": artifact.source_code,
            "refactor_diff": refactor_diff,
            "pre_refactor_code": pre_refactor_code,
            "token_usage": new_token_usage,
        }

    return refactor_node


def _build_refactor_prompt(source_code: str, file_name: str) -> str:
    """
    Build the LLM prompt for refactoring *source_code*.

    Instructs the LLM to:
    - Add inline comments explaining non-obvious logic.
    - Improve variable naming for clarity.
    - Remove dead code and unused imports.
    - Preserve the exact same behaviour and output.

    Args:
        source_code: The working source code to refactor.
        file_name:   The filename (used for context in the prompt).

    Returns:
        The complete prompt string to send to the LLM.
    """
    return (
        f"Refactor the following Python script (`{file_name}`) to improve its "
        "readability and maintainability. Specifically:\n"
        "1. Add inline comments explaining non-obvious logic.\n"
        "2. Improve variable and function naming for clarity.\n"
        "3. Remove any dead code, unused imports, or redundant statements.\n"
        "4. Preserve the exact same behaviour and output — do NOT change what "
        "the code does.\n\n"
        "Return the refactored code as a CodeArtifact with the same "
        "`file_name`, `dependencies`, and `language` as the original.\n\n"
        f"```python\n{source_code}\n```"
    )


