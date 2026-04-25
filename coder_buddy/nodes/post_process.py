"""
Post-Process Node — explanation and confidence scoring.

Conditionally invokes the LLM for a plain-language explanation of the
final code (when ``explanation_enabled``) and always invokes the LLM for
a confidence score on successful runs.  Confidence scores outside ``[1, 5]``
are clamped; scores ≤ 2 trigger a warning in the response.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from coder_buddy.logging_utils import log_node_event

if TYPE_CHECKING:
    from coder_buddy.config import AgentConfig
    from coder_buddy.llm.client import LLMClient
    from coder_buddy.state import AgentState


# --------------------------------------------------------------------------- #
# Pydantic models for structured LLM output
# --------------------------------------------------------------------------- #


class ExplanationOutput(BaseModel):
    """Structured LLM output for the plain-language explanation."""

    explanation: str


class ConfidenceOutput(BaseModel):
    """Structured LLM output for the confidence score."""

    confidence_score: int


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def make_post_process_node(llm_client: "LLMClient", config: "AgentConfig"):
    """
    Factory that returns a ``post_process_node`` closure bound to
    *llm_client* and *config*.

    Args:
        llm_client: Configured ``LLMClient`` instance.
        config:     ``AgentConfig`` instance; ``config.explanation_enabled``
                    controls whether the explanation LLM call is made.

    Returns:
        A ``post_process_node(state) -> dict`` function suitable for use as
        a LangGraph node.
    """

    def post_process_node(state: "AgentState") -> dict:
        """
        Generate explanation and confidence score for the final code.

        Steps:
        1. If ``config.explanation_enabled`` is ``True``, call the LLM for
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
        retry_count: int = state["retry_count"]
        current_code: str = state["current_code"]
        file_name: str = state["file_name"]
        execution_logs: str = state.get("execution_logs", "")  # type: ignore[call-overload]
        token_usage = state["token_usage"]

        log_node_event(
            node="post_process_node",
            event="start",
            retry_count=retry_count,
        )

        # --- Optional explanation ---
        explanation: str | None = None
        if config.explanation_enabled:
            explanation_prompt = _build_explanation_prompt(
                source_code=current_code,
                file_name=file_name,
                retry_count=retry_count,
            )
            explanation_output, explanation_token_record = llm_client.generate(
                explanation_prompt, ExplanationOutput
            )
            explanation = explanation_output.explanation
            token_usage = token_usage.model_copy(
                update={"explanation": explanation_token_record}
            )

        # --- Always: confidence score ---
        confidence_prompt = _build_confidence_prompt(
            source_code=current_code,
            retry_count=retry_count,
            execution_logs=execution_logs,
        )
        confidence_output, confidence_token_record = llm_client.generate(
            confidence_prompt, ConfidenceOutput
        )

        # Clamp to [1, 5]
        confidence_score = max(1, min(5, confidence_output.confidence_score))

        token_usage = token_usage.model_copy(
            update={"confidence": confidence_token_record}
        )

        log_node_event(
            node="post_process_node",
            event="end",
            retry_count=retry_count,
            outcome="success",
            extra={"confidence_score": confidence_score},
        )

        result: dict = {
            "explanation": explanation,
            "confidence_score": confidence_score,
            "token_usage": token_usage,
        }

        if confidence_score <= 2:
            result["warning"] = (
                f"Confidence score is low ({confidence_score}/5). "
                "Please review the generated code carefully before use."
            )

        return result

    return post_process_node


# --------------------------------------------------------------------------- #
# Prompt builders
# --------------------------------------------------------------------------- #


def _build_explanation_prompt(
    source_code: str,
    file_name: str,
    retry_count: int,
) -> str:
    """
    Build the LLM prompt requesting a plain-language explanation of the code.

    Asks the LLM to explain:
    - What the code does overall.
    - Key libraries used and why.
    - Notable logic decisions.

    Args:
        source_code: The final generated source code.
        file_name:   The filename for context.
        retry_count: Number of retries taken (included for context).

    Returns:
        The complete prompt string to send to the LLM.
    """
    retry_note = (
        f" (generated after {retry_count} retry cycle(s))" if retry_count > 0 else ""
    )
    return (
        f"You are reviewing the following Python script (`{file_name}`){retry_note}.\n\n"
        "Please provide a concise plain-language explanation that covers:\n"
        "1. What the code does overall — its purpose and main behaviour.\n"
        "2. Key libraries used and why they were chosen.\n"
        "3. Notable logic decisions or non-obvious implementation choices.\n\n"
        "Return your explanation as an ExplanationOutput with a single "
        "`explanation` field containing the full explanation as a string.\n\n"
        f"```python\n{source_code}\n```"
    )


def _build_confidence_prompt(
    source_code: str,
    retry_count: int,
    execution_logs: str,
) -> str:
    """
    Build the LLM prompt requesting a confidence score for the generated code.

    Asks the LLM to rate confidence on a scale of 1–5, considering:
    - Number of retries needed to produce working code.
    - Code complexity and readability.
    - Edge-case handling and robustness.

    Args:
        source_code:    The final generated source code.
        retry_count:    Number of retries taken (higher → lower confidence).
        execution_logs: Combined stdout/stderr from the last sandbox run.

    Returns:
        The complete prompt string to send to the LLM.
    """
    logs_section = ""
    if execution_logs:
        logs_section = (
            "\n\nExecution logs from the final run:\n"
            f"```\n{execution_logs}\n```"
        )

    return (
        "Rate your confidence in the following Python code on a scale of 1 to 5, "
        "where 1 = very low confidence and 5 = very high confidence.\n\n"
        "Consider the following factors:\n"
        f"1. Retries needed: {retry_count} retry cycle(s) were required to produce "
        "working code (more retries suggest lower confidence).\n"
        "2. Code complexity: Is the code readable, well-structured, and maintainable?\n"
        "3. Edge-case handling: Does the code handle edge cases and potential errors "
        "robustly?\n\n"
        "Return your rating as a ConfidenceOutput with a single `confidence_score` "
        "field containing an integer between 1 and 5 (inclusive).\n\n"
        f"```python\n{source_code}\n```"
        f"{logs_section}"
    )
