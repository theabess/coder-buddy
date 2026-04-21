"""
LLM client factory for Coder Buddy.

``LLMClient`` is a thin wrapper around Pydantic AI's ``Agent`` class.
It handles model selection, structured output validation, per-call token
usage extraction, and retry logic for parse failures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, type_check_only

from coder_buddy.models import TokenRecord

if TYPE_CHECKING:
    from pydantic import BaseModel

# pydantic-ai is a required dependency
try:
    from pydantic_ai import Agent  # type: ignore[import]
except ImportError:
    Agent = None  # type: ignore[assignment]


class LLMClient:
    """
    Thin wrapper around Pydantic AI's ``Agent`` class.

    Handles:
    - Model selection (Gemini, GPT-4o, Claude)
    - Structured output validation via Pydantic models
    - Per-call token usage extraction
    - Automatic retry on parse failure (delegated to Pydantic AI)

    Raises:
        LLMUnavailableError: On HTTP errors or authentication failures.
        ParseError: After exhausting all parse retries.
    """

    def __init__(self, model: str, api_key: str | None = None) -> None:
        """
        Initialise the LLM client for the given *model*.

        Args:
            model:   Model identifier, e.g. ``"gemini-1.5-pro"``.
            api_key: Optional API key; falls back to the relevant
                     environment variable if ``None``.
        """
        ...

    def generate(
        self,
        prompt: str,
        output_type: type["BaseModel"],
        max_retries: int = 3,
    ) -> tuple["BaseModel", TokenRecord]:
        """
        Call the LLM and validate the response against *output_type*.

        Args:
            prompt:      The full prompt string to send to the LLM.
            output_type: Pydantic model class to validate the response against.
            max_retries: Maximum number of parse retries before raising
                         ``ParseError``.

        Returns:
            A tuple of ``(validated_model_instance, token_record)``.

        Raises:
            LLMUnavailableError: On HTTP errors or authentication failures.
            ParseError: After *max_retries* parse failures.
        """
        ...
