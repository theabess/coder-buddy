"""
LLM client factory for Coder Buddy.

``LLMClient`` is a thin wrapper around Pydantic AI's ``Agent`` class.
It handles model selection, structured output validation, per-call token
usage extraction, and retry logic for parse failures.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from coder_buddy.config import LLMUnavailableError, ParseError
from coder_buddy.llm.pricing import estimate_cost
from coder_buddy.models import TokenRecord

if TYPE_CHECKING:
    from pydantic import BaseModel

# pydantic-ai is a required dependency
try:
    from pydantic_ai import Agent  # type: ignore[import]
    from pydantic_ai.exceptions import (  # type: ignore[import]
        ModelAPIError,
        ModelHTTPError,
        UnexpectedModelBehavior,
    )
    from pydantic_ai.usage import RunUsage  # type: ignore[import]
except ImportError:
    Agent = None  # type: ignore[assignment]
    ModelAPIError = None  # type: ignore[assignment]
    ModelHTTPError = None  # type: ignore[assignment]
    UnexpectedModelBehavior = None  # type: ignore[assignment]
    RunUsage = None  # type: ignore[assignment]


# Map of model names to environment variable names for API keys
_MODEL_ENV_VARS: dict[str, str] = {
    "gemini-2.5-flash": "GEMINI_API_KEY",
    "gemini-2.5-pro": "GEMINI_API_KEY",
    "gpt-4o": "OPENAI_API_KEY",
    "claude-3-5-sonnet": "ANTHROPIC_API_KEY",
}

# Map of model names to pydantic-ai model strings
_MODEL_STRINGS: dict[str, str] = {
    "gemini-2.5-flash": "google-gla:gemini-2.5-flash",
    "gemini-2.5-pro": "google-gla:gemini-2.5-pro",
    "gpt-4o": "openai:gpt-4o",
    "claude-3-5-sonnet": "anthropic:claude-3-5-sonnet-latest",
}


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
        self._model = model
        self._api_key = api_key

        # Set the API key in the environment if provided, so pydantic-ai picks it up
        if api_key is not None:
            env_var = _MODEL_ENV_VARS.get(model)
            if env_var is not None:
                os.environ[env_var] = api_key

        # Resolve the pydantic-ai model string (fall back to the raw model name)
        self._model_string = _MODEL_STRINGS.get(model, model)

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
        if Agent is None:
            raise LLMUnavailableError(
                "pydantic-ai is not installed. Install it with: pip install pydantic-ai"
            )

        # Create an Agent with the output type and retry settings.
        # ``retries`` in pydantic-ai controls how many times it retries on
        # validation/parse failures before raising UnexpectedModelBehavior.
        agent: Agent = Agent(
            model=self._model_string,
            output_type=output_type,
            retries=max_retries,
        )

        try:
            result = agent.run_sync(prompt)
        except ModelHTTPError as exc:
            raise LLMUnavailableError(
                f"HTTP error from LLM provider (status {exc.status_code}): {exc}"
            ) from exc
        except ModelAPIError as exc:
            raise LLMUnavailableError(
                f"LLM API error: {exc}"
            ) from exc
        except UnexpectedModelBehavior as exc:
            # pydantic-ai raises this after exhausting retries on parse failures
            raise ParseError(
                f"LLM failed to produce a valid {output_type.__name__} "
                f"after {max_retries} retries: {exc}"
            ) from exc
        except Exception as exc:
            # Catch network-level errors (ConnectionError, TimeoutError, etc.)
            exc_type = type(exc).__name__
            if any(
                keyword in exc_type.lower()
                for keyword in ("connection", "timeout", "network", "http", "ssl", "auth")
            ):
                raise LLMUnavailableError(
                    f"Network/connection error communicating with LLM: {exc}"
                ) from exc
            raise

        # Extract token usage from the result
        usage: RunUsage = result.usage()
        input_tokens: int = usage.input_tokens or 0
        output_tokens: int = usage.output_tokens or 0

        # Compute estimated cost using the pricing table
        cost = estimate_cost(
            model=self._model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        token_record = TokenRecord(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=cost,
        )

        return result.output, token_record
