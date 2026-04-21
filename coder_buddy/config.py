"""
Agent configuration and exception hierarchy for Coder Buddy.

``AgentConfig`` is a dataclass that holds all tuneable parameters.
Its ``__post_init__`` validator raises ``ValueError`` for out-of-range
or unsupported values so that misconfiguration is caught at construction
time rather than at runtime.

The ``CoderBuddyError`` hierarchy provides typed exceptions for every
failure mode the agent can encounter.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# Exception hierarchy
# --------------------------------------------------------------------------- #


class CoderBuddyError(Exception):
    """Base exception for all Coder Buddy errors."""


class ConfigurationError(CoderBuddyError):
    """Raised when ``AgentConfig`` contains invalid values."""


class SandboxUnavailableError(CoderBuddyError):
    """Raised when the sandbox backend is not reachable at startup."""


class LLMUnavailableError(CoderBuddyError):
    """Raised on LLM HTTP errors or authentication failures."""


class ParseError(CoderBuddyError):
    """Raised after exhausting all LLM parse retries."""


class LanguageNotSupportedError(CoderBuddyError):
    """Raised when the requested language is not ``"python"`` in V1."""


# --------------------------------------------------------------------------- #
# Valid backend names
# --------------------------------------------------------------------------- #

VALID_SANDBOX_BACKENDS: frozenset[str] = frozenset(
    {"subprocess+venv", "docker", "e2b", "pyodide"}
)

VALID_LLM_BACKENDS: frozenset[str] = frozenset(
    {"gemini-1.5-pro", "gpt-4o", "claude-3-5-sonnet"}
)


# --------------------------------------------------------------------------- #
# AgentConfig
# --------------------------------------------------------------------------- #


@dataclass
class AgentConfig:
    """
    All tuneable parameters for a ``CoderBuddy`` instance.

    Validated in ``__post_init__``; raises ``ValueError`` for:
    - ``max_retries`` outside ``[1, 10]``
    - ``sandbox_backend`` not in the supported set
    - ``llm_backend`` not in the supported set
    """

    # LLM
    llm_backend: str = "gemini-1.5-pro"
    """LLM model name.  One of ``"gemini-1.5-pro"``, ``"gpt-4o"``,
    ``"claude-3-5-sonnet"``."""

    llm_api_key: str | None = None
    """Optional API key; falls back to the relevant environment variable."""

    # Sandbox
    sandbox_backend: str = "subprocess+venv"
    """Sandbox backend.  One of ``"subprocess+venv"``, ``"docker"``,
    ``"e2b"``, ``"pyodide"``."""

    sandbox_timeout_seconds: float = 10.0
    """Maximum wall-clock seconds allowed for a single sandbox execution."""

    # Retry
    max_retries: int = 5
    """Maximum number of write-execute-check cycles.  Must be in ``[1, 10]``."""

    # Session memory
    session_history_context_n: int = 5
    """Number of history entries to include in the LLM prompt."""

    session_history_max: int = 10
    """Hard cap on the number of entries stored in the session history deque."""

    # Feature flags
    explanation_enabled: bool = True
    """When ``True``, the post-process node generates a plain-language
    explanation of the final code."""

    test_generation_enabled: bool = True
    """When ``True``, the test node generates and runs a pytest suite."""

    diff_view_enabled: bool = True
    """When ``True``, a unified diff is computed and stored in the response."""

    # Pricing (optional, for cost estimation)
    price_per_input_token: float | None = None
    """Override the built-in price table for input tokens."""

    price_per_output_token: float | None = None
    """Override the built-in price table for output tokens."""

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not (1 <= self.max_retries <= 10):
            raise ValueError(
                f"max_retries must be between 1 and 10 inclusive, got {self.max_retries}"
            )
        if self.sandbox_backend not in VALID_SANDBOX_BACKENDS:
            raise ValueError(
                f"sandbox_backend '{self.sandbox_backend}' is not supported. "
                f"Choose one of: {sorted(VALID_SANDBOX_BACKENDS)}"
            )
        if self.llm_backend not in VALID_LLM_BACKENDS:
            raise ValueError(
                f"llm_backend '{self.llm_backend}' is not supported. "
                f"Choose one of: {sorted(VALID_LLM_BACKENDS)}"
            )
