"""
CoderBuddy Agent — public API and session management.

This module exposes the top-level ``CoderBuddy`` class that owns the
session history, constructs the LangGraph state graph, and provides the
``run`` / ``reset`` public interface.
"""

from __future__ import annotations

import collections
import time
from typing import TYPE_CHECKING

from coder_buddy.config import AgentConfig
from coder_buddy.models import AgentResponse, HistoryEntry, TokenUsage

if TYPE_CHECKING:
    pass


class CoderBuddy:
    """
    Top-level agent class.

    Owns the session history (bounded deque), constructs the sandbox
    backend and LLM client, builds the LangGraph state graph, and
    exposes a simple ``run`` / ``reset`` API.
    """

    def __init__(self, config: AgentConfig) -> None:
        """
        Validate *config*, instantiate the sandbox backend, build the
        LangGraph ``StateGraph``, and initialise session history.

        Raises:
            ConfigurationError: If *config* contains invalid values.
            SandboxUnavailableError: If the sandbox backend fails its
                health check.
        """
        ...

    def run(self, prompt: str) -> AgentResponse:
        """
        Run a single *prompt* through the full agent cycle.

        Constructs the initial ``AgentState`` (with ``retry_count=0``,
        ``error_status=False``, and the last N history entries), invokes
        the compiled graph, appends the result to session history, emits
        the final run-summary JSON log, and returns an ``AgentResponse``.

        Returns:
            AgentResponse regardless of success or failure.
        """
        ...

    def reset(self) -> None:
        """Clear session history."""
        ...
