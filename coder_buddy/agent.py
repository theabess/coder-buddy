"""
CoderBuddy Agent — public API and session management.

This module exposes the top-level ``CoderBuddy`` class that owns the
session history, constructs the LangGraph state graph, and provides the
``run`` / ``reset`` public interface.
"""

from __future__ import annotations

import collections
import json
import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from coder_buddy.config import AgentConfig, ConfigurationError
from coder_buddy.graph import build_graph
from coder_buddy.llm.client import LLMClient
from coder_buddy.models import AgentResponse, HistoryEntry, TokenUsage
from coder_buddy.sandbox.docker_backend import DockerBackend
from coder_buddy.sandbox.e2b_backend import E2BBackend
from coder_buddy.sandbox.pyodide_backend import PyodideBackend
from coder_buddy.sandbox.subprocess_venv import SubprocessVenvBackend

if TYPE_CHECKING:
    pass

_logger = logging.getLogger("coder_buddy")


def _make_sandbox(sandbox_backend: str):
    """
    Instantiate the correct sandbox backend for the given *sandbox_backend* name.

    Args:
        sandbox_backend: One of ``"subprocess+venv"``, ``"docker"``,
            ``"e2b"``, ``"pyodide"``.

    Returns:
        A concrete ``SandboxBackend`` instance.

    Raises:
        ConfigurationError: If *sandbox_backend* is not a recognised value.
    """
    if sandbox_backend == "subprocess+venv":
        return SubprocessVenvBackend()
    elif sandbox_backend == "docker":
        return DockerBackend()
    elif sandbox_backend == "e2b":
        return E2BBackend()
    elif sandbox_backend == "pyodide":
        return PyodideBackend()
    else:
        raise ConfigurationError(
            f"Unknown sandbox_backend '{sandbox_backend}'. "
            "Must be one of: 'subprocess+venv', 'docker', 'e2b', 'pyodide'."
        )


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
        self._config = config

        # Instantiate the sandbox backend based on config
        sandbox = _make_sandbox(config.sandbox_backend)

        # Verify the sandbox is available — SandboxUnavailableError propagates
        sandbox.health_check()

        self._sandbox = sandbox

        # Instantiate the LLM client
        self._llm_client = LLMClient(
            model=config.llm_backend,
            api_key=config.llm_api_key,
        )

        # Build the compiled LangGraph state graph
        self._graph = build_graph(self._sandbox, self._llm_client, config)

        # Initialise bounded session history (max 10 entries, FIFO)
        self._history: collections.deque = collections.deque(maxlen=10)

    def run(self, prompt: str) -> AgentResponse:
        """
        Run a single *prompt* through the full agent cycle.

        Constructs the initial ``AgentState`` (with ``retry_count=0``,
        ``error_status=False``, and the last N history entries), invokes
        the compiled graph, appends the result to session history, emits
        the final run-summary JSON log, and returns an ``AgentResponse``.

        If the prompt explicitly requests a non-Python language (detected
        via a simple keyword scan), returns an error ``AgentResponse``
        immediately without invoking the graph (Req 8.3).

        Returns:
            AgentResponse regardless of success or failure.
        """
        start_time = time.monotonic()

        # Early-exit for unsupported languages (Req 8.3).
        # Detect common non-Python language keywords in the prompt so we can
        # return an error before entering the graph cycle.
        _UNSUPPORTED_LANGUAGE_KEYWORDS: list[tuple[str, str]] = [
            ("javascript", "JavaScript"),
            ("typescript", "TypeScript"),
            ("java ", "Java"),
            ("in java", "Java"),
            ("ruby", "Ruby"),
            ("golang", "Go"),
            ("in go ", "Go"),
            ("rust ", "Rust"),
            ("in rust", "Rust"),
            ("c++ ", "C++"),
            ("in c++", "C++"),
            ("c# ", "C#"),
            ("in c#", "C#"),
            ("php ", "PHP"),
            ("in php", "PHP"),
            ("swift ", "Swift"),
            ("in swift", "Swift"),
            ("kotlin", "Kotlin"),
            ("scala ", "Scala"),
            ("in scala", "Scala"),
            ("haskell", "Haskell"),
            ("r script", "R"),
            ("in r ", "R"),
        ]
        prompt_lower = prompt.lower()
        for keyword, lang_name in _UNSUPPORTED_LANGUAGE_KEYWORDS:
            if keyword in prompt_lower:
                elapsed = time.monotonic() - start_time
                return AgentResponse(
                    success=False,
                    source_code="",
                    file_name="",
                    dependencies=[],
                    execution_logs="",
                    retry_count=0,
                    explanation=None,
                    test_code=None,
                    confidence_score=None,
                    refactor_diff=None,
                    token_usage=TokenUsage(),
                    elapsed_seconds=round(elapsed, 3),
                    failure_reason=(
                        f"Language '{lang_name}' is not supported in V1; "
                        "only 'python' is supported."
                    ),
                )

        # Slice the last N history entries to inject into the initial state
        history_slice = list(self._history)[-self._config.session_history_context_n :]

        # Construct the initial AgentState
        initial_state = {
            "user_prompt": prompt,
            "current_code": "",
            "execution_logs": "",
            "error_status": False,
            "retry_count": 0,
            "dependencies": [],
            "file_name": "main.py",
            "language": "python",
            "explanation": None,
            "test_code": None,
            "test_logs": None,
            "confidence_score": None,
            "refactor_diff": None,
            "token_usage": TokenUsage(),
            "session_history": history_slice,
            "max_retries": self._config.max_retries,
            "pre_refactor_code": None,
            "warning": None,
        }

        # Invoke the compiled LangGraph state graph
        final_state = self._graph.invoke(initial_state)

        elapsed = time.monotonic() - start_time

        # Success means the final state has no execution errors.
        # When max_retries is exhausted the evaluator routes to END with
        # error_status still True, so this single check covers both cases.
        success = not final_state.get("error_status", False)

        # Append to session history (bounded deque handles FIFO eviction)
        if final_state.get("current_code"):
            self._history.append(
                HistoryEntry(
                    prompt=prompt,
                    source_code=final_state.get("current_code", ""),
                    file_name=final_state.get("file_name", "main.py"),
                    dependencies=final_state.get("dependencies", []),
                    timestamp=datetime.now(tz=timezone.utc),
                )
            )

        token_usage: TokenUsage = final_state.get("token_usage", TokenUsage())

        # Emit the final run-summary JSON log entry (Req 9.3 / 9.4)
        log_entry: dict = {
            "ts": time.time(),
            "event": "run_complete",
            "success": success,
            "retry_count": final_state.get("retry_count", 0),
            "elapsed_seconds": round(elapsed, 3),
            "confidence_score": final_state.get("confidence_score"),
            "token_usage": {
                "write_node": {
                    "input": token_usage.write_node.input_tokens,
                    "output": token_usage.write_node.output_tokens,
                },
                "refactor_node": {
                    "input": token_usage.refactor_node.input_tokens,
                    "output": token_usage.refactor_node.output_tokens,
                },
                "total_input": token_usage.total_input_tokens,
                "total_output": token_usage.total_output_tokens,
                "estimated_cost_usd": token_usage.total_estimated_cost_usd,
            },
            "refactor_diff": final_state.get("refactor_diff"),
        }
        _logger.info(json.dumps(log_entry))

        # Build and return the AgentResponse
        failure_reason: str | None = None
        if not success:
            failure_reason = (
                f"Agent exhausted {final_state.get('retry_count', 0)} retries "
                f"without producing working code. "
                f"Last error: {final_state.get('execution_logs', '')[:500]}"
            )

        return AgentResponse(
            success=success,
            source_code=final_state.get("current_code", ""),
            file_name=final_state.get("file_name", "main.py"),
            dependencies=final_state.get("dependencies", []),
            execution_logs=final_state.get("execution_logs", ""),
            retry_count=final_state.get("retry_count", 0),
            explanation=final_state.get("explanation"),
            test_code=final_state.get("test_code"),
            confidence_score=final_state.get("confidence_score"),
            refactor_diff=final_state.get("refactor_diff"),
            token_usage=token_usage,
            elapsed_seconds=round(elapsed, 3),
            failure_reason=failure_reason,
            warning=final_state.get("warning"),
        )

    def reset(self) -> None:
        """Clear session history."""
        self._history.clear()
