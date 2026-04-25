"""
Unit tests for CoderBuddy agent (agent.py).

Covers:
- Successful run returns AgentResponse with success=True
- Max-retries exhaustion returns AgentResponse with success=False and all
  required failure fields populated
- reset() clears session history
- Unsupported language returns error AgentResponse without entering the cycle

Property tests:
- Property 9: for any final AgentState where retry_count == max_retries,
  the built AgentResponse has success=False and non-None source_code,
  execution_logs, retry_count.

All LLM and sandbox calls are mocked to avoid real network/process I/O.
"""

from __future__ import annotations

import collections
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from coder_buddy.config import AgentConfig
from coder_buddy.models import AgentResponse, TokenUsage


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _make_config(**overrides) -> AgentConfig:
    """Return a minimal AgentConfig suitable for unit tests."""
    defaults = {
        "llm_backend": "gemini-1.5-pro",
        "sandbox_backend": "subprocess+venv",
        "max_retries": 3,
        "explanation_enabled": False,
        "test_generation_enabled": False,
        "diff_view_enabled": False,
    }
    defaults.update(overrides)
    return AgentConfig(**defaults)


def _make_success_final_state(max_retries: int = 3) -> dict:
    """Return a fake final_state dict representing a successful run."""
    return {
        "current_code": "print('hello')",
        "file_name": "main.py",
        "dependencies": [],
        "execution_logs": "hello\n",
        "error_status": False,
        "retry_count": 1,
        "explanation": None,
        "test_code": None,
        "test_logs": None,
        "confidence_score": 4,
        "refactor_diff": "",
        "token_usage": TokenUsage(),
        "warning": None,
        "max_retries": max_retries,
        "pre_refactor_code": None,
        "session_history": [],
        "user_prompt": "write hello world",
        "language": "python",
    }


def _make_failure_final_state(max_retries: int = 3) -> dict:
    """Return a fake final_state dict representing a max-retries exhaustion."""
    return {
        "current_code": "print(undefined_var)",
        "file_name": "main.py",
        "dependencies": [],
        "execution_logs": "NameError: name 'undefined_var' is not defined\n",
        "error_status": True,
        "retry_count": max_retries,
        "explanation": None,
        "test_code": None,
        "test_logs": None,
        "confidence_score": None,
        "refactor_diff": None,
        "token_usage": TokenUsage(),
        "warning": None,
        "max_retries": max_retries,
        "pre_refactor_code": None,
        "session_history": [],
        "user_prompt": "write broken code",
        "language": "python",
    }


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture()
def patched_agent():
    """
    Yield a CoderBuddy instance with sandbox health_check and graph.invoke
    both mocked so no real I/O occurs.

    Returns a tuple of (agent, mock_graph_invoke).
    """
    from coder_buddy.agent import CoderBuddy

    config = _make_config()

    with (
        patch("coder_buddy.agent._make_sandbox") as mock_make_sandbox,
        patch("coder_buddy.agent.LLMClient") as mock_llm_cls,
        patch("coder_buddy.agent.build_graph") as mock_build_graph,
    ):
        # Sandbox mock — health_check does nothing
        mock_sandbox = MagicMock()
        mock_sandbox.health_check.return_value = None
        mock_make_sandbox.return_value = mock_sandbox

        # LLM client mock
        mock_llm_cls.return_value = MagicMock()

        # Graph mock — we'll set return_value per test
        mock_graph = MagicMock()
        mock_build_graph.return_value = mock_graph

        agent = CoderBuddy(config)
        yield agent, mock_graph


# --------------------------------------------------------------------------- #
# Test 1: Successful run returns AgentResponse with success=True
# --------------------------------------------------------------------------- #

class TestSuccessfulRun:
    """Req 4.6 — retry_count recorded; success=True on clean run."""

    def test_returns_agent_response_type(self, patched_agent):
        """run() must return an AgentResponse instance."""
        agent, mock_graph = patched_agent
        mock_graph.invoke.return_value = _make_success_final_state()

        result = agent.run("write hello world")

        assert isinstance(result, AgentResponse)

    def test_success_is_true(self, patched_agent):
        """success field must be True when error_status is False."""
        agent, mock_graph = patched_agent
        mock_graph.invoke.return_value = _make_success_final_state()

        result = agent.run("write hello world")

        assert result.success is True

    def test_source_code_non_empty(self, patched_agent):
        """source_code must be non-empty on a successful run."""
        agent, mock_graph = patched_agent
        mock_graph.invoke.return_value = _make_success_final_state()

        result = agent.run("write hello world")

        assert result.source_code != ""

    def test_required_fields_populated(self, patched_agent):
        """All required AgentResponse fields must be present and typed correctly."""
        agent, mock_graph = patched_agent
        mock_graph.invoke.return_value = _make_success_final_state()

        result = agent.run("write hello world")

        assert isinstance(result.file_name, str)
        assert isinstance(result.dependencies, list)
        assert isinstance(result.execution_logs, str)
        assert isinstance(result.retry_count, int)
        assert isinstance(result.token_usage, TokenUsage)
        assert isinstance(result.elapsed_seconds, float)

    def test_failure_reason_is_none_on_success(self, patched_agent):
        """failure_reason must be None when the run succeeds."""
        agent, mock_graph = patched_agent
        mock_graph.invoke.return_value = _make_success_final_state()

        result = agent.run("write hello world")

        assert result.failure_reason is None

    def test_graph_invoke_called_once(self, patched_agent):
        """graph.invoke must be called exactly once per run()."""
        agent, mock_graph = patched_agent
        mock_graph.invoke.return_value = _make_success_final_state()

        agent.run("write hello world")

        mock_graph.invoke.assert_called_once()


# --------------------------------------------------------------------------- #
# Test 2: Max-retries exhaustion returns AgentResponse with success=False
# --------------------------------------------------------------------------- #

class TestMaxRetriesExhaustion:
    """Req 4.4, 4.5, 4.6 — structured failure report, no exception."""

    def test_success_is_false(self, patched_agent):
        """success must be False when error_status is True at end of run."""
        agent, mock_graph = patched_agent
        mock_graph.invoke.return_value = _make_failure_final_state(max_retries=3)

        result = agent.run("write broken code")

        assert result.success is False

    def test_source_code_non_none(self, patched_agent):
        """source_code must be present (non-None) even on failure."""
        agent, mock_graph = patched_agent
        mock_graph.invoke.return_value = _make_failure_final_state(max_retries=3)

        result = agent.run("write broken code")

        assert result.source_code is not None

    def test_execution_logs_non_none(self, patched_agent):
        """execution_logs must be present (non-None) on failure."""
        agent, mock_graph = patched_agent
        mock_graph.invoke.return_value = _make_failure_final_state(max_retries=3)

        result = agent.run("write broken code")

        assert result.execution_logs is not None

    def test_retry_count_equals_max_retries(self, patched_agent):
        """retry_count must equal max_retries when exhausted."""
        agent, mock_graph = patched_agent
        max_retries = 3
        mock_graph.invoke.return_value = _make_failure_final_state(max_retries=max_retries)

        result = agent.run("write broken code")

        assert result.retry_count == max_retries

    def test_failure_reason_is_populated(self, patched_agent):
        """failure_reason must be a non-empty string on failure."""
        agent, mock_graph = patched_agent
        mock_graph.invoke.return_value = _make_failure_final_state(max_retries=3)

        result = agent.run("write broken code")

        assert result.failure_reason is not None
        assert len(result.failure_reason) > 0

    def test_no_exception_raised(self, patched_agent):
        """run() must NOT raise an exception on max-retries exhaustion (Req 4.5)."""
        agent, mock_graph = patched_agent
        mock_graph.invoke.return_value = _make_failure_final_state(max_retries=3)

        # Should not raise
        result = agent.run("write broken code")
        assert isinstance(result, AgentResponse)

    def test_retry_count_recorded_in_response(self, patched_agent):
        """retry_count must be recorded in every response (Req 4.6)."""
        agent, mock_graph = patched_agent
        mock_graph.invoke.return_value = _make_failure_final_state(max_retries=3)

        result = agent.run("write broken code")

        assert isinstance(result.retry_count, int)
        assert result.retry_count >= 0


# --------------------------------------------------------------------------- #
# Test 3: reset() clears session history
# --------------------------------------------------------------------------- #

class TestResetClearsHistory:
    """Req 10.5 — reset() initialises Session_History as an empty list."""

    def test_reset_clears_history_after_run(self, patched_agent):
        """After run() + reset(), _history must be empty."""
        agent, mock_graph = patched_agent
        mock_graph.invoke.return_value = _make_success_final_state()

        agent.run("write hello world")
        assert len(agent._history) > 0, "history should be non-empty after a run"

        agent.reset()

        assert len(agent._history) == 0

    def test_reset_on_empty_history_is_safe(self, patched_agent):
        """reset() on an already-empty history must not raise."""
        agent, _ = patched_agent
        assert len(agent._history) == 0

        agent.reset()  # should not raise

        assert len(agent._history) == 0

    def test_reset_clears_multiple_runs(self, patched_agent):
        """reset() must clear history accumulated over multiple runs."""
        agent, mock_graph = patched_agent
        mock_graph.invoke.return_value = _make_success_final_state()

        agent.run("first prompt")
        agent.run("second prompt")
        assert len(agent._history) == 2

        agent.reset()

        assert len(agent._history) == 0

    def test_history_is_deque(self, patched_agent):
        """_history must be a collections.deque instance."""
        agent, _ = patched_agent
        assert isinstance(agent._history, collections.deque)

    def test_run_after_reset_repopulates_history(self, patched_agent):
        """After reset(), a new run() should add to history again."""
        agent, mock_graph = patched_agent
        mock_graph.invoke.return_value = _make_success_final_state()

        agent.run("first prompt")
        agent.reset()
        assert len(agent._history) == 0

        agent.run("second prompt")
        assert len(agent._history) == 1


# --------------------------------------------------------------------------- #
# Test 4: Unsupported language returns error without entering cycle
# --------------------------------------------------------------------------- #

class TestUnsupportedLanguageEarlyExit:
    """Req 8.3 — unsupported language returns error without invoking the graph."""

    @pytest.mark.parametrize("prompt", [
        "write a JavaScript function that adds two numbers",
        "create a TypeScript class for a user model",
        "write a Ruby script to parse CSV files",
        "write a Golang HTTP server",
        "create a Kotlin data class",
        "write a PHP script to connect to MySQL",
    ])
    def test_unsupported_language_returns_failure(self, patched_agent, prompt):
        """Prompts requesting non-Python languages must return success=False."""
        agent, mock_graph = patched_agent

        result = agent.run(prompt)

        assert result.success is False

    @pytest.mark.parametrize("prompt", [
        "write a JavaScript function that adds two numbers",
        "create a TypeScript class for a user model",
        "write a Ruby script to parse CSV files",
    ])
    def test_graph_not_invoked_for_unsupported_language(self, patched_agent, prompt):
        """graph.invoke must NOT be called when the language is unsupported."""
        agent, mock_graph = patched_agent

        agent.run(prompt)

        mock_graph.invoke.assert_not_called()

    def test_failure_reason_mentions_language(self, patched_agent):
        """failure_reason must mention the unsupported language."""
        agent, mock_graph = patched_agent

        result = agent.run("write a JavaScript function")

        assert result.failure_reason is not None
        assert "JavaScript" in result.failure_reason or "not supported" in result.failure_reason

    def test_unsupported_language_returns_agent_response(self, patched_agent):
        """The return value must be an AgentResponse (not an exception)."""
        agent, mock_graph = patched_agent

        result = agent.run("write a Ruby script")

        assert isinstance(result, AgentResponse)

    def test_python_prompt_does_not_trigger_early_exit(self, patched_agent):
        """A Python prompt must proceed to graph.invoke normally."""
        agent, mock_graph = patched_agent
        mock_graph.invoke.return_value = _make_success_final_state()

        result = agent.run("write a Python script to print hello world")

        mock_graph.invoke.assert_called_once()
        assert result.success is True

    def test_unsupported_language_retry_count_is_zero(self, patched_agent):
        """retry_count must be 0 for an early-exit (no cycles entered)."""
        agent, mock_graph = patched_agent

        result = agent.run("write a TypeScript interface")

        assert result.retry_count == 0

    def test_unsupported_language_does_not_add_to_history(self, patched_agent):
        """An early-exit run must not add an entry to session history."""
        agent, mock_graph = patched_agent

        agent.run("write a JavaScript function")

        assert len(agent._history) == 0


# --------------------------------------------------------------------------- #
# Property 9: max-retries exhaustion always yields a well-formed failure response
# --------------------------------------------------------------------------- #

# Feature: coder-buddy, Property 9: for any final AgentState where
# retry_count == max_retries, the built AgentResponse has success=False and
# non-None source_code, execution_logs, retry_count.


@given(
    max_retries=st.integers(min_value=1, max_value=10),
    source_code=st.text(min_size=1),
    execution_logs=st.text(min_size=1),
    dependencies=st.lists(st.text()),
    file_name=st.text(min_size=1),
)
@settings(max_examples=200)
def test_property9_max_retries_exhaustion_yields_failure_response(
    max_retries: int,
    source_code: str,
    execution_logs: str,
    dependencies: list[str],
    file_name: str,
) -> None:
    """
    **Validates: Requirements 4.4, 4.5, 4.6**

    Property 9: For any final AgentState where retry_count == max_retries,
    the AgentResponse built by CoderBuddy.run() SHALL have:
    - success == False
    - source_code is not None
    - execution_logs is not None
    - retry_count is not None (and equals max_retries)
    """
    from coder_buddy.agent import CoderBuddy

    config = _make_config(max_retries=max_retries)

    # Construct a final state that simulates max-retries exhaustion:
    # retry_count == max_retries and error_status == True.
    exhausted_final_state = {
        "current_code": source_code,
        "file_name": file_name,
        "dependencies": dependencies,
        "execution_logs": execution_logs,
        "error_status": True,          # still failing — evaluator routed to END
        "retry_count": max_retries,    # exactly at the limit
        "explanation": None,
        "test_code": None,
        "test_logs": None,
        "confidence_score": None,
        "refactor_diff": None,
        "token_usage": TokenUsage(),
        "warning": None,
        "max_retries": max_retries,
        "pre_refactor_code": None,
        "session_history": [],
        "user_prompt": "write some code",
        "language": "python",
    }

    with (
        patch("coder_buddy.agent._make_sandbox") as mock_make_sandbox,
        patch("coder_buddy.agent.LLMClient"),
        patch("coder_buddy.agent.build_graph") as mock_build_graph,
    ):
        mock_sandbox = MagicMock()
        mock_sandbox.health_check.return_value = None
        mock_make_sandbox.return_value = mock_sandbox

        mock_graph = MagicMock()
        mock_graph.invoke.return_value = exhausted_final_state
        mock_build_graph.return_value = mock_graph

        agent = CoderBuddy(config)
        result = agent.run("write some code")

    # Core property assertions
    assert result.success is False, (
        f"Expected success=False when retry_count={max_retries} == max_retries={max_retries}, "
        f"got success={result.success}"
    )
    assert result.source_code is not None, (
        "source_code must not be None in a max-retries failure response"
    )
    assert result.execution_logs is not None, (
        "execution_logs must not be None in a max-retries failure response"
    )
    assert result.retry_count is not None, (
        "retry_count must not be None in a max-retries failure response"
    )
    assert result.retry_count == max_retries, (
        f"Expected retry_count={max_retries}, got {result.retry_count}"
    )


# --------------------------------------------------------------------------- #
# Property 10: AgentResponse.retry_count equals retry_count in the final
#              AgentState for any run outcome
# --------------------------------------------------------------------------- #

# Feature: coder-buddy, Property 10: AgentResponse.retry_count equals
# retry_count in the final AgentState for any run outcome


@given(
    retry_count=st.integers(min_value=0, max_value=10),
    success=st.booleans(),
    source_code=st.text(min_size=1),
    execution_logs=st.text(),
    dependencies=st.lists(st.text()),
    file_name=st.text(min_size=1),
)
@settings(max_examples=100)
def test_property10_retry_count_equals_final_state_retry_count(
    retry_count: int,
    success: bool,
    source_code: str,
    execution_logs: str,
    dependencies: list[str],
    file_name: str,
) -> None:
    """
    **Validates: Requirements 4.6**

    Property 10: For any run outcome (success or failure), the
    ``AgentResponse.retry_count`` returned by ``CoderBuddy.run()`` SHALL
    equal the ``retry_count`` field in the final ``AgentState`` that was
    used to build the response.
    """
    from coder_buddy.agent import CoderBuddy

    # max_retries must be >= retry_count to keep AgentConfig valid (1–10).
    # We clamp max_retries to [retry_count, 10], defaulting to 10 when
    # retry_count is 0 so the config validator is always satisfied.
    max_retries = max(retry_count, 1)

    config = _make_config(max_retries=max_retries)

    # Build a final state with the generated retry_count and the chosen
    # success/failure outcome.
    final_state = {
        "current_code": source_code,
        "file_name": file_name,
        "dependencies": dependencies,
        "execution_logs": execution_logs,
        "error_status": not success,   # error_status is the inverse of success
        "retry_count": retry_count,    # the value we want to verify is preserved
        "explanation": None,
        "test_code": None,
        "test_logs": None,
        "confidence_score": None,
        "refactor_diff": None,
        "token_usage": TokenUsage(),
        "warning": None,
        "max_retries": max_retries,
        "pre_refactor_code": None,
        "session_history": [],
        "user_prompt": "write some code",
        "language": "python",
    }

    with (
        patch("coder_buddy.agent._make_sandbox") as mock_make_sandbox,
        patch("coder_buddy.agent.LLMClient"),
        patch("coder_buddy.agent.build_graph") as mock_build_graph,
    ):
        mock_sandbox = MagicMock()
        mock_sandbox.health_check.return_value = None
        mock_make_sandbox.return_value = mock_sandbox

        mock_graph = MagicMock()
        mock_graph.invoke.return_value = final_state
        mock_build_graph.return_value = mock_graph

        agent = CoderBuddy(config)
        response = agent.run("write some code")

    # Core property assertion: response.retry_count must mirror the state
    assert response.retry_count == final_state["retry_count"], (
        f"Expected AgentResponse.retry_count={final_state['retry_count']} "
        f"(from final AgentState), got {response.retry_count}. "
        f"success={success}, retry_count={retry_count}"
    )


# --------------------------------------------------------------------------- #
# Property 13: initial AgentState always has retry_count == 0 and
#              error_status == False
# --------------------------------------------------------------------------- #

# Feature: coder-buddy, Property 13: the AgentState constructed at the start
# of any agent.run() call has retry_count == 0 and error_status == False


@given(
    prompt=st.text(min_size=1).filter(
        lambda p: not any(
            kw in p.lower()
            for kw in [
                "javascript", "typescript", "java ", "in java", "ruby",
                "golang", "in go ", "rust ", "in rust", "c++ ", "in c++",
                "c# ", "in c#", "php ", "in php", "swift ", "in swift",
                "kotlin", "scala ", "in scala", "haskell", "r script", "in r ",
            ]
        )
    ),
)
@settings(max_examples=100)
def test_property13_initial_agent_state_has_zero_retry_count_and_no_error(
    prompt: str,
) -> None:
    """
    **Validates: Requirements 6.3**

    Property 13: For any valid prompt passed to ``CoderBuddy.run()``, the
    ``AgentState`` constructed at the start of the call SHALL have:
    - ``retry_count == 0``
    - ``error_status == False``

    This is verified by intercepting the ``graph.invoke()`` call and
    inspecting the initial state dict passed to it.
    """
    from coder_buddy.agent import CoderBuddy

    config = _make_config()

    captured_states: list[dict] = []

    def _capture_and_return(state: dict) -> dict:
        """Side-effectful fake that records the initial state then returns a
        minimal valid final state so agent.run() can complete normally."""
        captured_states.append(dict(state))
        return _make_success_final_state()

    with (
        patch("coder_buddy.agent._make_sandbox") as mock_make_sandbox,
        patch("coder_buddy.agent.LLMClient"),
        patch("coder_buddy.agent.build_graph") as mock_build_graph,
    ):
        mock_sandbox = MagicMock()
        mock_sandbox.health_check.return_value = None
        mock_make_sandbox.return_value = mock_sandbox

        mock_graph = MagicMock()
        mock_graph.invoke.side_effect = _capture_and_return
        mock_build_graph.return_value = mock_graph

        agent = CoderBuddy(config)
        agent.run(prompt)

    # graph.invoke must have been called exactly once
    assert len(captured_states) == 1, (
        f"Expected graph.invoke to be called once, got {len(captured_states)} calls"
    )

    initial_state = captured_states[0]

    # Core property assertions
    assert initial_state["retry_count"] == 0, (
        f"Expected initial_state['retry_count'] == 0, "
        f"got {initial_state['retry_count']!r} for prompt={prompt!r}"
    )
    assert initial_state["error_status"] is False, (
        f"Expected initial_state['error_status'] == False, "
        f"got {initial_state['error_status']!r} for prompt={prompt!r}"
    )
