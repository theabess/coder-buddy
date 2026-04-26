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
        "llm_backend": "gemini-2.5-flash",
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

    def test_reset_preserves_maxlen(self, patched_agent):
        """After reset(), _history.maxlen must still be 10."""
        agent, mock_graph = patched_agent
        mock_graph.invoke.return_value = _make_success_final_state()

        # Populate history then reset
        agent.run("first prompt")
        agent.run("second prompt")
        agent.reset()

        assert agent._history.maxlen == 10

    def test_reset_on_empty_history_preserves_maxlen(self, patched_agent):
        """reset() on an already-empty deque must still preserve maxlen=10."""
        agent, _ = patched_agent
        assert len(agent._history) == 0

        agent.reset()

        assert agent._history.maxlen == 10

    def test_reset_clears_entries_and_preserves_maxlen(self, patched_agent):
        """After reset(), len == 0 AND maxlen == 10 simultaneously."""
        agent, mock_graph = patched_agent
        mock_graph.invoke.return_value = _make_success_final_state()

        for _ in range(5):
            agent.run("some prompt")

        agent.reset()

        assert len(agent._history) == 0
        assert agent._history.maxlen == 10


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
# Test 5: _history deque is bounded to maxlen=10 (FIFO eviction)
# --------------------------------------------------------------------------- #

class TestHistoryDequeMaxlen:
    """Req 18.1 — _history deque is bounded to maxlen=10; oldest entry discarded (FIFO)."""

    def test_history_maxlen_is_10(self, patched_agent):
        """_history deque must have maxlen=10."""
        agent, _ = patched_agent
        assert agent._history.maxlen == 10

    def test_history_bounded_at_10_entries(self, patched_agent):
        """After 11 runs, _history must contain exactly 10 entries."""
        agent, mock_graph = patched_agent

        for i in range(11):
            state = _make_success_final_state()
            state["current_code"] = f"print({i})"
            state["user_prompt"] = f"prompt {i}"
            mock_graph.invoke.return_value = state
            agent.run(f"write python script {i}")

        assert len(agent._history) == 10

    def test_oldest_entry_discarded_on_overflow(self, patched_agent):
        """When 11 entries are appended, the first (oldest) entry is discarded."""
        agent, mock_graph = patched_agent

        # Run 11 times; each run produces a unique source_code we can identify
        for i in range(11):
            state = _make_success_final_state()
            state["current_code"] = f"print({i})"
            mock_graph.invoke.return_value = state
            agent.run(f"write python script {i}")

        # The oldest entry (i=0, source_code="print(0)") must be gone
        history_codes = [entry.source_code for entry in agent._history]
        assert "print(0)" not in history_codes

    def test_most_recent_entries_retained(self, patched_agent):
        """After 11 runs, the 10 most recent entries (indices 1–10) are retained."""
        agent, mock_graph = patched_agent

        for i in range(11):
            state = _make_success_final_state()
            state["current_code"] = f"print({i})"
            mock_graph.invoke.return_value = state
            agent.run(f"write python script {i}")

        history_codes = [entry.source_code for entry in agent._history]
        # Entries for i=1 through i=10 must all be present
        for i in range(1, 11):
            assert f"print({i})" in history_codes, (
                f"Expected 'print({i})' in history but got: {history_codes}"
            )

    def test_fifo_order_preserved(self, patched_agent):
        """Entries in _history must be in insertion order (oldest first, newest last)."""
        agent, mock_graph = patched_agent

        for i in range(5):
            state = _make_success_final_state()
            state["current_code"] = f"print({i})"
            mock_graph.invoke.return_value = state
            agent.run(f"write python script {i}")

        history_codes = [entry.source_code for entry in agent._history]
        assert history_codes == [f"print({i})" for i in range(5)]

    def test_exactly_10_entries_no_eviction(self, patched_agent):
        """Exactly 10 runs must not evict any entry."""
        agent, mock_graph = patched_agent

        for i in range(10):
            state = _make_success_final_state()
            state["current_code"] = f"print({i})"
            mock_graph.invoke.return_value = state
            agent.run(f"write python script {i}")

        assert len(agent._history) == 10
        history_codes = [entry.source_code for entry in agent._history]
        for i in range(10):
            assert f"print({i})" in history_codes


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


# --------------------------------------------------------------------------- #
# Test 18.4: Integration test — two sequential agent.run() calls; second
#            prompt references first output; verify [Reference code] block
#            is injected in the LLM prompt for the second call.
# --------------------------------------------------------------------------- #


class TestSessionMemoryReferenceInjection:
    """
    Integration test for session memory reference injection (Req 18.4).

    Verifies that when two sequential ``agent.run()`` calls are made on the
    same ``CoderBuddy`` instance and the second prompt contains a reference
    keyword (e.g. "the script", "previous"), the ``Write_Node`` injects a
    ``[Reference code]`` block containing the first run's source code into
    the LLM prompt for the second call.

    The LLM client and sandbox are mocked to avoid real I/O.  The real
    LangGraph graph and ``write_node`` are used so the prompt construction
    logic is exercised end-to-end.
    """

    def _make_integration_config(self) -> "AgentConfig":
        """Return a minimal AgentConfig for integration tests."""
        return _make_config(
            explanation_enabled=False,
            test_generation_enabled=False,
            diff_view_enabled=False,
            max_retries=1,
        )

    def _make_mock_sandbox(self):
        """
        Return a mock sandbox that always reports a successful execution.

        ``execute()`` returns an ``ExecutionResult`` with exit_code=0 so
        the evaluator routes to ``refactor_node`` (success path).
        """
        from coder_buddy.sandbox.base import ExecutionResult

        mock_sandbox = MagicMock()
        mock_sandbox.health_check.return_value = None
        mock_sandbox.install_dependencies.return_value = None
        mock_sandbox.execute.return_value = ExecutionResult(
            stdout="hello\n",
            stderr="",
            exit_code=0,
            timed_out=False,
        )
        mock_sandbox.cleanup.return_value = None
        return mock_sandbox

    def _make_mock_llm_client(self, source_code: str = "print('hello')"):
        """
        Return a mock LLMClient whose ``generate()`` returns appropriate
        structured outputs based on the requested output type.

        - ``CodeArtifact`` requests → return a ``CodeArtifact``
        - ``ConfidenceOutput`` requests → return a ``ConfidenceOutput``
        """
        from coder_buddy.models import CodeArtifact, TokenRecord
        from coder_buddy.nodes.post_process import ConfidenceOutput

        artifact = CodeArtifact(
            source_code=source_code,
            file_name="main.py",
            dependencies=[],
            language="python",
        )
        token_record = TokenRecord(input_tokens=100, output_tokens=50)
        confidence_output = ConfidenceOutput(confidence_score=4)

        def _generate(prompt: str, output_type):
            if output_type is CodeArtifact:
                return (artifact, token_record)
            else:
                # ConfidenceOutput or any other structured type
                return (confidence_output, token_record)

        mock_client = MagicMock()
        mock_client.generate.side_effect = _generate
        return mock_client

    def test_reference_block_injected_in_second_run_prompt(self):
        """
        Core integration test: the second run's LLM prompt contains a
        ``[Reference code]`` block with the first run's source code.

        Flow:
        1. First ``agent.run("write a hello world script")`` completes
           successfully → ``_history`` now contains one ``HistoryEntry``
           with ``source_code="print('hello')"``.
        2. Second ``agent.run("make the script faster")`` is called.
           The prompt contains "the script" (a reference keyword).
        3. The ``Write_Node`` builds the LLM prompt with the session
           history injected → ``[Reference code]`` block is present.
        4. We capture the prompt passed to ``llm_client.generate`` on the
           first call of the second run (the ``write_node`` call) and
           assert it contains ``[Reference code — most recent script]``
           and the first run's source code.
        """
        from coder_buddy.agent import CoderBuddy
        from coder_buddy.graph import build_graph

        config = self._make_integration_config()
        mock_sandbox = self._make_mock_sandbox()
        first_run_source_code = "print('hello')"
        mock_llm_client = self._make_mock_llm_client(
            source_code=first_run_source_code
        )

        # Capture all prompts passed to llm_client.generate
        captured_prompts: list[str] = []
        original_side_effect = mock_llm_client.generate.side_effect

        def _capturing_generate(prompt: str, output_type):
            captured_prompts.append(prompt)
            return original_side_effect(prompt, output_type)

        mock_llm_client.generate.side_effect = _capturing_generate

        with (
            patch("coder_buddy.agent._make_sandbox", return_value=mock_sandbox),
            patch("coder_buddy.agent.LLMClient", return_value=mock_llm_client),
            patch(
                "coder_buddy.agent.build_graph",
                side_effect=lambda sandbox, llm, cfg: build_graph(
                    mock_sandbox, mock_llm_client, cfg
                ),
            ),
        ):
            agent = CoderBuddy(config)

            # --- First run ---
            result1 = agent.run("write a hello world script")

            # Verify the first run succeeded and history was populated
            assert result1.success is True, (
                f"First run should succeed, got success={result1.success}, "
                f"failure_reason={result1.failure_reason}"
            )
            assert len(agent._history) == 1, (
                f"Expected 1 history entry after first run, got {len(agent._history)}"
            )
            assert agent._history[0].source_code == first_run_source_code

            # Record how many generate calls happened in the first run
            first_run_call_count = len(captured_prompts)

            # --- Second run ---
            # "the script" is a reference keyword that triggers _has_prior_reference
            result2 = agent.run("make the script faster")

            assert result2.success is True, (
                f"Second run should succeed, got success={result2.success}, "
                f"failure_reason={result2.failure_reason}"
            )

        # The first generate call of the second run is the write_node call.
        # It should be at index `first_run_call_count` in captured_prompts.
        assert len(captured_prompts) > first_run_call_count, (
            "Expected at least one LLM call during the second run"
        )
        second_run_write_node_prompt = captured_prompts[first_run_call_count]

        # Core assertion: the [Reference code] block must be present
        assert "[Reference code" in second_run_write_node_prompt, (
            f"Expected '[Reference code' block in the second run's write_node prompt.\n"
            f"Prompt was:\n{second_run_write_node_prompt}"
        )

        # The reference block must contain the first run's source code
        assert first_run_source_code in second_run_write_node_prompt, (
            f"Expected first run's source code '{first_run_source_code}' in the "
            f"second run's write_node prompt.\n"
            f"Prompt was:\n{second_run_write_node_prompt}"
        )

        # The [End reference code] marker must also be present
        assert "[End reference code]" in second_run_write_node_prompt, (
            f"Expected '[End reference code]' marker in the second run's prompt.\n"
            f"Prompt was:\n{second_run_write_node_prompt}"
        )

    def test_no_reference_block_when_second_prompt_has_no_keyword(self):
        """
        Negative case: when the second prompt does NOT contain a reference
        keyword, the ``[Reference code]`` block is NOT injected even though
        session history is non-empty.
        """
        from coder_buddy.agent import CoderBuddy
        from coder_buddy.graph import build_graph

        config = self._make_integration_config()
        mock_sandbox = self._make_mock_sandbox()
        mock_llm_client = self._make_mock_llm_client(source_code="print('hello')")

        captured_prompts: list[str] = []
        original_side_effect = mock_llm_client.generate.side_effect

        def _capturing_generate(prompt: str, output_type):
            captured_prompts.append(prompt)
            return original_side_effect(prompt, output_type)

        mock_llm_client.generate.side_effect = _capturing_generate

        with (
            patch("coder_buddy.agent._make_sandbox", return_value=mock_sandbox),
            patch("coder_buddy.agent.LLMClient", return_value=mock_llm_client),
            patch(
                "coder_buddy.agent.build_graph",
                side_effect=lambda sandbox, llm, cfg: build_graph(
                    mock_sandbox, mock_llm_client, cfg
                ),
            ),
        ):
            agent = CoderBuddy(config)

            # First run populates history
            agent.run("write a hello world script")
            assert len(agent._history) == 1

            first_run_call_count = len(captured_prompts)

            # Second run with NO reference keyword
            agent.run("write a sorting algorithm")

        # The write_node prompt for the second run should NOT have [Reference code]
        assert len(captured_prompts) > first_run_call_count
        second_run_write_node_prompt = captured_prompts[first_run_call_count]

        assert "[Reference code" not in second_run_write_node_prompt, (
            f"Did NOT expect '[Reference code' block when no reference keyword used.\n"
            f"Prompt was:\n{second_run_write_node_prompt}"
        )

    def test_session_history_populated_after_first_run(self):
        """
        Verify that after the first ``agent.run()`` call, ``_history``
        contains exactly one entry with the correct source code and prompt.
        """
        from coder_buddy.agent import CoderBuddy
        from coder_buddy.graph import build_graph

        config = self._make_integration_config()
        mock_sandbox = self._make_mock_sandbox()
        first_run_source_code = "print('hello world')"
        mock_llm_client = self._make_mock_llm_client(
            source_code=first_run_source_code
        )

        with (
            patch("coder_buddy.agent._make_sandbox", return_value=mock_sandbox),
            patch("coder_buddy.agent.LLMClient", return_value=mock_llm_client),
            patch(
                "coder_buddy.agent.build_graph",
                side_effect=lambda sandbox, llm, cfg: build_graph(
                    mock_sandbox, mock_llm_client, cfg
                ),
            ),
        ):
            agent = CoderBuddy(config)
            first_prompt = "write a hello world script"
            agent.run(first_prompt)

        assert len(agent._history) == 1
        entry = agent._history[0]
        assert entry.source_code == first_run_source_code
        assert entry.prompt == first_prompt

    def test_reference_block_contains_correct_source_code(self):
        """
        Verify the ``[Reference code]`` block contains the exact source code
        from the first run (not some other code).
        """
        from coder_buddy.agent import CoderBuddy
        from coder_buddy.graph import build_graph
        from coder_buddy.models import CodeArtifact, TokenRecord
        from coder_buddy.nodes.post_process import ConfidenceOutput

        config = self._make_integration_config()
        mock_sandbox = self._make_mock_sandbox()

        # Use a distinctive source code for the first run
        first_run_source_code = "def fibonacci(n):\n    return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"
        second_run_source_code = "def fibonacci_fast(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a"

        call_count = [0]
        captured_prompts: list[str] = []
        token_record = TokenRecord(input_tokens=100, output_tokens=50)
        confidence_output = ConfidenceOutput(confidence_score=4)

        def _generate(prompt: str, output_type):
            captured_prompts.append(prompt)
            call_count[0] += 1
            if output_type is CodeArtifact:
                # First run returns first_run_source_code; second run returns second_run_source_code
                # We determine which run we're in by checking if history is populated
                code = first_run_source_code if call_count[0] <= 2 else second_run_source_code
                artifact = CodeArtifact(
                    source_code=code,
                    file_name="main.py",
                    dependencies=[],
                    language="python",
                )
                return (artifact, token_record)
            else:
                return (confidence_output, token_record)

        mock_llm_client = MagicMock()
        mock_llm_client.generate.side_effect = _generate

        with (
            patch("coder_buddy.agent._make_sandbox", return_value=mock_sandbox),
            patch("coder_buddy.agent.LLMClient", return_value=mock_llm_client),
            patch(
                "coder_buddy.agent.build_graph",
                side_effect=lambda sandbox, llm, cfg: build_graph(
                    mock_sandbox, mock_llm_client, cfg
                ),
            ),
        ):
            agent = CoderBuddy(config)

            # First run
            agent.run("write a fibonacci function")
            first_run_call_count = len(captured_prompts)

            # Second run with reference keyword "the script"
            agent.run("make the script faster")

        # The write_node prompt for the second run
        second_run_write_node_prompt = captured_prompts[first_run_call_count]

        # The reference block must contain the first run's source code
        assert first_run_source_code in second_run_write_node_prompt, (
            f"Expected first run's source code in the reference block.\n"
            f"first_run_source_code={first_run_source_code!r}\n"
            f"Prompt was:\n{second_run_write_node_prompt}"
        )

        # Verify the reference block structure
        ref_start = second_run_write_node_prompt.find("[Reference code")
        ref_end = second_run_write_node_prompt.find("[End reference code]")
        assert ref_start != -1 and ref_end != -1, (
            "Reference block markers not found in prompt"
        )
        reference_section = second_run_write_node_prompt[ref_start:ref_end]
        assert first_run_source_code in reference_section, (
            f"First run's source code not found within the reference block section.\n"
            f"Reference section: {reference_section!r}"
        )


# --------------------------------------------------------------------------- #
# Property 18: history is bounded to 10 entries and oldest entries are
#              discarded first after k > 10 sequential agent.run() calls
# --------------------------------------------------------------------------- #

# Feature: coder-buddy, Property 18: after any number k > 10 of sequential
# agent.run() calls, len(CoderBuddy._history) <= 10 and oldest entries are
# discarded first.


@given(
    k=st.integers(min_value=11, max_value=30),
)
@settings(max_examples=100)
def test_property18_history_bounded_and_fifo_after_many_runs(k: int) -> None:
    """
    **Validates: Requirements 10.4**

    Property 18: For any number k > 10 of sequential ``agent.run()`` calls
    on the same ``CoderBuddy`` instance:

    1. ``len(CoderBuddy._history) <= 10`` — the deque never exceeds its
       maximum capacity.
    2. The oldest entries are discarded first (FIFO) — after k runs, the
       history contains the k most recent entries (up to 10), not the
       oldest ones.

    Each run produces a uniquely identifiable ``source_code`` string
    (``f"print({i})"`` for run index ``i``), so we can verify which entries
    were retained and which were evicted.
    """
    from coder_buddy.agent import CoderBuddy

    config = _make_config()

    with (
        patch("coder_buddy.agent._make_sandbox") as mock_make_sandbox,
        patch("coder_buddy.agent.LLMClient"),
        patch("coder_buddy.agent.build_graph") as mock_build_graph,
    ):
        mock_sandbox = MagicMock()
        mock_sandbox.health_check.return_value = None
        mock_make_sandbox.return_value = mock_sandbox

        mock_graph = MagicMock()
        mock_build_graph.return_value = mock_graph

        agent = CoderBuddy(config)

        # Run k times; each run produces a unique, identifiable source_code
        for i in range(k):
            state = _make_success_final_state()
            state["current_code"] = f"print({i})"
            state["user_prompt"] = f"prompt {i}"
            mock_graph.invoke.return_value = state
            agent.run(f"write python script {i}")

    # --- Property assertion 1: history length never exceeds 10 ---
    assert len(agent._history) <= 10, (
        f"Expected len(_history) <= 10 after {k} runs, "
        f"got {len(agent._history)}"
    )

    # --- Property assertion 2: exactly 10 entries retained (since k > 10) ---
    assert len(agent._history) == 10, (
        f"Expected exactly 10 entries after {k} > 10 runs, "
        f"got {len(agent._history)}"
    )

    # --- Property assertion 3: oldest entries are discarded (FIFO) ---
    # After k runs (k > 10), the history must contain the LAST 10 runs
    # (indices k-10 through k-1), not the first ones (indices 0 through k-11).
    history_codes = [entry.source_code for entry in agent._history]

    # The 10 most recent entries must all be present
    for i in range(k - 10, k):
        assert f"print({i})" in history_codes, (
            f"Expected recent entry 'print({i})' in history after {k} runs, "
            f"but it was missing. History codes: {history_codes}"
        )

    # The oldest entries (indices 0 through k-11) must all be evicted
    for i in range(k - 10):
        assert f"print({i})" not in history_codes, (
            f"Expected old entry 'print({i})' to be evicted after {k} runs, "
            f"but it was still present. History codes: {history_codes}"
        )

    # --- Property assertion 4: FIFO order preserved within retained entries ---
    # The retained entries must appear in insertion order (oldest first)
    expected_order = [f"print({i})" for i in range(k - 10, k)]
    assert history_codes == expected_order, (
        f"Expected history in insertion order {expected_order}, "
        f"got {history_codes}"
    )


# --------------------------------------------------------------------------- #
# Test 20.3: AgentResponse.token_usage reflects accumulated totals from all
#            nodes in the run
# --------------------------------------------------------------------------- #


class TestTokenUsageAccumulation:
    """
    Req 20.3 — AgentResponse.token_usage reflects the accumulated totals
    from all nodes in the run.

    These tests verify that when the final AgentState carries a TokenUsage
    with per-node records populated, the AgentResponse returned by
    CoderBuddy.run() exposes those exact accumulated totals via
    token_usage.total_input_tokens and token_usage.total_output_tokens.
    """

    def test_token_usage_reflects_write_node_tokens(self, patched_agent):
        """token_usage must reflect write_node tokens from the final state."""
        from coder_buddy.models import TokenRecord, TokenUsage

        agent, mock_graph = patched_agent
        state = _make_success_final_state()
        state["token_usage"] = TokenUsage(
            write_node=TokenRecord(input_tokens=100, output_tokens=50),
        )
        mock_graph.invoke.return_value = state

        result = agent.run("write hello world")

        assert result.token_usage.write_node.input_tokens == 100
        assert result.token_usage.write_node.output_tokens == 50
        assert result.token_usage.total_input_tokens == 100
        assert result.token_usage.total_output_tokens == 50

    def test_token_usage_reflects_all_node_tokens(self, patched_agent):
        """token_usage must reflect accumulated totals from all nodes."""
        from coder_buddy.models import TokenRecord, TokenUsage

        agent, mock_graph = patched_agent
        state = _make_success_final_state()
        state["token_usage"] = TokenUsage(
            write_node=TokenRecord(input_tokens=100, output_tokens=50),
            refactor_node=TokenRecord(input_tokens=80, output_tokens=40),
            explanation=TokenRecord(input_tokens=60, output_tokens=30),
            test_node=TokenRecord(input_tokens=40, output_tokens=20),
            confidence=TokenRecord(input_tokens=20, output_tokens=10),
        )
        mock_graph.invoke.return_value = state

        result = agent.run("write hello world")

        assert result.token_usage.total_input_tokens == 300   # 100+80+60+40+20
        assert result.token_usage.total_output_tokens == 150  # 50+40+30+20+10

    def test_token_usage_per_node_values_preserved(self, patched_agent):
        """Each per-node TokenRecord must be preserved exactly in the response."""
        from coder_buddy.models import TokenRecord, TokenUsage

        agent, mock_graph = patched_agent
        state = _make_success_final_state()
        state["token_usage"] = TokenUsage(
            write_node=TokenRecord(input_tokens=200, output_tokens=100),
            refactor_node=TokenRecord(input_tokens=150, output_tokens=75),
            confidence=TokenRecord(input_tokens=30, output_tokens=15),
        )
        mock_graph.invoke.return_value = state

        result = agent.run("write hello world")

        assert result.token_usage.write_node.input_tokens == 200
        assert result.token_usage.write_node.output_tokens == 100
        assert result.token_usage.refactor_node.input_tokens == 150
        assert result.token_usage.refactor_node.output_tokens == 75
        assert result.token_usage.confidence.input_tokens == 30
        assert result.token_usage.confidence.output_tokens == 15

    def test_token_usage_is_zero_when_state_has_default_token_usage(self, patched_agent):
        """When the final state has a default TokenUsage, totals must be 0."""
        from coder_buddy.models import TokenUsage

        agent, mock_graph = patched_agent
        state = _make_success_final_state()
        state["token_usage"] = TokenUsage()  # all zeros
        mock_graph.invoke.return_value = state

        result = agent.run("write hello world")

        assert result.token_usage.total_input_tokens == 0
        assert result.token_usage.total_output_tokens == 0

    def test_token_usage_type_is_token_usage(self, patched_agent):
        """AgentResponse.token_usage must be a TokenUsage instance."""
        from coder_buddy.models import TokenRecord, TokenUsage

        agent, mock_graph = patched_agent
        state = _make_success_final_state()
        state["token_usage"] = TokenUsage(
            write_node=TokenRecord(input_tokens=100, output_tokens=50),
        )
        mock_graph.invoke.return_value = state

        result = agent.run("write hello world")

        assert isinstance(result.token_usage, TokenUsage)

    def test_token_usage_accumulated_totals_match_sum_of_nodes(self, patched_agent):
        """
        End-to-end: total_input_tokens and total_output_tokens must equal
        the arithmetic sum of all per-node input/output tokens.
        """
        from coder_buddy.models import TokenRecord, TokenUsage

        agent, mock_graph = patched_agent

        write_in, write_out = 120, 60
        refactor_in, refactor_out = 90, 45
        explanation_in, explanation_out = 70, 35
        test_in, test_out = 50, 25
        confidence_in, confidence_out = 30, 15

        state = _make_success_final_state()
        state["token_usage"] = TokenUsage(
            write_node=TokenRecord(input_tokens=write_in, output_tokens=write_out),
            refactor_node=TokenRecord(input_tokens=refactor_in, output_tokens=refactor_out),
            explanation=TokenRecord(input_tokens=explanation_in, output_tokens=explanation_out),
            test_node=TokenRecord(input_tokens=test_in, output_tokens=test_out),
            confidence=TokenRecord(input_tokens=confidence_in, output_tokens=confidence_out),
        )
        mock_graph.invoke.return_value = state

        result = agent.run("write hello world")

        expected_total_input = write_in + refactor_in + explanation_in + test_in + confidence_in
        expected_total_output = write_out + refactor_out + explanation_out + test_out + confidence_out

        assert result.token_usage.total_input_tokens == expected_total_input, (
            f"Expected total_input_tokens={expected_total_input}, "
            f"got {result.token_usage.total_input_tokens}"
        )
        assert result.token_usage.total_output_tokens == expected_total_output, (
            f"Expected total_output_tokens={expected_total_output}, "
            f"got {result.token_usage.total_output_tokens}"
        )

    def test_token_usage_on_failure_run_reflects_state(self, patched_agent):
        """
        Even on a failed run (success=False), token_usage must reflect
        the accumulated totals from the final state.
        """
        from coder_buddy.models import TokenRecord, TokenUsage

        agent, mock_graph = patched_agent
        state = _make_failure_final_state()
        state["token_usage"] = TokenUsage(
            write_node=TokenRecord(input_tokens=300, output_tokens=150),
        )
        mock_graph.invoke.return_value = state

        result = agent.run("write broken code")

        assert result.success is False
        assert result.token_usage.write_node.input_tokens == 300
        assert result.token_usage.write_node.output_tokens == 150
        assert result.token_usage.total_input_tokens == 300
        assert result.token_usage.total_output_tokens == 150


class TestTokenUsageEndToEndWithRealGraph:
    """
    End-to-end test that exercises the real LangGraph graph with mocked
    LLM and sandbox, verifying that each node's TokenRecord is accumulated
    into AgentResponse.token_usage.

    This test uses the real graph (not a mocked graph.invoke) so the actual
    node token accumulation logic is exercised.
    """

    def _make_integration_config(self, **overrides) -> "AgentConfig":
        defaults = {
            "llm_backend": "gemini-2.5-flash",
            "sandbox_backend": "subprocess+venv",
            "max_retries": 1,
            "explanation_enabled": False,
            "test_generation_enabled": False,
            "diff_view_enabled": False,
        }
        defaults.update(overrides)
        return AgentConfig(**defaults)

    def _make_mock_sandbox(self):
        from coder_buddy.sandbox.base import ExecutionResult

        mock_sandbox = MagicMock()
        mock_sandbox.health_check.return_value = None
        mock_sandbox.install_dependencies.return_value = None
        mock_sandbox.execute.return_value = ExecutionResult(
            stdout="hello\n",
            stderr="",
            exit_code=0,
            timed_out=False,
        )
        mock_sandbox.cleanup.return_value = None
        return mock_sandbox

    def test_write_node_tokens_accumulated_in_response(self):
        """
        With the real graph, write_node's TokenRecord must appear in
        AgentResponse.token_usage after a successful run.
        """
        from coder_buddy.agent import CoderBuddy
        from coder_buddy.graph import build_graph
        from coder_buddy.models import CodeArtifact, TokenRecord
        from coder_buddy.nodes.post_process import ConfidenceOutput

        config = self._make_integration_config()
        mock_sandbox = self._make_mock_sandbox()

        write_token_record = TokenRecord(input_tokens=100, output_tokens=50)
        confidence_token_record = TokenRecord(input_tokens=20, output_tokens=10)
        refactor_token_record = TokenRecord(input_tokens=80, output_tokens=40)

        artifact = CodeArtifact(
            source_code="print('hello')",
            file_name="main.py",
            dependencies=[],
            language="python",
        )
        confidence_output = ConfidenceOutput(confidence_score=4)

        call_count = [0]

        def _generate(prompt: str, output_type):
            call_count[0] += 1
            if output_type is CodeArtifact:
                return (artifact, write_token_record)
            elif output_type is ConfidenceOutput:
                return (confidence_output, confidence_token_record)
            else:
                return (confidence_output, confidence_token_record)

        mock_llm_client = MagicMock()
        mock_llm_client.generate.side_effect = _generate

        with (
            patch("coder_buddy.agent._make_sandbox", return_value=mock_sandbox),
            patch("coder_buddy.agent.LLMClient", return_value=mock_llm_client),
            patch(
                "coder_buddy.agent.build_graph",
                side_effect=lambda sandbox, llm, cfg: build_graph(
                    mock_sandbox, mock_llm_client, cfg
                ),
            ),
        ):
            agent = CoderBuddy(config)
            result = agent.run("write a hello world script")

        assert result.success is True, (
            f"Expected success=True, got failure_reason={result.failure_reason}"
        )
        assert isinstance(result.token_usage, TokenUsage)

        # write_node tokens must be accumulated
        assert result.token_usage.write_node.input_tokens == 100, (
            f"Expected write_node.input_tokens=100, "
            f"got {result.token_usage.write_node.input_tokens}"
        )
        assert result.token_usage.write_node.output_tokens == 50, (
            f"Expected write_node.output_tokens=50, "
            f"got {result.token_usage.write_node.output_tokens}"
        )

        # confidence tokens must be accumulated (post_process always runs on success)
        assert result.token_usage.confidence.input_tokens == 20, (
            f"Expected confidence.input_tokens=20, "
            f"got {result.token_usage.confidence.input_tokens}"
        )

        # total must reflect the sum
        expected_total_input = (
            result.token_usage.write_node.input_tokens
            + result.token_usage.refactor_node.input_tokens
            + result.token_usage.explanation.input_tokens
            + result.token_usage.test_node.input_tokens
            + result.token_usage.confidence.input_tokens
        )
        assert result.token_usage.total_input_tokens == expected_total_input, (
            f"total_input_tokens={result.token_usage.total_input_tokens} "
            f"does not match sum of per-node records={expected_total_input}"
        )
