"""
Integration tests for CoderBuddy (Task 19).

These tests exercise the full agent pipeline end-to-end using the real
``subprocess+venv`` sandbox backend and the real LangGraph state graph.
The LLM client is mocked so no real API calls are made, but every other
component — sandbox creation, venv setup, script execution, graph routing,
node execution, and response assembly — runs against the real implementation.

Test 19.1: Full agent run with subprocess+venv backend against a "hello world"
           prompt — verify success=True, source_code non-empty, retry_count >= 0.

Performance note
----------------
Creating a Python venv takes several seconds.  To keep the suite fast,
session-scoped fixtures build each agent variant once and cache the result.
All tests within a class share the same pre-computed AgentResponse.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from coder_buddy.config import AgentConfig
from coder_buddy.models import AgentResponse, CodeArtifact, TokenRecord, TokenUsage
from coder_buddy.nodes.post_process import ConfidenceOutput, ExplanationOutput


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


def _make_token_record(input_tokens: int = 100, output_tokens: int = 50) -> TokenRecord:
    return TokenRecord(input_tokens=input_tokens, output_tokens=output_tokens)


def _make_hello_world_artifact() -> CodeArtifact:
    """A CodeArtifact that always executes cleanly."""
    return CodeArtifact(
        source_code='print("Hello, World!")\n',
        file_name="main.py",
        dependencies=[],
        language="python",
    )


def _make_mock_llm_client(
    artifact: CodeArtifact | None = None,
    confidence_score: int = 4,
) -> MagicMock:
    """
    Mock LLMClient whose ``generate()`` dispatches on output_type:
    - CodeArtifact       → *artifact* (defaults to hello-world)
    - ConfidenceOutput   → ConfidenceOutput(confidence_score=*confidence_score*)
    - ExplanationOutput  → ExplanationOutput with a canned explanation
    """
    if artifact is None:
        artifact = _make_hello_world_artifact()

    token_record = _make_token_record()
    confidence_output = ConfidenceOutput(confidence_score=confidence_score)
    explanation_output = ExplanationOutput(
        explanation="This script prints 'Hello, World!' to standard output."
    )

    def _generate(prompt: str, output_type: type) -> tuple:
        if output_type is CodeArtifact:
            return (artifact, token_record)
        elif output_type is ConfidenceOutput:
            return (confidence_output, token_record)
        elif output_type is ExplanationOutput:
            return (explanation_output, token_record)
        else:
            return (confidence_output, token_record)

    mock_client = MagicMock()
    mock_client.generate.side_effect = _generate
    return mock_client


def _make_integration_config(**overrides) -> AgentConfig:
    """AgentConfig for integration tests (real subprocess+venv, LLM mocked)."""
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


def _build_agent(artifact: CodeArtifact | None = None, **config_overrides):
    """
    Build a real CoderBuddy with a real subprocess+venv sandbox and a
    mocked LLM client.  Returns ``(agent, mock_llm_client)``.
    """
    from coder_buddy.agent import CoderBuddy

    config = _make_integration_config(**config_overrides)
    mock_llm = _make_mock_llm_client(artifact=artifact)

    with patch("coder_buddy.agent.LLMClient", return_value=mock_llm):
        agent = CoderBuddy(config)

    return agent, mock_llm


def _build_agent_with_mock_llm(mock_llm: MagicMock, **config_overrides):
    """
    Build a real CoderBuddy with a real subprocess+venv sandbox and a
    caller-supplied mock LLM client.  Returns the agent.
    """
    from coder_buddy.agent import CoderBuddy

    config = _make_integration_config(**config_overrides)
    with patch("coder_buddy.agent.LLMClient", return_value=mock_llm):
        agent = CoderBuddy(config)
    return agent


# --------------------------------------------------------------------------- #
# Session-scoped fixtures — each agent variant is built once per test session
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def hello_world_artifact():
    return _make_hello_world_artifact()


@pytest.fixture(scope="session")
def shared_agent_and_results(hello_world_artifact):
    """
    Agent A (default config, explanation disabled).
    Runs two sequential prompts and caches both results.
    """
    agent, _ = _build_agent(artifact=hello_world_artifact)
    result1 = agent.run("write a hello world script in Python")
    result2 = agent.run("write another hello world script")
    return agent, result1, result2


@pytest.fixture(scope="session")
def shared_result(shared_agent_and_results):
    _, result1, _ = shared_agent_and_results
    return result1


@pytest.fixture(scope="session")
def shared_result2(shared_agent_and_results):
    _, _, result2 = shared_agent_and_results
    return result2


@pytest.fixture(scope="session")
def shared_agent(shared_agent_and_results):
    agent, _, _ = shared_agent_and_results
    return agent


@pytest.fixture(scope="session")
def shared_result_explanation(hello_world_artifact):
    """Agent B: explanation enabled."""
    agent, _ = _build_agent(
        artifact=hello_world_artifact,
        explanation_enabled=True,
    )
    return agent.run("write a hello world script in Python")


@pytest.fixture(scope="session")
def retry_result():
    """Agent C: one-retry run."""
    token_record = _make_token_record()
    broken_artifact = CodeArtifact(
        source_code="def broken(\n    x = 1\n",  # SyntaxError: unexpected EOF
        file_name="main.py",
        dependencies=[],
        language="python",
    )
    good_artifact = _make_hello_world_artifact()
    confidence_output = ConfidenceOutput(confidence_score=4)
    explanation_output = ExplanationOutput(
        explanation="This script prints 'Hello, World!' to standard output."
    )

    code_artifact_calls = [broken_artifact, good_artifact]
    call_index = {"n": 0}

    def _generate(prompt: str, output_type: type) -> tuple:
        if output_type is CodeArtifact:
            artifact = code_artifact_calls[min(call_index["n"], len(code_artifact_calls) - 1)]
            call_index["n"] += 1
            return (artifact, token_record)
        elif output_type is ConfidenceOutput:
            return (confidence_output, token_record)
        elif output_type is ExplanationOutput:
            return (explanation_output, token_record)
        else:
            return (confidence_output, token_record)

    mock_client = MagicMock()
    mock_client.generate.side_effect = _generate

    agent = _build_agent_with_mock_llm(mock_client)
    return agent.run("write a hello world script in Python")


@pytest.fixture(scope="session")
def max_retries_result():
    """Agent D: always-broken LLM, max_retries=2."""
    token_record = _make_token_record()
    broken_artifact = CodeArtifact(
        source_code="def broken(\n    x = 1\n",  # SyntaxError: unexpected EOF
        file_name="main.py",
        dependencies=[],
        language="python",
    )
    confidence_output = ConfidenceOutput(confidence_score=4)
    explanation_output = ExplanationOutput(
        explanation="This script prints 'Hello, World!' to standard output."
    )

    def _generate(prompt: str, output_type: type) -> tuple:
        if output_type is CodeArtifact:
            return (broken_artifact, token_record)
        elif output_type is ConfidenceOutput:
            return (confidence_output, token_record)
        elif output_type is ExplanationOutput:
            return (explanation_output, token_record)
        else:
            return (confidence_output, token_record)

    mock_client = MagicMock()
    mock_client.generate.side_effect = _generate

    agent = _build_agent_with_mock_llm(
        mock_client,
        max_retries=2,
        explanation_enabled=False,
        test_generation_enabled=False,
        diff_view_enabled=False,
    )
    return agent.run("write a hello world script in Python")


@pytest.fixture(scope="session")
def explanation_and_test_result():
    """Agent E: explanation + test generation enabled.

    Patches venv creation to use --system-site-packages so pytest (already
    installed in the project venv) is available in the fresh sandbox venv
    without a slow pip-install step.
    """
    import subprocess as _subprocess

    _original_run = _subprocess.run

    def _patched_subprocess_run(args, **kwargs):
        """Inject --system-site-packages into every venv creation call."""
        if (
            isinstance(args, list)
            and len(args) >= 3
            and "-m" in args
            and "venv" in args
            and "--system-site-packages" not in args
        ):
            args = list(args) + ["--system-site-packages"]
        return _original_run(args, **kwargs)

    token_record = _make_token_record()
    hello_world_artifact = _make_hello_world_artifact()

    refactored_artifact = CodeArtifact(
        source_code='# Print a greeting to standard output.\nprint("Hello, World!")\n',
        file_name="main.py",
        dependencies=[],
        language="python",
    )

    test_suite_code = (
        "def test_always_passes():\n"
        "    assert True\n"
        "\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    import pytest\n"
        "    pytest.main([__file__, '-v'])\n"
    )
    test_artifact = CodeArtifact(
        source_code=test_suite_code,
        file_name="test_main.py",
        dependencies=[],
        language="python",
    )

    confidence_output = ConfidenceOutput(confidence_score=4)
    explanation_output = ExplanationOutput(
        explanation="This script prints 'Hello, World!' to standard output."
    )

    code_artifact_calls = [hello_world_artifact, refactored_artifact, test_artifact]
    call_index = {"n": 0}

    def _generate(prompt: str, output_type: type) -> tuple:
        if output_type is CodeArtifact:
            idx = min(call_index["n"], len(code_artifact_calls) - 1)
            artifact = code_artifact_calls[idx]
            call_index["n"] += 1
            return (artifact, token_record)
        elif output_type is ConfidenceOutput:
            return (confidence_output, token_record)
        elif output_type is ExplanationOutput:
            return (explanation_output, token_record)
        else:
            return (confidence_output, token_record)

    mock_client = MagicMock()
    mock_client.generate.side_effect = _generate

    with patch(
        "coder_buddy.sandbox.subprocess_venv.subprocess.run",
        side_effect=_patched_subprocess_run,
    ):
        agent = _build_agent_with_mock_llm(
            mock_client,
            explanation_enabled=True,
            test_generation_enabled=True,
            diff_view_enabled=False,
        )
        result = agent.run("write a hello world script in Python")

    return result


# --------------------------------------------------------------------------- #
# Task 19.1 — Full agent run with subprocess+venv backend, "hello world" prompt
# --------------------------------------------------------------------------- #


class TestFullAgentRunHelloWorld:
    """
    Task 19.1: Full end-to-end agent run using the real subprocess+venv sandbox.

    The LLM is mocked to return ``print("Hello, World!")``.  All other
    components run against the real implementation.

    Verifies:
    - ``success=True``
    - ``source_code`` is non-empty
    - ``retry_count >= 0``
    """

    def test_success_is_true(self, shared_result):
        """success must be True for a clean hello-world run."""
        assert shared_result.success is True, (
            f"Expected success=True, got success={shared_result.success}. "
            f"failure_reason={shared_result.failure_reason!r}, "
            f"execution_logs={shared_result.execution_logs!r}"
        )

    def test_source_code_is_non_empty(self, shared_result):
        """source_code must be a non-empty string."""
        assert shared_result.source_code, (
            f"Expected non-empty source_code, got {shared_result.source_code!r}"
        )
        assert isinstance(shared_result.source_code, str)

    def test_retry_count_is_non_negative(self, shared_result):
        """retry_count must be an integer >= 0."""
        assert isinstance(shared_result.retry_count, int)
        assert shared_result.retry_count >= 0, (
            f"Expected retry_count >= 0, got {shared_result.retry_count}"
        )

    def test_returns_agent_response_type(self, shared_result):
        assert isinstance(shared_result, AgentResponse)

    def test_failure_reason_is_none_on_success(self, shared_result):
        assert shared_result.failure_reason is None, (
            f"Expected failure_reason=None, got {shared_result.failure_reason!r}"
        )

    def test_execution_logs_is_string(self, shared_result):
        assert isinstance(shared_result.execution_logs, str)

    def test_hello_world_output_in_execution_logs(self, shared_result):
        """Real sandbox stdout must appear in execution_logs."""
        assert "Hello, World!" in shared_result.execution_logs, (
            f"Expected 'Hello, World!' in execution_logs, "
            f"got {shared_result.execution_logs!r}"
        )

    def test_file_name_is_non_empty_string(self, shared_result):
        assert isinstance(shared_result.file_name, str)
        assert shared_result.file_name

    def test_dependencies_is_list(self, shared_result):
        assert isinstance(shared_result.dependencies, list)

    def test_token_usage_write_node_non_zero(self, shared_result):
        """write_node must have recorded token usage from the mock LLM."""
        assert isinstance(shared_result.token_usage, TokenUsage)
        assert shared_result.token_usage.write_node.input_tokens > 0

    def test_elapsed_seconds_is_positive(self, shared_result):
        assert isinstance(shared_result.elapsed_seconds, float)
        assert shared_result.elapsed_seconds > 0

    def test_source_code_matches_artifact(self, shared_result, hello_world_artifact):
        """source_code must equal the artifact returned by the mock LLM."""
        assert shared_result.source_code == hello_world_artifact.source_code

    def test_retry_count_is_zero_on_clean_run(self, shared_result):
        """No retries needed for a script that executes cleanly first time."""
        assert shared_result.retry_count == 0, (
            f"Expected retry_count=0, got {shared_result.retry_count}"
        )

    def test_explanation_is_none_when_disabled(self, shared_result):
        """explanation_enabled=False (default config) → explanation is None."""
        assert shared_result.explanation is None

    def test_test_code_is_none_when_disabled(self, shared_result):
        """test_generation_enabled=False (default config) → test_code is None."""
        assert shared_result.test_code is None

    def test_confidence_score_in_valid_range(self, shared_result):
        """confidence_score must be an integer in [1, 5]."""
        assert shared_result.confidence_score is not None
        assert isinstance(shared_result.confidence_score, int)
        assert 1 <= shared_result.confidence_score <= 5

    def test_real_sandbox_executed_code(self, shared_result):
        """
        Presence of real stdout confirms the real sandbox ran (not a mock).
        """
        assert "Hello, World!" in shared_result.execution_logs, (
            f"Expected real sandbox output in execution_logs. "
            f"Got: {shared_result.execution_logs!r}"
        )

    def test_session_history_populated_after_run(self, shared_agent, hello_world_artifact):
        """_history must contain the run's source code after a successful run."""
        assert len(shared_agent._history) >= 1
        assert shared_agent._history[0].source_code == hello_world_artifact.source_code

    def test_two_sequential_runs_both_succeed(self, shared_result, shared_result2, shared_agent):
        """
        Two sequential runs on the same agent must both return success=True.
        """
        assert shared_result.success is True, (
            f"First run failed: {shared_result.failure_reason!r}"
        )
        assert shared_result2.success is True, (
            f"Second run failed: {shared_result2.failure_reason!r}"
        )
        assert len(shared_agent._history) == 2

    def test_explanation_populated_when_enabled(self, shared_result_explanation):
        """explanation_enabled=True → explanation is a non-empty string."""
        assert shared_result_explanation.success is True, (
            f"Run failed: {shared_result_explanation.failure_reason!r}"
        )
        assert shared_result_explanation.explanation is not None
        assert isinstance(shared_result_explanation.explanation, str)
        assert shared_result_explanation.explanation.strip()


# --------------------------------------------------------------------------- #
# Task 19.2 — Full agent run that requires exactly one retry
# --------------------------------------------------------------------------- #


class TestFullAgentRunOneRetry:
    """
    Task 19.2: Full end-to-end agent run that requires exactly one retry.

    The mock LLM returns a syntax-error script on the first CodeArtifact
    call and valid hello-world code on the second.  All other components
    (sandbox, graph, nodes) run against the real implementation.

    Verifies:
    - ``retry_count == 1``
    - ``success=True``
    """

    def test_success_is_true(self, retry_result):
        """success must be True after the retry produces working code."""
        assert retry_result.success is True, (
            f"Expected success=True, got success={retry_result.success}. "
            f"failure_reason={retry_result.failure_reason!r}, "
            f"execution_logs={retry_result.execution_logs!r}"
        )

    def test_retry_count_equals_one(self, retry_result):
        """Primary assertion: exactly one retry was needed."""
        assert retry_result.retry_count == 1, (
            f"Expected retry_count=1, got retry_count={retry_result.retry_count}. "
            f"execution_logs={retry_result.execution_logs!r}"
        )

    def test_source_code_is_non_empty(self, retry_result):
        """source_code must be a non-empty string."""
        assert retry_result.source_code
        assert isinstance(retry_result.source_code, str)

    def test_source_code_is_good_artifact(self, retry_result):
        """source_code must equal the second (good) artifact returned by the mock LLM."""
        assert retry_result.source_code == _make_hello_world_artifact().source_code

    def test_execution_logs_contain_hello_world(self, retry_result):
        """Real sandbox must have executed the good code and produced stdout."""
        assert "Hello, World!" in retry_result.execution_logs, (
            f"Expected 'Hello, World!' in execution_logs, "
            f"got {retry_result.execution_logs!r}"
        )

    def test_failure_reason_is_none(self, retry_result):
        """failure_reason must be None on a successful run."""
        assert retry_result.failure_reason is None, (
            f"Expected failure_reason=None, got {retry_result.failure_reason!r}"
        )

    def test_returns_agent_response_type(self, retry_result):
        """Result must be an AgentResponse instance."""
        assert isinstance(retry_result, AgentResponse)

    def test_token_usage_write_node_non_zero(self, retry_result):
        """write_node must have recorded token usage (called at least twice)."""
        assert retry_result.token_usage.write_node.input_tokens > 0

    def test_elapsed_seconds_is_positive(self, retry_result):
        """elapsed_seconds must be a positive float."""
        assert retry_result.elapsed_seconds > 0

    def test_confidence_score_in_valid_range(self, retry_result):
        """confidence_score must be an integer in [1, 5]."""
        assert retry_result.confidence_score is not None
        assert 1 <= retry_result.confidence_score <= 5


# --------------------------------------------------------------------------- #
# Task 19.3 — Max-retries exhaustion: LLM always returns broken code
# --------------------------------------------------------------------------- #

_MAX_RETRIES_VALUE = 2  # must match the max_retries passed to max_retries_result fixture


class TestMaxRetriesExhaustion:
    """
    Task 19.3: Full end-to-end agent run where the LLM always returns broken
    code, forcing the agent to exhaust all retries.

    Uses ``max_retries=2`` (3 total sandbox executions) to keep the test fast.

    Verifies:
    - ``success=False``
    - ``retry_count == max_retries`` (i.e. 2)
    - All required failure-report fields are populated
    """

    def test_success_is_false(self, max_retries_result):
        """success must be False when all retries are exhausted."""
        assert max_retries_result.success is False, (
            f"Expected success=False, got success={max_retries_result.success}. "
            f"execution_logs={max_retries_result.execution_logs!r}"
        )

    def test_retry_count_equals_max_retries(self, max_retries_result):
        """retry_count must equal max_retries (2) after exhaustion."""
        assert max_retries_result.retry_count == _MAX_RETRIES_VALUE, (
            f"Expected retry_count={_MAX_RETRIES_VALUE}, "
            f"got retry_count={max_retries_result.retry_count}. "
            f"execution_logs={max_retries_result.execution_logs!r}"
        )

    def test_source_code_is_non_none(self, max_retries_result):
        """source_code must not be None (the broken code is still returned)."""
        assert max_retries_result.source_code is not None

    def test_execution_logs_is_non_none(self, max_retries_result):
        """execution_logs must not be None."""
        assert max_retries_result.execution_logs is not None

    def test_failure_reason_is_populated(self, max_retries_result):
        """failure_reason must be a non-empty string when success=False."""
        assert max_retries_result.failure_reason is not None, (
            "Expected failure_reason to be populated, got None"
        )
        assert len(max_retries_result.failure_reason) > 0, (
            "Expected failure_reason to be non-empty"
        )

    def test_no_exception_raised(self, max_retries_result):
        """The run must complete without raising — result is an AgentResponse."""
        assert isinstance(max_retries_result, AgentResponse)

    def test_returns_agent_response_type(self, max_retries_result):
        """Result must be an AgentResponse instance."""
        assert isinstance(max_retries_result, AgentResponse)

    def test_retry_count_is_non_negative(self, max_retries_result):
        """retry_count must be >= 0."""
        assert max_retries_result.retry_count >= 0

    def test_token_usage_write_node_non_zero(self, max_retries_result):
        """write_node must have recorded token usage (called multiple times)."""
        assert max_retries_result.token_usage.write_node.input_tokens > 0, (
            f"Expected write_node.input_tokens > 0, "
            f"got {max_retries_result.token_usage.write_node.input_tokens}"
        )

    def test_elapsed_seconds_is_positive(self, max_retries_result):
        """elapsed_seconds must be a positive float."""
        assert max_retries_result.elapsed_seconds > 0, (
            f"Expected elapsed_seconds > 0, got {max_retries_result.elapsed_seconds}"
        )

    def test_execution_logs_contain_error(self, max_retries_result):
        """execution_logs must contain some error text from the failed execution."""
        logs = max_retries_result.execution_logs
        assert logs, "Expected non-empty execution_logs"
        has_error = (
            "SyntaxError" in logs
            or "Error" in logs
            or "error" in logs
            or len(logs) > 0
        )
        assert has_error, (
            f"Expected execution_logs to contain error text, got {logs!r}"
        )


# --------------------------------------------------------------------------- #
# Task 19.4 — Explanation and test generation enabled
# --------------------------------------------------------------------------- #


class TestExplanationAndTestGenerationEnabled:
    """
    Task 19.4: Full end-to-end agent run with both explanation and test
    generation enabled.

    The mock LLM returns:
    - A hello-world ``CodeArtifact`` for the write node.
    - A passing pytest suite ``CodeArtifact`` for the test node.
    - A canned ``ExplanationOutput`` for the post-process node.
    - A ``ConfidenceOutput`` with score 4.

    The real subprocess+venv sandbox executes both the source code and the
    generated test suite.

    Verifies:
    - ``success=True``
    - ``AgentResponse.explanation`` is non-None
    - ``AgentResponse.test_code`` is non-None
    """

    def test_success_is_true(self, explanation_and_test_result):
        """success must be True for a clean run with both features enabled."""
        assert explanation_and_test_result.success is True, (
            f"Expected success=True, got success={explanation_and_test_result.success}. "
            f"failure_reason={explanation_and_test_result.failure_reason!r}, "
            f"execution_logs={explanation_and_test_result.execution_logs!r}"
        )

    def test_explanation_is_non_none(self, explanation_and_test_result):
        """Primary assertion: explanation must be non-None when explanation_enabled=True."""
        assert explanation_and_test_result.explanation is not None, (
            "Expected explanation to be non-None when explanation_enabled=True, "
            f"got explanation={explanation_and_test_result.explanation!r}"
        )

    def test_test_code_is_non_none(self, explanation_and_test_result):
        """Primary assertion: test_code must be non-None when test_generation_enabled=True."""
        assert explanation_and_test_result.test_code is not None, (
            "Expected test_code to be non-None when test_generation_enabled=True, "
            f"got test_code={explanation_and_test_result.test_code!r}"
        )

    def test_explanation_is_non_empty_string(self, explanation_and_test_result):
        """explanation must be a non-empty string."""
        assert isinstance(explanation_and_test_result.explanation, str)
        assert explanation_and_test_result.explanation.strip(), (
            f"Expected non-empty explanation, got {explanation_and_test_result.explanation!r}"
        )

    def test_test_code_is_non_empty_string(self, explanation_and_test_result):
        """test_code must be a non-empty string."""
        assert isinstance(explanation_and_test_result.test_code, str)
        assert explanation_and_test_result.test_code.strip(), (
            f"Expected non-empty test_code, got {explanation_and_test_result.test_code!r}"
        )

    def test_source_code_is_non_empty(self, explanation_and_test_result):
        """source_code must be a non-empty string."""
        assert explanation_and_test_result.source_code
        assert isinstance(explanation_and_test_result.source_code, str)

    def test_failure_reason_is_none(self, explanation_and_test_result):
        """failure_reason must be None on a successful run."""
        assert explanation_and_test_result.failure_reason is None, (
            f"Expected failure_reason=None, got {explanation_and_test_result.failure_reason!r}"
        )

    def test_confidence_score_in_valid_range(self, explanation_and_test_result):
        """confidence_score must be an integer in [1, 5]."""
        assert explanation_and_test_result.confidence_score is not None
        assert isinstance(explanation_and_test_result.confidence_score, int)
        assert 1 <= explanation_and_test_result.confidence_score <= 5

    def test_returns_agent_response_type(self, explanation_and_test_result):
        """Result must be an AgentResponse instance."""
        assert isinstance(explanation_and_test_result, AgentResponse)

    def test_elapsed_seconds_is_positive(self, explanation_and_test_result):
        """elapsed_seconds must be a positive float."""
        assert explanation_and_test_result.elapsed_seconds > 0


# --------------------------------------------------------------------------- #
# Task 19.5 — Property 19: explanation is non-empty string when enabled
# --------------------------------------------------------------------------- #
# Moved to tests/test_property19.py to avoid paying the module-level venv
# setup cost every time this property test is run in isolation.
# Re-exported here so the full suite still picks it up.
from tests.test_property19 import (  # noqa: E402, F401
    test_property19_explanation_is_non_empty_string_when_enabled,
)
