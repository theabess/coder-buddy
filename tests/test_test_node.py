"""
Unit tests for test_node.

Validates four scenarios:
1. Test generation disabled: when ``config.test_generation_enabled=False``,
   the node returns ``{test_code: None, test_logs: None}`` immediately.
2. Successful test generation: LLM returns a test suite that passes on the
   first attempt → ``{test_code, test_logs}`` with no warning.
3. Retry on failure: when the test suite fails, the node retries up to 3
   times; on eventual success, returns ``{test_code, test_logs}`` with no
   warning.
4. All retries exhausted: when all 3 attempts fail, the node returns the
   last test code with a warning message.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from coder_buddy.models import CodeArtifact, TokenRecord, TokenUsage
from coder_buddy.nodes.test_node import (
    _build_runner_script,
    _build_test_prompt,
    make_test_node,
)
from coder_buddy.sandbox.base import ExecutionResult


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_state(
    *,
    current_code: str = "def add(a, b):\n    return a + b\n",
    file_name: str = "main.py",
    dependencies: list[str] | None = None,
    retry_count: int = 0,
    max_retries: int = 5,
) -> dict:
    """Build a minimal AgentState dict for test_node testing."""
    return {
        "current_code": current_code,
        "file_name": file_name,
        "dependencies": dependencies if dependencies is not None else [],
        "retry_count": retry_count,
        "user_prompt": "Write a script",
        "execution_logs": "",
        "error_status": False,
        "language": "python",
        "explanation": None,
        "test_code": None,
        "test_logs": None,
        "confidence_score": None,
        "refactor_diff": None,
        "token_usage": TokenUsage(),
        "session_history": [],
        "max_retries": max_retries,
        "pre_refactor_code": None,
    }


def _make_mock_llm_client(
    test_source_code: str = "def test_add():\n    assert add(1, 2) == 3\n",
    file_name: str = "test_main.py",
) -> MagicMock:
    """Return a mock LLMClient whose generate() returns a valid CodeArtifact."""
    artifact = CodeArtifact(
        source_code=test_source_code,
        file_name=file_name,
        dependencies=[],
        language="python",
    )
    token_record = TokenRecord(input_tokens=100, output_tokens=50)
    mock_client = MagicMock()
    mock_client.generate.return_value = (artifact, token_record)
    return mock_client


def _make_mock_config(
    test_generation_enabled: bool = True,
    sandbox_timeout_seconds: float = 10.0,
) -> MagicMock:
    """Return a mock AgentConfig."""
    config = MagicMock()
    config.test_generation_enabled = test_generation_enabled
    config.sandbox_timeout_seconds = sandbox_timeout_seconds
    return config


def _make_sandbox(
    *,
    stdout: str = "1 passed",
    stderr: str = "",
    exit_code: int = 0,
    timed_out: bool = False,
) -> MagicMock:
    """Return a mock SandboxBackend whose execute() returns the given result."""
    result = ExecutionResult(
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        timed_out=timed_out,
    )
    sandbox = MagicMock()
    sandbox.execute.return_value = result
    return sandbox


# ---------------------------------------------------------------------------
# Scenario 1: Test generation disabled
# ---------------------------------------------------------------------------


class TestGenerationDisabled:
    """Scenario 1: When test_generation_enabled=False, node returns None fields."""

    def test_returns_test_code_none_when_disabled(self):
        """test_code is None when test_generation_enabled=False."""
        sandbox = _make_sandbox()
        llm_client = _make_mock_llm_client()
        config = _make_mock_config(test_generation_enabled=False)
        node = make_test_node(sandbox, llm_client, config)

        with patch("coder_buddy.nodes.test_node.log_node_event"):
            result = node(_make_state())

        assert result["test_code"] is None

    def test_returns_test_logs_none_when_disabled(self):
        """test_logs is None when test_generation_enabled=False."""
        sandbox = _make_sandbox()
        llm_client = _make_mock_llm_client()
        config = _make_mock_config(test_generation_enabled=False)
        node = make_test_node(sandbox, llm_client, config)

        with patch("coder_buddy.nodes.test_node.log_node_event"):
            result = node(_make_state())

        assert result["test_logs"] is None

    def test_llm_not_called_when_disabled(self):
        """LLMClient.generate is NOT called when test_generation_enabled=False."""
        sandbox = _make_sandbox()
        llm_client = _make_mock_llm_client()
        config = _make_mock_config(test_generation_enabled=False)
        node = make_test_node(sandbox, llm_client, config)

        with patch("coder_buddy.nodes.test_node.log_node_event"):
            node(_make_state())

        llm_client.generate.assert_not_called()

    def test_sandbox_not_called_when_disabled(self):
        """sandbox.execute is NOT called when test_generation_enabled=False."""
        sandbox = _make_sandbox()
        llm_client = _make_mock_llm_client()
        config = _make_mock_config(test_generation_enabled=False)
        node = make_test_node(sandbox, llm_client, config)

        with patch("coder_buddy.nodes.test_node.log_node_event"):
            node(_make_state())

        sandbox.execute.assert_not_called()

    def test_no_warning_when_disabled(self):
        """No warning key is set when test_generation_enabled=False."""
        sandbox = _make_sandbox()
        llm_client = _make_mock_llm_client()
        config = _make_mock_config(test_generation_enabled=False)
        node = make_test_node(sandbox, llm_client, config)

        with patch("coder_buddy.nodes.test_node.log_node_event"):
            result = node(_make_state())

        assert "warning" not in result

    def test_token_usage_unchanged_when_disabled(self):
        """token_usage is returned unchanged when test_generation_enabled=False."""
        sandbox = _make_sandbox()
        llm_client = _make_mock_llm_client()
        config = _make_mock_config(test_generation_enabled=False)
        node = make_test_node(sandbox, llm_client, config)
        original_usage = TokenUsage()
        state = _make_state()
        state["token_usage"] = original_usage

        with patch("coder_buddy.nodes.test_node.log_node_event"):
            result = node(state)

        assert result["token_usage"] is original_usage


# ---------------------------------------------------------------------------
# Scenario 2: Successful test generation on first attempt
# ---------------------------------------------------------------------------


class TestSuccessfulGeneration:
    """Scenario 2: LLM returns a test suite that passes on the first attempt."""

    def test_test_code_set_to_artifact_source_code(self):
        """test_code in the result equals the LLM artifact's source_code."""
        test_source = "def test_add():\n    assert 1 + 1 == 2\n"
        sandbox = _make_sandbox(stdout="1 passed", exit_code=0)
        llm_client = _make_mock_llm_client(test_source_code=test_source)
        config = _make_mock_config()
        node = make_test_node(sandbox, llm_client, config)

        with patch("coder_buddy.nodes.test_node.log_node_event"):
            result = node(_make_state())

        assert result["test_code"] == test_source

    def test_test_logs_set_to_sandbox_output(self):
        """test_logs in the result equals the sandbox combined_output."""
        sandbox = _make_sandbox(stdout="1 passed in 0.01s", exit_code=0)
        llm_client = _make_mock_llm_client()
        config = _make_mock_config()
        node = make_test_node(sandbox, llm_client, config)

        with patch("coder_buddy.nodes.test_node.log_node_event"):
            result = node(_make_state())

        assert "1 passed" in result["test_logs"]

    def test_no_warning_on_success(self):
        """No warning is set when tests pass on the first attempt."""
        sandbox = _make_sandbox(stdout="1 passed", exit_code=0)
        llm_client = _make_mock_llm_client()
        config = _make_mock_config()
        node = make_test_node(sandbox, llm_client, config)

        with patch("coder_buddy.nodes.test_node.log_node_event"):
            result = node(_make_state())

        assert "warning" not in result

    def test_llm_called_once_on_first_success(self):
        """LLMClient.generate is called exactly once when tests pass first time."""
        sandbox = _make_sandbox(exit_code=0)
        llm_client = _make_mock_llm_client()
        config = _make_mock_config()
        node = make_test_node(sandbox, llm_client, config)

        with patch("coder_buddy.nodes.test_node.log_node_event"):
            node(_make_state())

        assert llm_client.generate.call_count == 1

    def test_sandbox_execute_called_once_on_first_success(self):
        """sandbox.execute is called exactly once when tests pass first time."""
        sandbox = _make_sandbox(exit_code=0)
        llm_client = _make_mock_llm_client()
        config = _make_mock_config()
        node = make_test_node(sandbox, llm_client, config)

        with patch("coder_buddy.nodes.test_node.log_node_event"):
            node(_make_state())

        assert sandbox.execute.call_count == 1

    def test_sandbox_cleanup_called_on_success(self):
        """sandbox.cleanup() is called after a successful test run."""
        sandbox = _make_sandbox(exit_code=0)
        llm_client = _make_mock_llm_client()
        config = _make_mock_config()
        node = make_test_node(sandbox, llm_client, config)

        with patch("coder_buddy.nodes.test_node.log_node_event"):
            node(_make_state())

        sandbox.cleanup.assert_called()

    def test_token_usage_updated_on_success(self):
        """token_usage has the test_node record updated after a successful run."""
        sandbox = _make_sandbox(exit_code=0)
        llm_client = _make_mock_llm_client()
        config = _make_mock_config()
        node = make_test_node(sandbox, llm_client, config)

        with patch("coder_buddy.nodes.test_node.log_node_event"):
            result = node(_make_state())

        assert result["token_usage"].test_node.input_tokens == 100
        assert result["token_usage"].test_node.output_tokens == 50

    def test_result_contains_required_keys(self):
        """The result dict contains test_code, test_logs, and token_usage."""
        sandbox = _make_sandbox(exit_code=0)
        llm_client = _make_mock_llm_client()
        config = _make_mock_config()
        node = make_test_node(sandbox, llm_client, config)

        with patch("coder_buddy.nodes.test_node.log_node_event"):
            result = node(_make_state())

        assert "test_code" in result
        assert "test_logs" in result
        assert "token_usage" in result


# ---------------------------------------------------------------------------
# Scenario 3: Retry on failure — eventual success
# ---------------------------------------------------------------------------


class TestRetryOnFailure:
    """Scenario 3: Tests fail initially but succeed after retries."""

    def test_success_after_one_retry(self):
        """Node succeeds after one failure and one retry."""
        fail_result = ExecutionResult(
            stdout="", stderr="FAILED test_add", exit_code=1, timed_out=False
        )
        pass_result = ExecutionResult(
            stdout="1 passed", stderr="", exit_code=0, timed_out=False
        )

        sandbox = MagicMock()
        sandbox.execute.side_effect = [fail_result, pass_result]

        llm_client = _make_mock_llm_client()
        config = _make_mock_config()
        node = make_test_node(sandbox, llm_client, config)

        with patch("coder_buddy.nodes.test_node.log_node_event"):
            result = node(_make_state())

        assert result["test_code"] is not None
        assert "warning" not in result

    def test_llm_called_twice_after_one_retry(self):
        """LLMClient.generate is called twice: once for initial, once for retry."""
        fail_result = ExecutionResult(
            stdout="", stderr="FAILED", exit_code=1, timed_out=False
        )
        pass_result = ExecutionResult(
            stdout="1 passed", stderr="", exit_code=0, timed_out=False
        )

        sandbox = MagicMock()
        sandbox.execute.side_effect = [fail_result, pass_result]

        llm_client = _make_mock_llm_client()
        config = _make_mock_config()
        node = make_test_node(sandbox, llm_client, config)

        with patch("coder_buddy.nodes.test_node.log_node_event"):
            node(_make_state())

        assert llm_client.generate.call_count == 2

    def test_failure_logs_included_in_retry_prompt(self):
        """The failure logs from the first attempt are included in the retry prompt."""
        fail_logs = "FAILED test_add - AssertionError"
        fail_result = ExecutionResult(
            stdout=fail_logs, stderr="", exit_code=1, timed_out=False
        )
        pass_result = ExecutionResult(
            stdout="1 passed", stderr="", exit_code=0, timed_out=False
        )

        sandbox = MagicMock()
        sandbox.execute.side_effect = [fail_result, pass_result]

        llm_client = _make_mock_llm_client()
        config = _make_mock_config()
        node = make_test_node(sandbox, llm_client, config)

        with patch("coder_buddy.nodes.test_node.log_node_event"):
            node(_make_state())

        # The second call to generate should include the failure logs
        assert llm_client.generate.call_count == 2
        second_call_prompt = llm_client.generate.call_args_list[1][0][0]
        assert fail_logs in second_call_prompt

    def test_no_warning_on_eventual_success(self):
        """No warning is set when tests eventually pass after retries."""
        fail_result = ExecutionResult(
            stdout="", stderr="FAILED", exit_code=1, timed_out=False
        )
        pass_result = ExecutionResult(
            stdout="1 passed", stderr="", exit_code=0, timed_out=False
        )

        sandbox = MagicMock()
        sandbox.execute.side_effect = [fail_result, pass_result]

        llm_client = _make_mock_llm_client()
        config = _make_mock_config()
        node = make_test_node(sandbox, llm_client, config)

        with patch("coder_buddy.nodes.test_node.log_node_event"):
            result = node(_make_state())

        assert "warning" not in result

    def test_cleanup_called_after_each_attempt(self):
        """sandbox.cleanup() is called after each execution attempt."""
        fail_result = ExecutionResult(
            stdout="", stderr="FAILED", exit_code=1, timed_out=False
        )
        pass_result = ExecutionResult(
            stdout="1 passed", stderr="", exit_code=0, timed_out=False
        )

        sandbox = MagicMock()
        sandbox.execute.side_effect = [fail_result, pass_result]

        llm_client = _make_mock_llm_client()
        config = _make_mock_config()
        node = make_test_node(sandbox, llm_client, config)

        with patch("coder_buddy.nodes.test_node.log_node_event"):
            node(_make_state())

        # cleanup should be called once per execution attempt (2 attempts)
        assert sandbox.cleanup.call_count == 2


# ---------------------------------------------------------------------------
# Scenario 4: All retries exhausted — warning returned
# ---------------------------------------------------------------------------


class TestAllRetriesExhausted:
    """Scenario 4: All 3 attempts fail → last test code returned with warning."""

    def _make_always_failing_sandbox(self) -> MagicMock:
        """Return a sandbox that always returns a failing result."""
        fail_result = ExecutionResult(
            stdout="", stderr="FAILED test_add", exit_code=1, timed_out=False
        )
        sandbox = MagicMock()
        sandbox.execute.return_value = fail_result
        return sandbox

    def test_warning_set_after_all_retries_fail(self):
        """A warning message is set when all 3 test attempts fail."""
        sandbox = self._make_always_failing_sandbox()
        llm_client = _make_mock_llm_client()
        config = _make_mock_config()
        node = make_test_node(sandbox, llm_client, config)

        with patch("coder_buddy.nodes.test_node.log_node_event"):
            result = node(_make_state())

        assert "warning" in result
        assert result["warning"] is not None
        assert len(result["warning"]) > 0

    def test_test_code_returned_after_all_retries_fail(self):
        """test_code is still returned (not None) after all retries fail."""
        sandbox = self._make_always_failing_sandbox()
        llm_client = _make_mock_llm_client()
        config = _make_mock_config()
        node = make_test_node(sandbox, llm_client, config)

        with patch("coder_buddy.nodes.test_node.log_node_event"):
            result = node(_make_state())

        assert result["test_code"] is not None

    def test_test_logs_returned_after_all_retries_fail(self):
        """test_logs contains the last failure output after all retries fail."""
        fail_stderr = "FAILED test_add - AssertionError: assert 0 == 1"
        fail_result = ExecutionResult(
            stdout="", stderr=fail_stderr, exit_code=1, timed_out=False
        )
        sandbox = MagicMock()
        sandbox.execute.return_value = fail_result

        llm_client = _make_mock_llm_client()
        config = _make_mock_config()
        node = make_test_node(sandbox, llm_client, config)

        with patch("coder_buddy.nodes.test_node.log_node_event"):
            result = node(_make_state())

        assert result["test_logs"] is not None
        assert fail_stderr in result["test_logs"]

    def test_llm_called_three_times_when_all_fail(self):
        """LLMClient.generate is called 3 times (initial + 2 retries) when all fail."""
        sandbox = self._make_always_failing_sandbox()
        llm_client = _make_mock_llm_client()
        config = _make_mock_config()
        node = make_test_node(sandbox, llm_client, config)

        with patch("coder_buddy.nodes.test_node.log_node_event"):
            node(_make_state())

        assert llm_client.generate.call_count == 3

    def test_sandbox_execute_called_three_times_when_all_fail(self):
        """sandbox.execute is called 3 times when all attempts fail."""
        sandbox = self._make_always_failing_sandbox()
        llm_client = _make_mock_llm_client()
        config = _make_mock_config()
        node = make_test_node(sandbox, llm_client, config)

        with patch("coder_buddy.nodes.test_node.log_node_event"):
            node(_make_state())

        assert sandbox.execute.call_count == 3

    def test_cleanup_called_three_times_when_all_fail(self):
        """sandbox.cleanup() is called 3 times (once per attempt) when all fail."""
        sandbox = self._make_always_failing_sandbox()
        llm_client = _make_mock_llm_client()
        config = _make_mock_config()
        node = make_test_node(sandbox, llm_client, config)

        with patch("coder_buddy.nodes.test_node.log_node_event"):
            node(_make_state())

        assert sandbox.cleanup.call_count == 3

    def test_warning_mentions_retry_count(self):
        """The warning message mentions the number of failed attempts."""
        sandbox = self._make_always_failing_sandbox()
        llm_client = _make_mock_llm_client()
        config = _make_mock_config()
        node = make_test_node(sandbox, llm_client, config)

        with patch("coder_buddy.nodes.test_node.log_node_event"):
            result = node(_make_state())

        # Warning should mention the number of attempts
        assert "3" in result["warning"]

    def test_source_code_not_modified_after_all_retries_fail(self):
        """current_code in state is NOT modified when all test retries fail."""
        original_code = "def add(a, b):\n    return a + b\n"
        sandbox = self._make_always_failing_sandbox()
        llm_client = _make_mock_llm_client()
        config = _make_mock_config()
        node = make_test_node(sandbox, llm_client, config)
        state = _make_state(current_code=original_code)

        with patch("coder_buddy.nodes.test_node.log_node_event"):
            result = node(state)

        # The result dict should NOT contain current_code (source_code unchanged)
        assert "current_code" not in result


# ---------------------------------------------------------------------------
# Tests for _build_test_prompt
# ---------------------------------------------------------------------------


class TestBuildTestPrompt:
    """Unit tests for the _build_test_prompt helper."""

    def test_initial_prompt_contains_source_code(self):
        """The initial prompt (no failure_logs) contains the source code."""
        source_code = "def add(a, b):\n    return a + b\n"
        prompt = _build_test_prompt(
            source_code=source_code,
            file_name="main.py",
            failure_logs=None,
        )
        assert source_code in prompt

    def test_initial_prompt_contains_file_name(self):
        """The initial prompt contains the file_name."""
        prompt = _build_test_prompt(
            source_code="print('hello')",
            file_name="calculator.py",
            failure_logs=None,
        )
        assert "calculator.py" in prompt

    def test_initial_prompt_mentions_pytest(self):
        """The initial prompt instructs the LLM to use pytest."""
        prompt = _build_test_prompt(
            source_code="print('hello')",
            file_name="main.py",
            failure_logs=None,
        )
        assert "pytest" in prompt.lower()

    def test_retry_prompt_contains_failure_logs(self):
        """The retry prompt (with failure_logs) contains the failure output."""
        failure_logs = "FAILED test_add - AssertionError: assert 0 == 1"
        prompt = _build_test_prompt(
            source_code="def add(a, b): return a + b",
            file_name="main.py",
            failure_logs=failure_logs,
        )
        assert failure_logs in prompt

    def test_retry_prompt_contains_source_code(self):
        """The retry prompt also contains the source code."""
        source_code = "def multiply(a, b):\n    return a * b\n"
        prompt = _build_test_prompt(
            source_code=source_code,
            file_name="main.py",
            failure_logs="FAILED",
        )
        assert source_code in prompt

    def test_initial_prompt_no_failure_logs_section(self):
        """The initial prompt does not contain a failure output section."""
        prompt = _build_test_prompt(
            source_code="print('hello')",
            file_name="main.py",
            failure_logs=None,
        )
        assert "Failure output" not in prompt

    def test_retry_prompt_mentions_revise(self):
        """The retry prompt asks the LLM to revise the tests."""
        prompt = _build_test_prompt(
            source_code="print('hello')",
            file_name="main.py",
            failure_logs="some error",
        )
        assert "revis" in prompt.lower()


# ---------------------------------------------------------------------------
# Tests for _build_runner_script
# ---------------------------------------------------------------------------


class TestBuildRunnerScript:
    """Unit tests for the _build_runner_script helper."""

    def test_runner_script_contains_source_code(self):
        """The runner script embeds the source code."""
        source_code = "def add(a, b):\n    return a + b\n"
        runner = _build_runner_script(
            test_code="def test_add(): pass",
            source_code=source_code,
            file_name="main.py",
        )
        assert repr(source_code) in runner

    def test_runner_script_contains_test_code(self):
        """The runner script embeds the test code."""
        test_code = "def test_add():\n    assert add(1, 2) == 3\n"
        runner = _build_runner_script(
            test_code=test_code,
            source_code="def add(a, b): return a + b",
            file_name="main.py",
        )
        assert repr(test_code) in runner

    def test_runner_script_imports_pytest(self):
        """The runner script imports pytest."""
        runner = _build_runner_script(
            test_code="def test_x(): pass",
            source_code="x = 1",
            file_name="main.py",
        )
        assert "import pytest" in runner

    def test_runner_script_calls_pytest_main(self):
        """The runner script calls pytest.main to execute the tests."""
        runner = _build_runner_script(
            test_code="def test_x(): pass",
            source_code="x = 1",
            file_name="main.py",
        )
        assert "pytest.main" in runner

    def test_runner_script_exits_with_pytest_exit_code(self):
        """The runner script calls sys.exit with pytest's exit code."""
        runner = _build_runner_script(
            test_code="def test_x(): pass",
            source_code="x = 1",
            file_name="main.py",
        )
        assert "sys.exit" in runner

    def test_runner_script_is_valid_python(self):
        """The runner script is syntactically valid Python."""
        import ast

        runner = _build_runner_script(
            test_code="def test_add():\n    assert 1 + 1 == 2\n",
            source_code="def add(a, b):\n    return a + b\n",
            file_name="main.py",
        )
        # Should not raise SyntaxError
        ast.parse(runner)

    def test_runner_script_adds_tmpdir_to_sys_path(self):
        """The runner script adds the temp directory to sys.path."""
        runner = _build_runner_script(
            test_code="def test_x(): pass",
            source_code="x = 1",
            file_name="main.py",
        )
        assert "sys.path" in runner


# ---------------------------------------------------------------------------
# Integration: log_node_event is called at start and end
# ---------------------------------------------------------------------------


class TestLogNodeEventCalls:
    """Verify log_node_event is called at start and end of test_node."""

    def test_log_node_event_called_at_start(self):
        """log_node_event is called with event='start' at the beginning."""
        sandbox = _make_sandbox(exit_code=0)
        llm_client = _make_mock_llm_client()
        config = _make_mock_config()
        node = make_test_node(sandbox, llm_client, config)

        with patch("coder_buddy.nodes.test_node.log_node_event") as mock_log:
            node(_make_state())

        start_calls = [
            c for c in mock_log.call_args_list if c.kwargs.get("event") == "start"
        ]
        assert len(start_calls) >= 1

    def test_log_node_event_called_at_end(self):
        """log_node_event is called with event='end' at the end."""
        sandbox = _make_sandbox(exit_code=0)
        llm_client = _make_mock_llm_client()
        config = _make_mock_config()
        node = make_test_node(sandbox, llm_client, config)

        with patch("coder_buddy.nodes.test_node.log_node_event") as mock_log:
            node(_make_state())

        end_calls = [
            c for c in mock_log.call_args_list if c.kwargs.get("event") == "end"
        ]
        assert len(end_calls) >= 1

    def test_log_node_event_called_with_node_name(self):
        """log_node_event is called with node='test_node'."""
        sandbox = _make_sandbox(exit_code=0)
        llm_client = _make_mock_llm_client()
        config = _make_mock_config()
        node = make_test_node(sandbox, llm_client, config)

        with patch("coder_buddy.nodes.test_node.log_node_event") as mock_log:
            node(_make_state())

        for c in mock_log.call_args_list:
            assert c.kwargs.get("node") == "test_node"


# ---------------------------------------------------------------------------
# Token accumulation via model_copy(update=...) — task 20.2
# ---------------------------------------------------------------------------


class TestTestNodeTokenUsageAccumulation:
    """
    Verify that test_node correctly accumulates its TokenRecord into
    AgentState.token_usage using model_copy(update={"test_node": token_record}).
    """

    def test_token_usage_other_fields_unchanged_on_success(self):
        """
        After a successful test run, all TokenRecord fields other than
        test_node retain their prior zero values.
        """
        sandbox = _make_sandbox(exit_code=0)
        llm_client = _make_mock_llm_client()
        config = _make_mock_config()
        node = make_test_node(sandbox, llm_client, config)
        state = _make_state()

        with patch("coder_buddy.nodes.test_node.log_node_event"):
            result = node(state)

        usage = result["token_usage"]
        assert usage.write_node.input_tokens == 0
        assert usage.write_node.output_tokens == 0
        assert usage.refactor_node.input_tokens == 0
        assert usage.refactor_node.output_tokens == 0
        assert usage.explanation.input_tokens == 0
        assert usage.explanation.output_tokens == 0
        assert usage.confidence.input_tokens == 0
        assert usage.confidence.output_tokens == 0
        # test_node is the only updated field
        assert usage.test_node.input_tokens == 100
        assert usage.test_node.output_tokens == 50

    def test_token_usage_prior_values_preserved_on_success(self):
        """
        When the input state already has non-zero token_usage for other nodes,
        test_node only updates the test_node field and leaves others intact.
        """
        prior_usage = TokenUsage(
            write_node=TokenRecord(input_tokens=400, output_tokens=200),
            refactor_node=TokenRecord(input_tokens=250, output_tokens=125),
        )
        sandbox = _make_sandbox(exit_code=0)
        llm_client = _make_mock_llm_client()
        config = _make_mock_config()
        node = make_test_node(sandbox, llm_client, config)
        state = _make_state()
        state["token_usage"] = prior_usage

        with patch("coder_buddy.nodes.test_node.log_node_event"):
            result = node(state)

        usage = result["token_usage"]
        # test_node field updated
        assert usage.test_node.input_tokens == 100
        assert usage.test_node.output_tokens == 50
        # prior values preserved
        assert usage.write_node.input_tokens == 400
        assert usage.write_node.output_tokens == 200
        assert usage.refactor_node.input_tokens == 250
        assert usage.refactor_node.output_tokens == 125

    def test_token_usage_is_new_object_not_same_reference(self):
        """
        test_node creates a new TokenUsage via model_copy, so the returned
        token_usage is a different object from the input state's token_usage.
        """
        sandbox = _make_sandbox(exit_code=0)
        llm_client = _make_mock_llm_client()
        config = _make_mock_config()
        node = make_test_node(sandbox, llm_client, config)
        original_usage = TokenUsage()
        state = _make_state()
        state["token_usage"] = original_usage

        with patch("coder_buddy.nodes.test_node.log_node_event"):
            result = node(state)

        assert result["token_usage"] is not original_usage

    def test_token_usage_accumulates_across_retries(self):
        """
        When test_node retries (due to test failure), each retry call to
        llm_client.generate overwrites the test_node field via model_copy.
        The final token_usage reflects the last LLM call's token record.
        """
        fail_result = ExecutionResult(
            stdout="", stderr="FAILED", exit_code=1, timed_out=False
        )
        pass_result = ExecutionResult(
            stdout="1 passed", stderr="", exit_code=0, timed_out=False
        )
        sandbox = MagicMock()
        sandbox.execute.side_effect = [fail_result, pass_result]

        # Each generate call returns the same token record (100 in, 50 out)
        llm_client = _make_mock_llm_client()
        config = _make_mock_config()
        node = make_test_node(sandbox, llm_client, config)
        state = _make_state()

        with patch("coder_buddy.nodes.test_node.log_node_event"):
            result = node(state)

        # After 2 LLM calls, test_node field reflects the last record
        usage = result["token_usage"]
        assert usage.test_node.input_tokens == 100
        assert usage.test_node.output_tokens == 50
        # Other fields untouched
        assert usage.write_node.input_tokens == 0
        assert usage.refactor_node.input_tokens == 0
