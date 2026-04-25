"""
Unit tests for execute_node.

Validates four scenarios:
1. Successful execution sets ``error_status=False``.
2. Failed execution (non-zero exit code) sets ``error_status=True``.
3. Timeout sets ``error_status=True`` with the timeout message in ``execution_logs``.
4. ``sandbox.cleanup()`` is always called, even when an exception is raised
   during execution.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from coder_buddy.nodes.execute_node import make_execute_node
from coder_buddy.sandbox.base import ExecutionResult


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_sandbox(
    *,
    stdout: str = "",
    stderr: str = "",
    exit_code: int = 0,
    timed_out: bool = False,
) -> MagicMock:
    """Return a mock SandboxBackend whose ``execute`` returns the given result."""
    result = ExecutionResult(
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        timed_out=timed_out,
    )
    sandbox = MagicMock()
    sandbox.execute.return_value = result
    return sandbox


def _make_config(timeout: float = 10.0) -> MagicMock:
    """Return a mock AgentConfig with the given ``sandbox_timeout_seconds``."""
    config = MagicMock()
    config.sandbox_timeout_seconds = timeout
    return config


def _make_state(
    *,
    current_code: str = "print('hello')",
    dependencies: list[str] | None = None,
    retry_count: int = 0,
) -> dict:
    """Build a minimal AgentState dict for execute_node testing."""
    return {
        "current_code": current_code,
        "dependencies": dependencies if dependencies is not None else [],
        "retry_count": retry_count,
        # Remaining fields are not read by execute_node but keep the dict
        # structurally complete.
        "user_prompt": "Write a script",
        "execution_logs": "",
        "error_status": False,
        "file_name": "main.py",
        "language": "python",
        "explanation": None,
        "test_code": None,
        "test_logs": None,
        "confidence_score": None,
        "refactor_diff": None,
        "token_usage": MagicMock(),
        "session_history": [],
        "max_retries": 5,
        "pre_refactor_code": None,
    }


# ---------------------------------------------------------------------------
# Scenario 1: Successful execution sets error_status=False
# ---------------------------------------------------------------------------


class TestSuccessfulExecution:
    """Scenario 1: A zero-exit-code, non-timed-out result sets error_status=False."""

    def test_error_status_false_on_success(self):
        """Successful execution (exit_code=0, timed_out=False) sets error_status=False."""
        sandbox = _make_sandbox(stdout="Hello, world!\n", exit_code=0, timed_out=False)
        config = _make_config()
        node = make_execute_node(sandbox, config)

        with patch("coder_buddy.nodes.execute_node.log_node_event"):
            result = node(_make_state())

        assert result["error_status"] is False

    def test_execution_logs_contain_stdout_on_success(self):
        """execution_logs contains the stdout from a successful run."""
        sandbox = _make_sandbox(stdout="42\n", exit_code=0)
        config = _make_config()
        node = make_execute_node(sandbox, config)

        with patch("coder_buddy.nodes.execute_node.log_node_event"):
            result = node(_make_state())

        assert "42" in result["execution_logs"]

    def test_execute_called_with_correct_code_and_timeout(self):
        """sandbox.execute is called with current_code and the configured timeout."""
        sandbox = _make_sandbox(exit_code=0)
        config = _make_config(timeout=30.0)
        node = make_execute_node(sandbox, config)
        code = "x = 1 + 1\nprint(x)"

        with patch("coder_buddy.nodes.execute_node.log_node_event"):
            node(_make_state(current_code=code))

        sandbox.execute.assert_called_once_with(code, 30.0)

    def test_install_dependencies_not_called_when_empty(self):
        """install_dependencies is NOT called when dependencies list is empty."""
        sandbox = _make_sandbox(exit_code=0)
        config = _make_config()
        node = make_execute_node(sandbox, config)

        with patch("coder_buddy.nodes.execute_node.log_node_event"):
            node(_make_state(dependencies=[]))

        sandbox.install_dependencies.assert_not_called()

    def test_install_dependencies_called_when_non_empty(self):
        """install_dependencies IS called when dependencies list is non-empty."""
        sandbox = _make_sandbox(exit_code=0)
        config = _make_config()
        node = make_execute_node(sandbox, config)
        deps = ["requests", "numpy"]

        with patch("coder_buddy.nodes.execute_node.log_node_event"):
            node(_make_state(dependencies=deps))

        sandbox.install_dependencies.assert_called_once_with(deps)


# ---------------------------------------------------------------------------
# Scenario 2: Failed execution (non-zero exit code) sets error_status=True
# ---------------------------------------------------------------------------


class TestFailedExecution:
    """Scenario 2: A non-zero exit code sets error_status=True."""

    def test_error_status_true_on_nonzero_exit_code(self):
        """exit_code=1 sets error_status=True."""
        sandbox = _make_sandbox(stderr="NameError: name 'x' is not defined", exit_code=1)
        config = _make_config()
        node = make_execute_node(sandbox, config)

        with patch("coder_buddy.nodes.execute_node.log_node_event"):
            result = node(_make_state())

        assert result["error_status"] is True

    def test_error_status_true_on_exit_code_2(self):
        """exit_code=2 also sets error_status=True."""
        sandbox = _make_sandbox(stderr="SyntaxError: invalid syntax", exit_code=2)
        config = _make_config()
        node = make_execute_node(sandbox, config)

        with patch("coder_buddy.nodes.execute_node.log_node_event"):
            result = node(_make_state())

        assert result["error_status"] is True

    def test_execution_logs_contain_stderr_on_failure(self):
        """execution_logs contains the stderr from a failed run."""
        error_msg = "Traceback (most recent call last):\n  NameError: name 'x' is not defined"
        sandbox = _make_sandbox(stderr=error_msg, exit_code=1)
        config = _make_config()
        node = make_execute_node(sandbox, config)

        with patch("coder_buddy.nodes.execute_node.log_node_event"):
            result = node(_make_state())

        assert error_msg in result["execution_logs"]

    def test_execution_logs_contain_combined_output_on_failure(self):
        """execution_logs contains both stdout and stderr when both are present."""
        sandbox = _make_sandbox(stdout="partial output", stderr="RuntimeError: oops", exit_code=1)
        config = _make_config()
        node = make_execute_node(sandbox, config)

        with patch("coder_buddy.nodes.execute_node.log_node_event"):
            result = node(_make_state())

        assert "partial output" in result["execution_logs"]
        assert "RuntimeError: oops" in result["execution_logs"]


# ---------------------------------------------------------------------------
# Scenario 3: Timeout sets error_status=True with timeout message in logs
# ---------------------------------------------------------------------------


class TestTimeoutExecution:
    """Scenario 3: A timed-out execution sets error_status=True and includes the
    timeout notice in execution_logs."""

    def test_error_status_true_on_timeout(self):
        """timed_out=True sets error_status=True."""
        sandbox = _make_sandbox(exit_code=0, timed_out=True)
        config = _make_config()
        node = make_execute_node(sandbox, config)

        with patch("coder_buddy.nodes.execute_node.log_node_event"):
            result = node(_make_state())

        assert result["error_status"] is True

    def test_execution_logs_contain_timeout_message(self):
        """execution_logs contains the '[TIMEOUT: ...]' notice when timed_out=True."""
        sandbox = _make_sandbox(exit_code=0, timed_out=True)
        config = _make_config()
        node = make_execute_node(sandbox, config)

        with patch("coder_buddy.nodes.execute_node.log_node_event"):
            result = node(_make_state())

        assert "[TIMEOUT" in result["execution_logs"]

    def test_execution_logs_timeout_message_verbatim(self):
        """execution_logs contains the exact timeout string from ExecutionResult.combined_output."""
        sandbox = _make_sandbox(stdout="some output", exit_code=0, timed_out=True)
        config = _make_config()
        node = make_execute_node(sandbox, config)

        with patch("coder_buddy.nodes.execute_node.log_node_event"):
            result = node(_make_state())

        # The combined_output property appends this exact string when timed_out=True
        assert "[TIMEOUT: execution exceeded time limit]" in result["execution_logs"]

    def test_timeout_with_nonzero_exit_code_still_sets_error_status_true(self):
        """Timeout combined with non-zero exit code still sets error_status=True."""
        sandbox = _make_sandbox(stderr="killed", exit_code=137, timed_out=True)
        config = _make_config()
        node = make_execute_node(sandbox, config)

        with patch("coder_buddy.nodes.execute_node.log_node_event"):
            result = node(_make_state())

        assert result["error_status"] is True
        assert "[TIMEOUT" in result["execution_logs"]


# ---------------------------------------------------------------------------
# Scenario 4: cleanup always called, even on exception
# ---------------------------------------------------------------------------


class TestCleanupAlwaysCalled:
    """Scenario 4: sandbox.cleanup() is called in a finally block, so it runs
    even when sandbox.execute() raises an exception."""

    def test_cleanup_called_on_successful_execution(self):
        """cleanup() is called after a successful execution."""
        sandbox = _make_sandbox(exit_code=0)
        config = _make_config()
        node = make_execute_node(sandbox, config)

        with patch("coder_buddy.nodes.execute_node.log_node_event"):
            node(_make_state())

        sandbox.cleanup.assert_called_once()

    def test_cleanup_called_on_failed_execution(self):
        """cleanup() is called after a failed execution (non-zero exit code)."""
        sandbox = _make_sandbox(exit_code=1, stderr="error")
        config = _make_config()
        node = make_execute_node(sandbox, config)

        with patch("coder_buddy.nodes.execute_node.log_node_event"):
            node(_make_state())

        sandbox.cleanup.assert_called_once()

    def test_cleanup_called_on_timeout(self):
        """cleanup() is called after a timed-out execution."""
        sandbox = _make_sandbox(exit_code=0, timed_out=True)
        config = _make_config()
        node = make_execute_node(sandbox, config)

        with patch("coder_buddy.nodes.execute_node.log_node_event"):
            node(_make_state())

        sandbox.cleanup.assert_called_once()

    def test_cleanup_called_even_when_execute_raises(self):
        """cleanup() is called even when sandbox.execute() raises an exception."""
        sandbox = MagicMock()
        sandbox.execute.side_effect = RuntimeError("sandbox crashed")
        config = _make_config()
        node = make_execute_node(sandbox, config)

        with patch("coder_buddy.nodes.execute_node.log_node_event"):
            with pytest.raises(RuntimeError, match="sandbox crashed"):
                node(_make_state())

        sandbox.cleanup.assert_called_once()

    def test_cleanup_called_even_when_install_dependencies_raises(self):
        """cleanup() is called even when install_dependencies() raises an exception."""
        sandbox = MagicMock()
        sandbox.install_dependencies.side_effect = OSError("pip failed")
        config = _make_config()
        node = make_execute_node(sandbox, config)

        with patch("coder_buddy.nodes.execute_node.log_node_event"):
            with pytest.raises(OSError, match="pip failed"):
                node(_make_state(dependencies=["requests"]))

        sandbox.cleanup.assert_called_once()

    def test_cleanup_called_exactly_once_not_multiple_times(self):
        """cleanup() is called exactly once per node invocation, not multiple times."""
        sandbox = _make_sandbox(exit_code=0)
        config = _make_config()
        node = make_execute_node(sandbox, config)

        with patch("coder_buddy.nodes.execute_node.log_node_event"):
            node(_make_state())

        assert sandbox.cleanup.call_count == 1


# ---------------------------------------------------------------------------
# Property 5: for any non-empty dependencies list, install_dependencies is
#              called before execute
# ---------------------------------------------------------------------------

# Feature: coder-buddy, Property 5: for any non-empty dependencies list, install_dependencies is called before execute

from hypothesis import given, settings
from hypothesis import strategies as st


@given(
    dependencies=st.lists(
        st.text(
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="-_."),
            min_size=1,
            max_size=30,
        ),
        min_size=1,
        max_size=10,
    )
)
@settings(max_examples=100)
def test_property5_install_dependencies_called_before_execute(dependencies):
    """
    **Validates: Requirements 3.3**

    Property 5: For any non-empty `dependencies` list, `install_dependencies`
    is called before `execute` on the sandbox backend.
    """
    # Track the order in which methods are called
    call_order: list[str] = []

    result = ExecutionResult(stdout="ok", stderr="", exit_code=0, timed_out=False)

    sandbox = MagicMock()
    sandbox.install_dependencies.side_effect = lambda deps: call_order.append("install_dependencies")
    sandbox.execute.side_effect = lambda code, timeout: (call_order.append("execute"), result)[1]

    config = _make_config()
    node = make_execute_node(sandbox, config)

    with patch("coder_buddy.nodes.execute_node.log_node_event"):
        node(_make_state(dependencies=dependencies))

    # Both methods must have been called
    assert "install_dependencies" in call_order, "install_dependencies was not called"
    assert "execute" in call_order, "execute was not called"

    # install_dependencies must appear before execute in the call order
    install_idx = call_order.index("install_dependencies")
    execute_idx = call_order.index("execute")
    assert install_idx < execute_idx, (
        f"install_dependencies (position {install_idx}) was not called before "
        f"execute (position {execute_idx}); call_order={call_order}"
    )


# ---------------------------------------------------------------------------
# Property 6: for any sandbox outcome (success, failure, timeout),
#              execution_logs is populated and cleanup is called exactly once
# ---------------------------------------------------------------------------

# Feature: coder-buddy, Property 6: for any sandbox outcome (success, failure, timeout), execution_logs is populated and cleanup is called exactly once

# Strategy: generate one of three outcome types
_success_results = st.builds(
    ExecutionResult,
    stdout=st.text(min_size=1),  # non-empty stdout so combined_output is non-empty
    stderr=st.just(""),
    exit_code=st.just(0),
    timed_out=st.just(False),
)

_failure_results = st.builds(
    ExecutionResult,
    stdout=st.just(""),
    stderr=st.text(min_size=1),  # non-empty stderr so combined_output is non-empty
    exit_code=st.integers(min_value=1, max_value=255),
    timed_out=st.just(False),
)

_timeout_results = st.builds(
    ExecutionResult,
    stdout=st.text(),
    stderr=st.text(),
    exit_code=st.integers(min_value=0, max_value=255),
    timed_out=st.just(True),  # timed_out=True always appends the TIMEOUT notice
)

_any_outcome = st.one_of(_success_results, _failure_results, _timeout_results)


@given(execution_result=_any_outcome)
@settings(max_examples=100)
def test_property6_execution_logs_populated_and_cleanup_called_once(execution_result):
    """
    **Validates: Requirements 3.5 and 3.6**

    Property 6: For any sandbox outcome (success, failure, timeout),
    ``execution_logs`` is populated (non-empty) and ``sandbox.cleanup()``
    is called exactly once.
    """
    sandbox = MagicMock()
    sandbox.execute.return_value = execution_result

    config = _make_config()
    node = make_execute_node(sandbox, config)

    with patch("coder_buddy.nodes.execute_node.log_node_event"):
        result = node(_make_state())

    # execution_logs must be a non-empty string
    assert isinstance(result["execution_logs"], str), (
        f"execution_logs should be a str, got {type(result['execution_logs'])}"
    )
    assert len(result["execution_logs"]) > 0, (
        f"execution_logs must be non-empty for outcome: "
        f"stdout={execution_result.stdout!r}, stderr={execution_result.stderr!r}, "
        f"exit_code={execution_result.exit_code}, timed_out={execution_result.timed_out}"
    )

    # cleanup must be called exactly once
    assert sandbox.cleanup.call_count == 1, (
        f"cleanup() should be called exactly once, but was called "
        f"{sandbox.cleanup.call_count} time(s)"
    )
