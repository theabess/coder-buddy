"""
Unit tests for E2BBackend.

Covers:
1. Missing API key raises SandboxUnavailableError in health_check()
2. cleanup() calls sandbox.close()
3. Additional behaviours: e2b package unavailable, sandbox creation failure,
   execute() and install_dependencies() when sandbox is uninitialised,
   cleanup() is idempotent, capture_output() returns combined output.

All tests mock the e2b SDK so no real E2B account or network is required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from coder_buddy.config import SandboxUnavailableError
from coder_buddy.sandbox.base import ExecutionResult
from coder_buddy.sandbox.e2b_backend import E2BBackend


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_mock_sandbox(
    process_result=None,
    process_side_effect=None,
):
    """Return a mock E2B Sandbox object with configurable process behaviour."""
    sandbox = MagicMock()
    if process_side_effect is not None:
        sandbox.process.start_and_wait.side_effect = process_side_effect
    elif process_result is not None:
        sandbox.process.start_and_wait.return_value = process_result
    else:
        # Default: successful process with empty output
        default_proc = MagicMock()
        default_proc.stdout = ""
        default_proc.stderr = ""
        default_proc.exit_code = 0
        sandbox.process.start_and_wait.return_value = default_proc
    return sandbox


def _make_process_result(stdout="", stderr="", exit_code=0):
    """Return a mock process result object."""
    proc = MagicMock()
    proc.stdout = stdout
    proc.stderr = stderr
    proc.exit_code = exit_code
    return proc


# --------------------------------------------------------------------------- #
# 1. Missing API key raises SandboxUnavailableError
# --------------------------------------------------------------------------- #


class TestHealthCheckMissingApiKey:
    """health_check() raises SandboxUnavailableError when E2B_API_KEY is absent."""

    def test_missing_api_key_raises_sandbox_unavailable(self, monkeypatch):
        """No E2B_API_KEY env var → SandboxUnavailableError."""
        monkeypatch.delenv("E2B_API_KEY", raising=False)

        with patch("coder_buddy.sandbox.e2b_backend._E2B_AVAILABLE", True):
            backend = E2BBackend()
            with pytest.raises(SandboxUnavailableError):
                backend.health_check()

    def test_missing_api_key_error_message(self, monkeypatch):
        """SandboxUnavailableError message mentions E2B_API_KEY."""
        monkeypatch.delenv("E2B_API_KEY", raising=False)

        with patch("coder_buddy.sandbox.e2b_backend._E2B_AVAILABLE", True):
            backend = E2BBackend()
            with pytest.raises(SandboxUnavailableError, match="E2B_API_KEY"):
                backend.health_check()

    def test_empty_api_key_raises_sandbox_unavailable(self, monkeypatch):
        """Empty string E2B_API_KEY is treated as missing → SandboxUnavailableError."""
        monkeypatch.setenv("E2B_API_KEY", "")

        with patch("coder_buddy.sandbox.e2b_backend._E2B_AVAILABLE", True):
            backend = E2BBackend()
            with pytest.raises(SandboxUnavailableError):
                backend.health_check()

    def test_e2b_package_unavailable_raises_sandbox_unavailable(self, monkeypatch):
        """When the e2b package is not installed → SandboxUnavailableError."""
        monkeypatch.setenv("E2B_API_KEY", "test-key")

        with patch("coder_buddy.sandbox.e2b_backend._E2B_AVAILABLE", False):
            backend = E2BBackend()
            with pytest.raises(SandboxUnavailableError, match="e2b"):
                backend.health_check()

    def test_sandbox_creation_failure_raises_sandbox_unavailable(self, monkeypatch):
        """Exception from Sandbox.create() → SandboxUnavailableError."""
        monkeypatch.setenv("E2B_API_KEY", "test-key")

        mock_sandbox_cls = MagicMock()
        mock_sandbox_cls.create.side_effect = Exception("network error")

        with (
            patch("coder_buddy.sandbox.e2b_backend._E2B_AVAILABLE", True),
            patch("coder_buddy.sandbox.e2b_backend.Sandbox", mock_sandbox_cls),
        ):
            backend = E2BBackend()
            with pytest.raises(SandboxUnavailableError, match="network error"):
                backend.health_check()

    def test_health_check_passes_with_valid_api_key(self, monkeypatch):
        """health_check() succeeds when API key is set and Sandbox.create() works."""
        monkeypatch.setenv("E2B_API_KEY", "test-key")

        mock_sandbox_instance = MagicMock()
        mock_sandbox_cls = MagicMock()
        mock_sandbox_cls.create.return_value = mock_sandbox_instance

        with (
            patch("coder_buddy.sandbox.e2b_backend._E2B_AVAILABLE", True),
            patch("coder_buddy.sandbox.e2b_backend.Sandbox", mock_sandbox_cls),
        ):
            backend = E2BBackend()
            # Should not raise
            backend.health_check()

        mock_sandbox_cls.create.assert_called_once_with(timeout=5)
        mock_sandbox_instance.close.assert_called_once()


# --------------------------------------------------------------------------- #
# 2. cleanup() calls sandbox.close()
# --------------------------------------------------------------------------- #


class TestCleanupCallsSandboxClose:
    """cleanup() calls sandbox.close() and clears internal state."""

    def _make_backend_with_sandbox(self):
        """Return an E2BBackend with a mock sandbox already attached."""
        backend = E2BBackend()
        mock_sandbox = _make_mock_sandbox()
        backend._sandbox = mock_sandbox
        return backend, mock_sandbox

    def test_cleanup_calls_sandbox_close(self):
        """sandbox.close() is called during cleanup()."""
        backend, mock_sandbox = self._make_backend_with_sandbox()

        backend.cleanup()

        mock_sandbox.close.assert_called_once()

    def test_cleanup_sets_sandbox_to_none(self):
        """After cleanup(), _sandbox is None."""
        backend, _ = self._make_backend_with_sandbox()

        backend.cleanup()

        assert backend._sandbox is None

    def test_cleanup_clears_last_result(self):
        """After cleanup(), _last_result is None."""
        backend, _ = self._make_backend_with_sandbox()
        backend._last_result = ExecutionResult(
            stdout="out", stderr="", exit_code=0, timed_out=False
        )

        backend.cleanup()

        assert backend._last_result is None

    def test_cleanup_is_idempotent(self):
        """Calling cleanup() twice does not raise and close() is called only once."""
        backend, mock_sandbox = self._make_backend_with_sandbox()

        backend.cleanup()
        backend.cleanup()  # second call — sandbox is already None

        mock_sandbox.close.assert_called_once()

    def test_cleanup_tolerates_close_exception(self):
        """cleanup() does not propagate exceptions from sandbox.close()."""
        backend, mock_sandbox = self._make_backend_with_sandbox()
        mock_sandbox.close.side_effect = Exception("already closed")

        # Should not raise
        backend.cleanup()

        assert backend._sandbox is None

    def test_cleanup_with_no_sandbox_does_not_raise(self):
        """cleanup() on a fresh backend (no sandbox) does not raise."""
        backend = E2BBackend()
        # Should not raise
        backend.cleanup()


# --------------------------------------------------------------------------- #
# 3. execute() and install_dependencies() without initialised sandbox
# --------------------------------------------------------------------------- #


class TestUninitialised:
    """execute() and install_dependencies() raise when sandbox is not set."""

    def test_execute_without_sandbox_raises(self):
        """execute() raises SandboxUnavailableError when _sandbox is None."""
        backend = E2BBackend()
        with pytest.raises(SandboxUnavailableError):
            backend.execute("print('hello')")

    def test_install_dependencies_without_sandbox_raises(self):
        """install_dependencies() raises SandboxUnavailableError when _sandbox is None."""
        backend = E2BBackend()
        with pytest.raises(SandboxUnavailableError):
            backend.install_dependencies(["requests"])

    def test_install_dependencies_empty_list_does_not_raise(self):
        """install_dependencies([]) is a no-op even without a sandbox."""
        backend = E2BBackend()
        # Should not raise — empty list is a no-op
        backend.install_dependencies([])


# --------------------------------------------------------------------------- #
# 4. execute() behaviour with a mock sandbox
# --------------------------------------------------------------------------- #


class TestExecute:
    """execute() correctly maps process output to ExecutionResult."""

    def _make_backend_with_sandbox(self, process_result=None, process_side_effect=None):
        backend = E2BBackend()
        mock_sandbox = _make_mock_sandbox(
            process_result=process_result,
            process_side_effect=process_side_effect,
        )
        backend._sandbox = mock_sandbox
        return backend, mock_sandbox

    def test_execute_returns_execution_result(self):
        """execute() returns an ExecutionResult instance."""
        proc = _make_process_result(stdout="hello\n", stderr="", exit_code=0)
        backend, _ = self._make_backend_with_sandbox(process_result=proc)

        result = backend.execute("print('hello')")

        assert isinstance(result, ExecutionResult)

    def test_execute_captures_stdout(self):
        """execute() populates stdout from process output."""
        proc = _make_process_result(stdout="hello\n", stderr="", exit_code=0)
        backend, _ = self._make_backend_with_sandbox(process_result=proc)

        result = backend.execute("print('hello')")

        assert result.stdout == "hello\n"

    def test_execute_captures_stderr(self):
        """execute() populates stderr from process output."""
        proc = _make_process_result(stdout="", stderr="error!\n", exit_code=1)
        backend, _ = self._make_backend_with_sandbox(process_result=proc)

        result = backend.execute("raise ValueError('error!')")

        assert result.stderr == "error!\n"
        assert result.exit_code == 1

    def test_execute_timeout_sets_timed_out_true(self):
        """execute() sets timed_out=True when TimeoutError is raised."""
        backend, _ = self._make_backend_with_sandbox(
            process_side_effect=TimeoutError("timed out")
        )

        result = backend.execute("import time; time.sleep(100)", timeout=0.01)

        assert result.timed_out is True
        assert result.exit_code == -1

    def test_execute_writes_script_to_filesystem(self):
        """execute() writes source_code to /home/user/script.py."""
        proc = _make_process_result()
        backend, mock_sandbox = self._make_backend_with_sandbox(process_result=proc)

        backend.execute("print('test')")

        mock_sandbox.filesystem.write.assert_called_once_with(
            "/home/user/script.py", "print('test')"
        )

    def test_execute_stores_last_result(self):
        """execute() stores the result in _last_result."""
        proc = _make_process_result(stdout="out", stderr="", exit_code=0)
        backend, _ = self._make_backend_with_sandbox(process_result=proc)

        result = backend.execute("print('out')")

        assert backend._last_result is result


# --------------------------------------------------------------------------- #
# 5. capture_output()
# --------------------------------------------------------------------------- #


class TestCaptureOutput:
    """capture_output() returns combined stdout+stderr from the last execution."""

    def test_capture_output_returns_empty_when_no_execution(self):
        """capture_output() returns '' when no execution has occurred."""
        backend = E2BBackend()
        assert backend.capture_output() == ""

    def test_capture_output_returns_combined_output(self):
        """capture_output() returns combined stdout and stderr."""
        backend = E2BBackend()
        backend._last_result = ExecutionResult(
            stdout="hello\n", stderr="warn\n", exit_code=0, timed_out=False
        )

        output = backend.capture_output()

        assert "hello\n" in output
        assert "warn\n" in output
