"""
Unit tests for PyodideBackend.

Covers:
1. stdout capture — print() output is captured in ExecutionResult.stdout
2. stderr capture — exceptions/writes to sys.stderr appear in ExecutionResult.stderr
3. cleanup resets namespace — variables defined before cleanup are gone after it
4. Additional behaviours: health_check, install_dependencies, timeout, capture_output,
   idempotent cleanup, exit_code handling, combined_output, has_errors.

PyodideBackend runs code via exec() with io.StringIO redirection, so no real
Pyodide/WASM runtime is required — the tests exercise the backend directly in
CPython.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from coder_buddy.config import SandboxUnavailableError
from coder_buddy.sandbox.base import ExecutionResult
from coder_buddy.sandbox.pyodide_backend import PyodideBackend


# --------------------------------------------------------------------------- #
# 1. stdout capture
# --------------------------------------------------------------------------- #


class TestStdoutCapture:
    """execute() captures everything written to stdout."""

    def test_print_captured_in_stdout(self):
        """print() output appears in ExecutionResult.stdout."""
        backend = PyodideBackend()
        result = backend.execute("print('hello world')")

        assert "hello world" in result.stdout

    def test_multiple_print_lines_captured(self):
        """Multiple print() calls are all captured in stdout."""
        backend = PyodideBackend()
        result = backend.execute("print('line1')\nprint('line2')\nprint('line3')")

        assert "line1" in result.stdout
        assert "line2" in result.stdout
        assert "line3" in result.stdout

    def test_stdout_empty_when_no_output(self):
        """stdout is empty when the script produces no output."""
        backend = PyodideBackend()
        result = backend.execute("x = 1 + 1")

        assert result.stdout == ""

    def test_stdout_write_captured(self):
        """sys.stdout.write() is also captured."""
        backend = PyodideBackend()
        result = backend.execute("import sys; sys.stdout.write('direct write')")

        assert "direct write" in result.stdout

    def test_exit_code_zero_on_successful_stdout(self):
        """exit_code is 0 when code runs successfully and produces stdout."""
        backend = PyodideBackend()
        result = backend.execute("print('ok')")

        assert result.exit_code == 0

    def test_has_errors_false_on_successful_stdout(self):
        """has_errors is False when code runs successfully."""
        backend = PyodideBackend()
        result = backend.execute("print('ok')")

        assert result.has_errors is False

    def test_timed_out_false_on_successful_stdout(self):
        """timed_out is False for a normal (non-timeout) execution."""
        backend = PyodideBackend()
        result = backend.execute("print('ok')")

        assert result.timed_out is False

    def test_stdout_not_leaked_to_real_stdout(self, capsys):
        """Output captured by the backend does not appear on the real stdout."""
        backend = PyodideBackend()
        backend.execute("print('should be captured')")

        captured = capsys.readouterr()
        assert "should be captured" not in captured.out


# --------------------------------------------------------------------------- #
# 2. stderr capture
# --------------------------------------------------------------------------- #


class TestStderrCapture:
    """execute() captures exceptions and sys.stderr writes in ExecutionResult.stderr."""

    def test_exception_traceback_in_stderr(self):
        """An unhandled exception's traceback appears in stderr."""
        backend = PyodideBackend()
        result = backend.execute("raise ValueError('test error')")

        assert "ValueError" in result.stderr
        assert "test error" in result.stderr

    def test_name_error_in_stderr(self):
        """A NameError traceback is captured in stderr."""
        backend = PyodideBackend()
        result = backend.execute("print(undefined_variable)")

        assert "NameError" in result.stderr

    def test_syntax_error_in_stderr(self):
        """A SyntaxError is captured in stderr."""
        backend = PyodideBackend()
        result = backend.execute("def broken(:\n    pass")

        assert result.exit_code != 0

    def test_sys_stderr_write_captured(self):
        """sys.stderr.write() output is captured in stderr."""
        backend = PyodideBackend()
        result = backend.execute("import sys; sys.stderr.write('err output')")

        assert "err output" in result.stderr

    def test_exit_code_nonzero_on_exception(self):
        """exit_code is non-zero when an exception is raised."""
        backend = PyodideBackend()
        result = backend.execute("raise RuntimeError('boom')")

        assert result.exit_code != 0

    def test_has_errors_true_on_exception(self):
        """has_errors is True when an exception is raised."""
        backend = PyodideBackend()
        result = backend.execute("raise RuntimeError('boom')")

        assert result.has_errors is True

    def test_stderr_empty_on_clean_execution(self):
        """stderr is empty when the script runs without errors."""
        backend = PyodideBackend()
        result = backend.execute("x = 42")

        assert result.stderr == ""

    def test_stdout_still_captured_alongside_stderr(self):
        """stdout output before an exception is still captured."""
        backend = PyodideBackend()
        result = backend.execute("print('before error')\nraise ValueError('after')")

        assert "before error" in result.stdout
        assert "ValueError" in result.stderr

    def test_stderr_not_leaked_to_real_stderr(self, capsys):
        """Stderr captured by the backend does not appear on the real stderr."""
        backend = PyodideBackend()
        backend.execute("raise ValueError('should be captured')")

        captured = capsys.readouterr()
        assert "should be captured" not in captured.err


# --------------------------------------------------------------------------- #
# 3. cleanup resets namespace
# --------------------------------------------------------------------------- #


class TestCleanupResetsNamespace:
    """cleanup() resets the execution namespace so prior state is gone."""

    def test_variable_defined_before_cleanup_is_gone_after(self):
        """A variable set in one execution is not accessible after cleanup()."""
        backend = PyodideBackend()

        # Define a variable in the namespace
        backend.execute("my_var = 42")
        assert "my_var" in backend._namespace

        backend.cleanup()

        # Namespace must be empty after cleanup
        assert "my_var" not in backend._namespace

    def test_namespace_is_empty_dict_after_cleanup(self):
        """After cleanup(), _namespace is a fresh empty dict."""
        backend = PyodideBackend()
        backend.execute("x = 1; y = 2; z = 3")

        backend.cleanup()

        assert backend._namespace == {}

    def test_execution_after_cleanup_starts_fresh(self):
        """Code executed after cleanup() cannot see variables from before cleanup."""
        backend = PyodideBackend()

        # Set a variable, then clean up
        backend.execute("secret = 'hidden'")
        backend.cleanup()

        # Trying to access the old variable should raise NameError
        result = backend.execute("print(secret)")

        assert result.exit_code != 0
        assert "NameError" in result.stderr

    def test_cleanup_clears_last_result(self):
        """After cleanup(), capture_output() returns an empty string."""
        backend = PyodideBackend()
        backend.execute("print('something')")
        assert backend.capture_output() != ""

        backend.cleanup()

        assert backend.capture_output() == ""

    def test_cleanup_is_idempotent(self):
        """Calling cleanup() multiple times does not raise."""
        backend = PyodideBackend()
        backend.execute("x = 1")

        backend.cleanup()
        backend.cleanup()  # second call — should be a no-op

        assert backend._namespace == {}

    def test_cleanup_on_fresh_backend_does_not_raise(self):
        """cleanup() on a brand-new backend (no prior execution) does not raise."""
        backend = PyodideBackend()
        # Should not raise
        backend.cleanup()

    def test_namespace_shared_across_executions_before_cleanup(self):
        """Variables persist across executions within the same session (before cleanup)."""
        backend = PyodideBackend()

        backend.execute("counter = 0")
        backend.execute("counter += 1")
        result = backend.execute("print(counter)")

        assert "1" in result.stdout

    def test_new_execution_after_cleanup_succeeds(self):
        """A fresh execution after cleanup() works correctly."""
        backend = PyodideBackend()
        backend.execute("x = 99")
        backend.cleanup()

        result = backend.execute("print('fresh start')")

        assert "fresh start" in result.stdout
        assert result.exit_code == 0


# --------------------------------------------------------------------------- #
# 4. health_check
# --------------------------------------------------------------------------- #


class TestHealthCheck:
    """health_check() raises SandboxUnavailableError when pyodide is unavailable."""

    def test_health_check_raises_when_pyodide_not_installed(self):
        """SandboxUnavailableError is raised when _PYODIDE_AVAILABLE is False."""
        with patch("coder_buddy.sandbox.pyodide_backend._PYODIDE_AVAILABLE", False):
            backend = PyodideBackend()
            with pytest.raises(SandboxUnavailableError, match="pyodide"):
                backend.health_check()

    def test_health_check_passes_when_pyodide_available(self):
        """health_check() returns None when pyodide is importable with a version."""
        mock_pyodide = MagicMock()
        mock_pyodide.__version__ = "0.24.0"

        with (
            patch("coder_buddy.sandbox.pyodide_backend._PYODIDE_AVAILABLE", True),
            patch("coder_buddy.sandbox.pyodide_backend.pyodide", mock_pyodide),
        ):
            backend = PyodideBackend()
            # Should not raise
            backend.health_check()

    def test_health_check_raises_when_version_missing(self):
        """SandboxUnavailableError is raised when pyodide has no __version__."""
        mock_pyodide = MagicMock(spec=[])  # no attributes

        with (
            patch("coder_buddy.sandbox.pyodide_backend._PYODIDE_AVAILABLE", True),
            patch("coder_buddy.sandbox.pyodide_backend.pyodide", mock_pyodide),
        ):
            backend = PyodideBackend()
            with pytest.raises(SandboxUnavailableError):
                backend.health_check()


# --------------------------------------------------------------------------- #
# 5. install_dependencies
# --------------------------------------------------------------------------- #


class TestInstallDependencies:
    """install_dependencies() delegates to micropip when available."""

    def test_empty_deps_is_noop(self):
        """install_dependencies([]) returns without calling micropip."""
        backend = PyodideBackend()
        # Should not raise even without micropip
        backend.install_dependencies([])

    def test_missing_micropip_raises_sandbox_unavailable(self):
        """SandboxUnavailableError is raised when micropip is not available."""
        with patch("coder_buddy.sandbox.pyodide_backend._MICROPIP_AVAILABLE", False):
            backend = PyodideBackend()
            with pytest.raises(SandboxUnavailableError, match="micropip"):
                backend.install_dependencies(["requests"])

    def test_micropip_install_called_with_deps(self):
        """micropip.install() is called with the dependency list."""
        mock_micropip = MagicMock()
        mock_micropip.install.return_value = None  # synchronous return

        with (
            patch("coder_buddy.sandbox.pyodide_backend._MICROPIP_AVAILABLE", True),
            patch("coder_buddy.sandbox.pyodide_backend.micropip", mock_micropip),
        ):
            backend = PyodideBackend()
            backend.install_dependencies(["requests", "numpy"])

        mock_micropip.install.assert_called_once_with(["requests", "numpy"])

    def test_micropip_failure_raises_sandbox_unavailable(self):
        """Exception from micropip.install() → SandboxUnavailableError."""
        mock_micropip = MagicMock()
        mock_micropip.install.side_effect = Exception("package not found")

        with (
            patch("coder_buddy.sandbox.pyodide_backend._MICROPIP_AVAILABLE", True),
            patch("coder_buddy.sandbox.pyodide_backend.micropip", mock_micropip),
        ):
            backend = PyodideBackend()
            with pytest.raises(SandboxUnavailableError, match="package not found"):
                backend.install_dependencies(["nonexistent-pkg"])


# --------------------------------------------------------------------------- #
# 6. Timeout handling
# --------------------------------------------------------------------------- #


class TestTimeoutHandling:
    """execute() sets timed_out=True when the execution thread does not finish in time."""

    def test_timed_out_true_for_infinite_loop(self):
        """timed_out=True when code runs longer than the timeout."""
        backend = PyodideBackend()
        result = backend.execute("while True: pass", timeout=0.05)

        assert result.timed_out is True

    def test_exit_code_negative_one_on_timeout(self):
        """exit_code is -1 when execution timed out."""
        backend = PyodideBackend()
        result = backend.execute("while True: pass", timeout=0.05)

        assert result.exit_code == -1

    def test_has_errors_true_on_timeout(self):
        """has_errors is True when execution timed out."""
        backend = PyodideBackend()
        result = backend.execute("while True: pass", timeout=0.05)

        assert result.has_errors is True

    def test_combined_output_contains_timeout_notice(self):
        """combined_output includes the TIMEOUT notice when timed_out=True."""
        backend = PyodideBackend()
        result = backend.execute("while True: pass", timeout=0.05)

        assert "TIMEOUT" in result.combined_output


# --------------------------------------------------------------------------- #
# 7. capture_output
# --------------------------------------------------------------------------- #


class TestCaptureOutput:
    """capture_output() returns combined stdout+stderr from the last execution."""

    def test_capture_output_empty_before_any_execution(self):
        """capture_output() returns '' when no execution has occurred."""
        backend = PyodideBackend()
        assert backend.capture_output() == ""

    def test_capture_output_returns_stdout(self):
        """capture_output() includes stdout from the last execution."""
        backend = PyodideBackend()
        backend.execute("print('captured')")

        assert "captured" in backend.capture_output()

    def test_capture_output_returns_stderr(self):
        """capture_output() includes stderr from the last execution."""
        backend = PyodideBackend()
        backend.execute("raise ValueError('err')")

        assert "ValueError" in backend.capture_output()

    def test_capture_output_reflects_last_execution_only(self):
        """capture_output() reflects the most recent execution, not earlier ones."""
        backend = PyodideBackend()
        backend.execute("print('first')")
        backend.execute("print('second')")

        output = backend.capture_output()
        assert "second" in output

    def test_capture_output_empty_after_cleanup(self):
        """capture_output() returns '' after cleanup()."""
        backend = PyodideBackend()
        backend.execute("print('something')")
        backend.cleanup()

        assert backend.capture_output() == ""


# --------------------------------------------------------------------------- #
# 8. SystemExit handling
# --------------------------------------------------------------------------- #


class TestSystemExitHandling:
    """execute() handles sys.exit() calls gracefully."""

    def test_sys_exit_zero_gives_exit_code_zero(self):
        """sys.exit(0) results in exit_code == 0."""
        backend = PyodideBackend()
        result = backend.execute("import sys; sys.exit(0)")

        assert result.exit_code == 0
        assert result.has_errors is False

    def test_sys_exit_nonzero_gives_nonzero_exit_code(self):
        """sys.exit(1) results in exit_code == 1."""
        backend = PyodideBackend()
        result = backend.execute("import sys; sys.exit(1)")

        assert result.exit_code == 1
        assert result.has_errors is True

    def test_sys_exit_none_gives_exit_code_zero(self):
        """sys.exit() with no argument (None) results in exit_code == 0."""
        backend = PyodideBackend()
        result = backend.execute("import sys; sys.exit()")

        assert result.exit_code == 0
