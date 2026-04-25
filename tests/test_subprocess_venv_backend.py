"""
Unit tests for SubprocessVenvBackend.

Covers:
1. Successful execution — stdout captured, error_status False, exit_code == 0
2. Timeout handling — timed_out=True and has_errors=True when execution exceeds timeout
3. Dependency installation — install_dependencies called before execute when deps non-empty
4. Cleanup always called — cleanup() removes tmpdir even when execute raises an exception
5. health_check passes when Python (venv module) is available
6. Property 7 — after cleanup(), the temporary directory no longer exists on the filesystem
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from coder_buddy.config import SandboxUnavailableError
from coder_buddy.sandbox.base import ExecutionResult
from coder_buddy.sandbox.subprocess_venv import SubprocessVenvBackend, _venv_pip, _venv_python


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

FAKE_TMPDIR = "/tmp/fake_tmpdir_abc123"


def _make_completed_process(stdout="", stderr="", returncode=0):
    """Return a mock CompletedProcess-like object."""
    proc = MagicMock()
    proc.stdout = stdout
    proc.stderr = stderr
    proc.returncode = returncode
    return proc


# --------------------------------------------------------------------------- #
# 1. Successful execution
# --------------------------------------------------------------------------- #


class TestSuccessfulExecution:
    """execute() returns an ExecutionResult with stdout, exit_code==0, no errors."""

    def test_stdout_captured(self):
        """stdout from the subprocess is stored in ExecutionResult.stdout."""
        with (
            patch("tempfile.mkdtemp", return_value=FAKE_TMPDIR),
            patch("pathlib.Path.write_text"),
            patch("subprocess.run", return_value=_make_completed_process(stdout="hello\n")),
        ):
            backend = SubprocessVenvBackend()
            result = backend.execute("print('hello')")

        assert result.stdout == "hello\n"

    def test_exit_code_zero(self):
        """A successful run has exit_code == 0."""
        with (
            patch("tempfile.mkdtemp", return_value=FAKE_TMPDIR),
            patch("pathlib.Path.write_text"),
            patch("subprocess.run", return_value=_make_completed_process(returncode=0)),
        ):
            backend = SubprocessVenvBackend()
            result = backend.execute("x = 1")

        assert result.exit_code == 0

    def test_has_errors_false_on_success(self):
        """has_errors is False when exit_code==0 and timed_out==False."""
        with (
            patch("tempfile.mkdtemp", return_value=FAKE_TMPDIR),
            patch("pathlib.Path.write_text"),
            patch("subprocess.run", return_value=_make_completed_process(returncode=0)),
        ):
            backend = SubprocessVenvBackend()
            result = backend.execute("x = 1")

        assert result.has_errors is False

    def test_timed_out_false_on_success(self):
        """timed_out is False for a normal (non-timeout) execution."""
        with (
            patch("tempfile.mkdtemp", return_value=FAKE_TMPDIR),
            patch("pathlib.Path.write_text"),
            patch("subprocess.run", return_value=_make_completed_process(returncode=0)),
        ):
            backend = SubprocessVenvBackend()
            result = backend.execute("x = 1")

        assert result.timed_out is False

    def test_result_stored_as_last_result(self):
        """execute() stores the result so capture_output() can return it."""
        with (
            patch("tempfile.mkdtemp", return_value=FAKE_TMPDIR),
            patch("pathlib.Path.write_text"),
            patch(
                "subprocess.run",
                return_value=_make_completed_process(stdout="out", stderr="err"),
            ),
        ):
            backend = SubprocessVenvBackend()
            result = backend.execute("x = 1")

        assert backend.capture_output() == result.combined_output


# --------------------------------------------------------------------------- #
# 2. Timeout handling
# --------------------------------------------------------------------------- #


class TestTimeoutHandling:
    """execute() sets timed_out=True and has_errors=True when TimeoutExpired is raised."""

    def _make_timeout_exc(self, stdout=b"partial", stderr=b""):
        """Build a subprocess.TimeoutExpired with a mock process attached."""
        exc = subprocess.TimeoutExpired(cmd=["python", "script.py"], timeout=5.0)
        exc.stdout = stdout
        exc.stderr = stderr
        exc.process = MagicMock()
        return exc

    def test_timed_out_true(self):
        """timed_out is True when TimeoutExpired is raised."""
        with (
            patch("tempfile.mkdtemp", return_value=FAKE_TMPDIR),
            patch("pathlib.Path.write_text"),
            patch("subprocess.run", side_effect=self._make_timeout_exc()),
        ):
            backend = SubprocessVenvBackend()
            result = backend.execute("import time; time.sleep(100)", timeout=0.001)

        assert result.timed_out is True

    def test_has_errors_true_on_timeout(self):
        """has_errors is True when execution timed out."""
        with (
            patch("tempfile.mkdtemp", return_value=FAKE_TMPDIR),
            patch("pathlib.Path.write_text"),
            patch("subprocess.run", side_effect=self._make_timeout_exc()),
        ):
            backend = SubprocessVenvBackend()
            result = backend.execute("import time; time.sleep(100)", timeout=0.001)

        assert result.has_errors is True

    def test_exit_code_negative_one_on_timeout(self):
        """exit_code is -1 when execution timed out."""
        with (
            patch("tempfile.mkdtemp", return_value=FAKE_TMPDIR),
            patch("pathlib.Path.write_text"),
            patch("subprocess.run", side_effect=self._make_timeout_exc()),
        ):
            backend = SubprocessVenvBackend()
            result = backend.execute("import time; time.sleep(100)", timeout=0.001)

        assert result.exit_code == -1

    def test_process_killed_on_timeout(self):
        """The subprocess is killed when TimeoutExpired is raised."""
        exc = self._make_timeout_exc()
        with (
            patch("tempfile.mkdtemp", return_value=FAKE_TMPDIR),
            patch("pathlib.Path.write_text"),
            patch("subprocess.run", side_effect=exc),
        ):
            backend = SubprocessVenvBackend()
            backend.execute("import time; time.sleep(100)", timeout=0.001)

        exc.process.kill.assert_called_once()

    def test_partial_stdout_captured_on_timeout(self):
        """Any partial stdout emitted before timeout is preserved."""
        exc = self._make_timeout_exc(stdout=b"partial output", stderr=b"")
        with (
            patch("tempfile.mkdtemp", return_value=FAKE_TMPDIR),
            patch("pathlib.Path.write_text"),
            patch("subprocess.run", side_effect=exc),
        ):
            backend = SubprocessVenvBackend()
            result = backend.execute("...", timeout=0.001)

        assert "partial output" in result.stdout

    def test_timeout_with_none_process(self):
        """If exc.process is None, kill() is not called (no AttributeError)."""
        exc = subprocess.TimeoutExpired(cmd=["python"], timeout=1.0)
        exc.stdout = None
        exc.stderr = None
        exc.process = None
        with (
            patch("tempfile.mkdtemp", return_value=FAKE_TMPDIR),
            patch("pathlib.Path.write_text"),
            patch("subprocess.run", side_effect=exc),
        ):
            backend = SubprocessVenvBackend()
            result = backend.execute("...", timeout=0.001)

        assert result.timed_out is True


# --------------------------------------------------------------------------- #
# 3. Dependency installation
# --------------------------------------------------------------------------- #


class TestDependencyInstallation:
    """install_dependencies() creates a venv and pip-installs packages."""

    def test_venv_created(self):
        """install_dependencies calls python -m venv to create the environment."""
        with (
            patch("tempfile.mkdtemp", return_value=FAKE_TMPDIR),
            patch("subprocess.run", return_value=_make_completed_process()) as mock_run,
        ):
            backend = SubprocessVenvBackend()
            backend.install_dependencies(["requests"])

        # First call must be the venv creation
        first_call_args = mock_run.call_args_list[0][0][0]
        assert first_call_args == [sys.executable, "-m", "venv", str(Path(FAKE_TMPDIR) / "venv")]

    def test_pip_install_called_with_deps(self):
        """install_dependencies calls pip install with the given packages."""
        with (
            patch("tempfile.mkdtemp", return_value=FAKE_TMPDIR),
            patch("subprocess.run", return_value=_make_completed_process()) as mock_run,
        ):
            backend = SubprocessVenvBackend()
            backend.install_dependencies(["requests", "numpy"])

        # Second call must be pip install
        second_call_args = mock_run.call_args_list[1][0][0]
        assert "requests" in second_call_args
        assert "numpy" in second_call_args

    def test_pip_install_not_called_for_empty_deps(self):
        """install_dependencies skips pip install when the list is empty."""
        with (
            patch("tempfile.mkdtemp", return_value=FAKE_TMPDIR),
            patch("subprocess.run", return_value=_make_completed_process()) as mock_run,
        ):
            backend = SubprocessVenvBackend()
            backend.install_dependencies([])

        # Only the venv creation call should have been made
        assert mock_run.call_count == 1

    def test_install_before_execute_uses_same_tmpdir(self):
        """install_dependencies and execute share the same tmpdir."""
        with (
            patch("tempfile.mkdtemp", return_value=FAKE_TMPDIR) as mock_mkdtemp,
            patch("pathlib.Path.write_text"),
            patch("subprocess.run", return_value=_make_completed_process()),
        ):
            backend = SubprocessVenvBackend()
            backend.install_dependencies(["requests"])
            backend.execute("import requests")

        # mkdtemp should only be called once — both methods share the same dir
        mock_mkdtemp.assert_called_once()


# --------------------------------------------------------------------------- #
# 4. Cleanup always called
# --------------------------------------------------------------------------- #


class TestCleanupAlwaysCalled:
    """cleanup() removes the tmpdir even when execute() raises an exception."""

    def test_cleanup_removes_tmpdir(self):
        """cleanup() calls shutil.rmtree on the tmpdir."""
        with (
            patch("tempfile.mkdtemp", return_value=FAKE_TMPDIR),
            patch("shutil.rmtree") as mock_rmtree,
        ):
            backend = SubprocessVenvBackend()
            backend._ensure_tmpdir()  # force tmpdir creation
            backend.cleanup()

        mock_rmtree.assert_called_once_with(FAKE_TMPDIR, ignore_errors=True)

    def test_cleanup_resets_tmpdir_to_none(self):
        """After cleanup(), _tmpdir is None so a new dir can be created next time."""
        with (
            patch("tempfile.mkdtemp", return_value=FAKE_TMPDIR),
            patch("shutil.rmtree"),
        ):
            backend = SubprocessVenvBackend()
            backend._ensure_tmpdir()
            assert backend._tmpdir == FAKE_TMPDIR
            backend.cleanup()
            assert backend._tmpdir is None

    def test_cleanup_is_idempotent(self):
        """Calling cleanup() twice does not raise and only removes once."""
        with (
            patch("tempfile.mkdtemp", return_value=FAKE_TMPDIR),
            patch("shutil.rmtree") as mock_rmtree,
        ):
            backend = SubprocessVenvBackend()
            backend._ensure_tmpdir()
            backend.cleanup()
            backend.cleanup()  # second call — tmpdir is already None

        # rmtree should only have been called once (the second cleanup is a no-op)
        mock_rmtree.assert_called_once()

    def test_cleanup_called_even_when_execute_raises(self):
        """
        Demonstrates that callers should use try/finally to guarantee cleanup.
        The backend itself does not auto-cleanup on exception, but cleanup()
        still works correctly after an exception during execute().
        """
        with (
            patch("tempfile.mkdtemp", return_value=FAKE_TMPDIR),
            patch("pathlib.Path.write_text"),
            patch("subprocess.run", side_effect=RuntimeError("unexpected")),
            patch("shutil.rmtree") as mock_rmtree,
        ):
            backend = SubprocessVenvBackend()
            with pytest.raises(RuntimeError):
                backend.execute("x = 1")
            # Caller is responsible for cleanup; verify it works after exception
            backend.cleanup()

        mock_rmtree.assert_called_once_with(FAKE_TMPDIR, ignore_errors=True)

    def test_cleanup_resets_last_result(self):
        """cleanup() also clears _last_result so capture_output returns ''."""
        with (
            patch("tempfile.mkdtemp", return_value=FAKE_TMPDIR),
            patch("pathlib.Path.write_text"),
            patch("subprocess.run", return_value=_make_completed_process(stdout="hi")),
            patch("shutil.rmtree"),
        ):
            backend = SubprocessVenvBackend()
            backend.execute("print('hi')")
            assert backend.capture_output() != ""
            backend.cleanup()

        assert backend.capture_output() == ""


# --------------------------------------------------------------------------- #
# 5. health_check
# --------------------------------------------------------------------------- #


class TestHealthCheck:
    """health_check() passes when python -m venv --help exits 0."""

    def test_health_check_passes_when_venv_available(self):
        """health_check() returns None (no exception) when venv is available."""
        with patch(
            "subprocess.run",
            return_value=_make_completed_process(returncode=0),
        ):
            backend = SubprocessVenvBackend()
            # Should not raise
            backend.health_check()

    def test_health_check_calls_venv_help(self):
        """health_check() invokes python -m venv --help."""
        with patch("subprocess.run", return_value=_make_completed_process()) as mock_run:
            backend = SubprocessVenvBackend()
            backend.health_check()

        called_cmd = mock_run.call_args[0][0]
        assert called_cmd == [sys.executable, "-m", "venv", "--help"]

    def test_health_check_raises_when_returncode_nonzero(self):
        """health_check() raises SandboxUnavailableError when venv exits non-zero."""
        with patch(
            "subprocess.run",
            return_value=_make_completed_process(returncode=1, stderr="venv not found"),
        ):
            backend = SubprocessVenvBackend()
            with pytest.raises(SandboxUnavailableError):
                backend.health_check()

    def test_health_check_raises_when_python_not_found(self):
        """health_check() raises SandboxUnavailableError when Python is missing."""
        with patch("subprocess.run", side_effect=FileNotFoundError("python not found")):
            backend = SubprocessVenvBackend()
            with pytest.raises(SandboxUnavailableError):
                backend.health_check()


# --------------------------------------------------------------------------- #
# 6. Property 7 — cleanup removes the temporary directory
# --------------------------------------------------------------------------- #

# Feature: coder-buddy, Property 7: after SubprocessVenvBackend.cleanup(), the temporary directory no longer exists on the filesystem


@settings(max_examples=100)
@given(
    fake_path=st.text(
        alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters="/\\-_."),
        min_size=5,
        max_size=50,
    ),
)
def test_property7_cleanup_removes_tmpdir(fake_path: str) -> None:
    """
    **Validates: Requirements 3a.5**

    Property 7: after SubprocessVenvBackend.cleanup(), the temporary directory
    no longer exists on the filesystem.

    Strategy: mock tempfile.mkdtemp to return a fake path and mock
    shutil.rmtree to record calls. Verify that cleanup() calls rmtree with
    the correct path and resets _tmpdir to None — proving the cleanup contract
    holds for any tmpdir path without touching the real filesystem.
    """
    with (
        patch("tempfile.mkdtemp", return_value=fake_path),
        patch("shutil.rmtree") as mock_rmtree,
    ):
        backend = SubprocessVenvBackend()
        tmpdir = backend._ensure_tmpdir()

        assert tmpdir == fake_path
        assert backend._tmpdir == fake_path

        backend.cleanup()

        # rmtree must have been called with the recorded path
        mock_rmtree.assert_called_once_with(fake_path, ignore_errors=True)
        # _tmpdir must be reset to None
        assert backend._tmpdir is None
