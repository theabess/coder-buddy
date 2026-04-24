"""
Unit tests for DockerBackend.

Covers:
1. Health check failure — docker.from_env() raises DockerException → SandboxUnavailableError
2. Cleanup removes container — container.remove(force=True) is called
3. Timeout sets timed_out=True — exec_run blocks longer than timeout → timed_out=True

All tests mock docker.from_env() so no real Docker daemon is required.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, call, patch

import pytest

from coder_buddy.config import SandboxUnavailableError
from coder_buddy.sandbox.base import ExecutionResult
from coder_buddy.sandbox.docker_backend import DockerBackend


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_mock_client(ping_side_effect=None):
    """Return a mock docker client; optionally make ping() raise."""
    client = MagicMock()
    if ping_side_effect is not None:
        client.ping.side_effect = ping_side_effect
    return client


def _make_mock_container(exec_run_return=None, exec_run_side_effect=None):
    """Return a mock container with configurable exec_run behaviour."""
    container = MagicMock()
    if exec_run_side_effect is not None:
        container.exec_run.side_effect = exec_run_side_effect
    elif exec_run_return is not None:
        container.exec_run.return_value = exec_run_return
    else:
        # Default: successful execution returning empty output
        container.exec_run.return_value = (0, (b"", b""))
    return container


# --------------------------------------------------------------------------- #
# 1. Health check failure raises SandboxUnavailableError
# --------------------------------------------------------------------------- #


class TestHealthCheckFailure:
    """health_check() raises SandboxUnavailableError when Docker is unavailable."""

    def test_docker_exception_raises_sandbox_unavailable(self):
        """DockerException from docker.from_env() → SandboxUnavailableError."""
        import docker.errors

        with patch(
            "coder_buddy.sandbox.docker_backend.docker.from_env",
            side_effect=docker.errors.DockerException("daemon not running"),
        ):
            backend = DockerBackend()
            with pytest.raises(SandboxUnavailableError):
                backend.health_check()

    def test_ping_raises_docker_exception(self):
        """DockerException from client.ping() → SandboxUnavailableError."""
        import docker.errors

        mock_client = _make_mock_client(
            ping_side_effect=docker.errors.DockerException("ping failed")
        )
        with patch(
            "coder_buddy.sandbox.docker_backend.docker.from_env",
            return_value=mock_client,
        ):
            backend = DockerBackend()
            with pytest.raises(SandboxUnavailableError):
                backend.health_check()

    def test_unexpected_exception_raises_sandbox_unavailable(self):
        """Any unexpected exception from docker.from_env() → SandboxUnavailableError."""
        with patch(
            "coder_buddy.sandbox.docker_backend.docker.from_env",
            side_effect=OSError("connection refused"),
        ):
            backend = DockerBackend()
            with pytest.raises(SandboxUnavailableError):
                backend.health_check()

    def test_error_message_contains_original_cause(self):
        """SandboxUnavailableError message includes the original exception text."""
        import docker.errors

        with patch(
            "coder_buddy.sandbox.docker_backend.docker.from_env",
            side_effect=docker.errors.DockerException("daemon not running"),
        ):
            backend = DockerBackend()
            with pytest.raises(SandboxUnavailableError, match="daemon not running"):
                backend.health_check()

    def test_health_check_passes_when_daemon_reachable(self):
        """health_check() returns None (no exception) when Docker is available."""
        mock_client = _make_mock_client()
        with patch(
            "coder_buddy.sandbox.docker_backend.docker.from_env",
            return_value=mock_client,
        ):
            backend = DockerBackend()
            # Should not raise
            backend.health_check()
            mock_client.ping.assert_called_once()


# --------------------------------------------------------------------------- #
# 2. Cleanup removes container
# --------------------------------------------------------------------------- #


class TestCleanupRemovesContainer:
    """cleanup() calls container.remove(force=True) and clears internal state."""

    def _make_backend_with_container(self):
        """Return a DockerBackend with a mock container already attached."""
        backend = DockerBackend()
        mock_container = _make_mock_container()
        mock_client = _make_mock_client()
        backend._container = mock_container
        backend._client = mock_client
        backend._tmpdir = "/tmp/fake_docker_tmpdir"
        return backend, mock_container, mock_client

    def test_cleanup_calls_remove_force_true(self):
        """container.remove(force=True) is called during cleanup()."""
        backend, mock_container, _ = self._make_backend_with_container()

        with patch("shutil.rmtree"):
            backend.cleanup()

        mock_container.remove.assert_called_once_with(force=True)

    def test_cleanup_sets_container_to_none(self):
        """After cleanup(), _container is None."""
        backend, _, _ = self._make_backend_with_container()

        with patch("shutil.rmtree"):
            backend.cleanup()

        assert backend._container is None

    def test_cleanup_closes_client(self):
        """cleanup() also closes the docker client."""
        backend, _, mock_client = self._make_backend_with_container()

        with patch("shutil.rmtree"):
            backend.cleanup()

        mock_client.close.assert_called_once()

    def test_cleanup_sets_client_to_none(self):
        """After cleanup(), _client is None."""
        backend, _, _ = self._make_backend_with_container()

        with patch("shutil.rmtree"):
            backend.cleanup()

        assert backend._client is None

    def test_cleanup_removes_tmpdir(self):
        """cleanup() calls shutil.rmtree on the temp directory."""
        backend, _, _ = self._make_backend_with_container()

        with patch("shutil.rmtree") as mock_rmtree:
            backend.cleanup()

        mock_rmtree.assert_called_once_with("/tmp/fake_docker_tmpdir", ignore_errors=True)

    def test_cleanup_is_idempotent(self):
        """Calling cleanup() twice does not raise."""
        backend, mock_container, _ = self._make_backend_with_container()

        with patch("shutil.rmtree"):
            backend.cleanup()
            backend.cleanup()  # second call — container is already None

        # remove should only have been called once
        mock_container.remove.assert_called_once_with(force=True)

    def test_cleanup_tolerates_remove_exception(self):
        """cleanup() does not propagate exceptions from container.remove()."""
        backend, mock_container, _ = self._make_backend_with_container()
        mock_container.remove.side_effect = Exception("already removed")

        with patch("shutil.rmtree"):
            # Should not raise
            backend.cleanup()

    def test_cleanup_with_no_container_does_not_raise(self):
        """cleanup() on a fresh backend (no container) does not raise."""
        backend = DockerBackend()
        # Should not raise
        backend.cleanup()


# --------------------------------------------------------------------------- #
# 3. Timeout sets timed_out=True
# --------------------------------------------------------------------------- #


class TestTimeoutSetsTimed_out:
    """execute() returns timed_out=True when exec_run takes longer than timeout."""

    def _make_backend_with_slow_container(self, sleep_seconds: float = 5.0):
        """
        Return a DockerBackend whose container.exec_run() blocks for
        *sleep_seconds*, simulating a long-running execution.
        """
        backend = DockerBackend()
        backend._tmpdir = "/tmp/fake_docker_tmpdir"

        def _slow_exec_run(*args, **kwargs):
            time.sleep(sleep_seconds)
            return (0, (b"output", b""))

        mock_container = _make_mock_container(exec_run_side_effect=_slow_exec_run)
        backend._container = mock_container
        return backend

    def test_timed_out_true_when_exec_run_blocks(self):
        """timed_out=True when exec_run takes longer than the timeout."""
        backend = self._make_backend_with_slow_container(sleep_seconds=5.0)

        with patch("pathlib.Path.write_text"):
            result = backend.execute("import time; time.sleep(5)", timeout=0.05)

        assert result.timed_out is True

    def test_exit_code_negative_one_on_timeout(self):
        """exit_code is -1 when execution timed out."""
        backend = self._make_backend_with_slow_container(sleep_seconds=5.0)

        with patch("pathlib.Path.write_text"):
            result = backend.execute("import time; time.sleep(5)", timeout=0.05)

        assert result.exit_code == -1

    def test_has_errors_true_on_timeout(self):
        """has_errors is True when execution timed out."""
        backend = self._make_backend_with_slow_container(sleep_seconds=5.0)

        with patch("pathlib.Path.write_text"):
            result = backend.execute("import time; time.sleep(5)", timeout=0.05)

        assert result.has_errors is True

    def test_result_is_execution_result_on_timeout(self):
        """execute() returns an ExecutionResult instance even on timeout."""
        backend = self._make_backend_with_slow_container(sleep_seconds=5.0)

        with patch("pathlib.Path.write_text"):
            result = backend.execute("...", timeout=0.05)

        assert isinstance(result, ExecutionResult)

    def test_timed_out_false_for_fast_execution(self):
        """timed_out=False when exec_run completes within the timeout."""
        backend = DockerBackend()
        backend._tmpdir = "/tmp/fake_docker_tmpdir"
        mock_container = _make_mock_container(
            exec_run_return=(0, (b"hello\n", b""))
        )
        backend._container = mock_container

        with patch("pathlib.Path.write_text"):
            result = backend.execute("print('hello')", timeout=5.0)

        assert result.timed_out is False
        assert result.stdout == "hello\n"
        assert result.exit_code == 0
