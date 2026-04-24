"""
Docker sandbox backend.

Uses the Docker SDK for Python to spin up a ``python:3.12-slim``
container, execute the script inside it, and remove the container on
cleanup.

Requires the ``docker`` package::

    pip install docker
"""

from __future__ import annotations

import shutil
import tempfile
import threading
from pathlib import Path

from coder_buddy.config import SandboxUnavailableError
from coder_buddy.sandbox.base import ExecutionResult, SandboxBackend

# docker SDK is an optional dependency
try:
    import docker  # type: ignore[import]
    from docker.errors import DockerException  # type: ignore[import]
except ImportError:
    docker = None  # type: ignore[assignment]
    DockerException = Exception  # type: ignore[assignment]

_DOCKER_IMAGE = "python:3.12-slim"


class DockerBackend(SandboxBackend):
    """
    Sandbox backend that executes code inside a Docker container.

    Algorithm:
    1. ``health_check()``: ``docker.from_env().ping()`` — raises
       ``SandboxUnavailableError`` if the Docker daemon is unreachable.
    2. ``install_dependencies(deps)``:
       a. Pull ``python:3.12-slim`` if not present.
       b. Create a container with a volume mount for the temp directory.
    3. ``execute(source_code, timeout)``:
       a. Write *source_code* to ``{tmpdir}/script.py``.
       b. ``container.exec_run("python /workspace/script.py")``.
    4. ``cleanup()``: ``container.remove(force=True)``.
    """

    def __init__(self) -> None:
        self._client = None
        self._container = None
        self._tmpdir: str | None = None
        self._last_result: ExecutionResult | None = None

    # ---------------------------------------------------------------------- #
    # health_check
    # ---------------------------------------------------------------------- #

    def health_check(self) -> None:
        """
        Ping the Docker daemon to verify it is reachable.

        Raises:
            SandboxUnavailableError: If Docker is not available or the
                daemon is not running.
        """
        if docker is None:
            raise SandboxUnavailableError(
                "The 'docker' package is not installed. "
                "Install it with: pip install docker"
            )
        try:
            client = docker.from_env()
            client.ping()
        except DockerException as exc:
            raise SandboxUnavailableError(
                f"Docker daemon is not reachable: {exc}"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise SandboxUnavailableError(
                f"Unexpected error while connecting to Docker: {exc}"
            ) from exc

    # ---------------------------------------------------------------------- #
    # install_dependencies
    # ---------------------------------------------------------------------- #

    def install_dependencies(self, dependencies: list[str]) -> None:
        """
        Pull the Docker image and create a long-running container with the
        temp directory mounted as ``/workspace``.

        Args:
            dependencies: List of pip-installable package names.
        """
        if docker is None:
            raise SandboxUnavailableError(
                "The 'docker' package is not installed. "
                "Install it with: pip install docker"
            )

        # Create temp directory (idempotent — reuse if already created)
        if self._tmpdir is None:
            self._tmpdir = tempfile.mkdtemp()

        self._client = docker.from_env()

        # Pull image if not already present
        try:
            self._client.images.get(_DOCKER_IMAGE)
        except docker.errors.ImageNotFound:
            self._client.images.pull(_DOCKER_IMAGE)

        # Create a detached, tty container so it stays alive for exec_run calls
        self._container = self._client.containers.run(
            _DOCKER_IMAGE,
            command="tail -f /dev/null",  # keep container alive
            volumes={self._tmpdir: {"bind": "/workspace", "mode": "rw"}},
            network_mode="bridge",
            detach=True,
            tty=True,
            remove=False,
        )

        # Install dependencies inside the container if any were requested
        if dependencies:
            deps_str = " ".join(dependencies)
            exit_code, output = self._container.exec_run(
                f"pip install {deps_str} --quiet",
                demux=False,
            )
            if exit_code != 0:
                output_text = (
                    output.decode("utf-8", errors="replace")
                    if isinstance(output, bytes)
                    else (output or "")
                )
                raise SandboxUnavailableError(
                    f"pip install failed (exit {exit_code}): {output_text}"
                )

    # ---------------------------------------------------------------------- #
    # execute
    # ---------------------------------------------------------------------- #

    def execute(self, source_code: str, timeout: float = 10.0) -> ExecutionResult:
        """
        Write *source_code* to the volume and execute it in the container.

        Docker SDK's ``exec_run`` does not natively support a wall-clock
        timeout, so we run it in a daemon thread and join with the given
        timeout.

        Args:
            source_code: Python source code to execute.
            timeout:     Maximum wall-clock seconds allowed.

        Returns:
            ``ExecutionResult`` populated from the container exec output.
        """
        if self._container is None or self._tmpdir is None:
            raise SandboxUnavailableError(
                "DockerBackend.install_dependencies() must be called before execute()."
            )

        # Write the script to the shared volume
        script_path = Path(self._tmpdir) / "script.py"
        script_path.write_text(source_code, encoding="utf-8")

        # Run exec_run in a thread so we can enforce a timeout
        exec_result: dict = {}
        timed_out_flag: list[bool] = [False]

        def _run() -> None:
            try:
                exit_code, (stdout_bytes, stderr_bytes) = self._container.exec_run(
                    "python /workspace/script.py",
                    demux=True,
                )
                exec_result["exit_code"] = exit_code
                exec_result["stdout"] = (
                    stdout_bytes.decode("utf-8", errors="replace")
                    if stdout_bytes
                    else ""
                )
                exec_result["stderr"] = (
                    stderr_bytes.decode("utf-8", errors="replace")
                    if stderr_bytes
                    else ""
                )
            except Exception as exc:  # noqa: BLE001
                exec_result["exit_code"] = -1
                exec_result["stdout"] = ""
                exec_result["stderr"] = str(exc)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            # Thread is still running — execution timed out
            timed_out_flag[0] = True
            result = ExecutionResult(
                stdout=exec_result.get("stdout", ""),
                stderr=exec_result.get("stderr", ""),
                exit_code=-1,
                timed_out=True,
            )
        else:
            result = ExecutionResult(
                stdout=exec_result.get("stdout", ""),
                stderr=exec_result.get("stderr", ""),
                exit_code=exec_result.get("exit_code", -1),
                timed_out=False,
            )

        self._last_result = result
        return result

    # ---------------------------------------------------------------------- #
    # capture_output
    # ---------------------------------------------------------------------- #

    def capture_output(self) -> str:
        """Return combined stdout + stderr from the last execution."""
        if self._last_result is None:
            return ""
        return self._last_result.combined_output

    # ---------------------------------------------------------------------- #
    # cleanup
    # ---------------------------------------------------------------------- #

    def cleanup(self) -> None:
        """
        Force-remove the Docker container and delete the temp directory.

        Idempotent — safe to call multiple times.
        """
        if self._container is not None:
            try:
                self._container.remove(force=True)
            except Exception:  # noqa: BLE001
                pass  # already removed or daemon gone — ignore
            self._container = None

        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
            self._client = None

        if self._tmpdir is not None:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = None

        self._last_result = None
