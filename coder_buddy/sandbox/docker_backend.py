"""
Docker sandbox backend.

Uses the Docker SDK for Python to spin up a ``python:3.12-slim``
container, execute the script inside it, and remove the container on
cleanup.

Requires the ``docker`` package::

    pip install docker
"""

from __future__ import annotations

from coder_buddy.sandbox.base import ExecutionResult, SandboxBackend

# docker SDK is an optional dependency
try:
    import docker  # type: ignore[import]
except ImportError:
    docker = None  # type: ignore[assignment]


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
       b. ``container.exec_run("pip install {deps} && python /workspace/script.py",
                               timeout=timeout)``.
    4. ``cleanup()``: ``container.remove(force=True)``.
    """

    def __init__(self) -> None:
        self._client = None
        self._container = None
        self._tmpdir: str | None = None
        self._last_result: ExecutionResult | None = None

    def install_dependencies(self, dependencies: list[str]) -> None:
        """
        Pull the Docker image and create a container with the temp volume.

        Args:
            dependencies: List of pip-installable package names.
        """
        ...

    def execute(self, source_code: str, timeout: float = 10.0) -> ExecutionResult:
        """
        Write *source_code* to the volume and execute it in the container.

        Args:
            source_code: Python source code to execute.
            timeout:     Maximum wall-clock seconds allowed.

        Returns:
            ``ExecutionResult`` populated from the container exec output.
        """
        ...

    def capture_output(self) -> str:
        """Return combined stdout + stderr from the last execution."""
        ...

    def cleanup(self) -> None:
        """Force-remove the Docker container and temp directory."""
        ...

    def health_check(self) -> None:
        """
        Ping the Docker daemon to verify it is reachable.

        Raises:
            SandboxUnavailableError: If Docker is not available.
        """
        ...
