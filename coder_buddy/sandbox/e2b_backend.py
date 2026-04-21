"""
E2B cloud sandbox backend.

Uses the E2B SDK to create a remote cloud sandbox, install dependencies,
execute the script, and close the sandbox on cleanup.

Requires the ``e2b`` package and the ``E2B_API_KEY`` environment variable::

    pip install e2b
    export E2B_API_KEY=your_api_key
"""

from __future__ import annotations

from coder_buddy.sandbox.base import ExecutionResult, SandboxBackend

# e2b SDK is an optional dependency
try:
    from e2b import Sandbox  # type: ignore[import]
except ImportError:
    Sandbox = None  # type: ignore[assignment]


class E2BBackend(SandboxBackend):
    """
    Sandbox backend that executes code in an E2B cloud sandbox.

    Algorithm:
    1. ``health_check()``: verify ``E2B_API_KEY`` env var is set; attempt
       ``Sandbox.create()`` with a 5-second timeout.
    2. ``install_dependencies(deps)``:
       a. ``sandbox.process.start_and_wait(f"pip install {deps}")``.
    3. ``execute(source_code, timeout)``:
       a. ``sandbox.filesystem.write("/home/user/script.py", source_code)``.
       b. ``result = sandbox.process.start_and_wait(
              "python /home/user/script.py", timeout=timeout)``.
    4. ``cleanup()``: ``sandbox.close()``.
    """

    def __init__(self) -> None:
        self._sandbox = None
        self._last_result: ExecutionResult | None = None

    def install_dependencies(self, dependencies: list[str]) -> None:
        """
        Install *dependencies* inside the E2B sandbox via pip.

        Args:
            dependencies: List of pip-installable package names.
        """
        ...

    def execute(self, source_code: str, timeout: float = 10.0) -> ExecutionResult:
        """
        Write *source_code* to the sandbox filesystem and execute it.

        Args:
            source_code: Python source code to execute.
            timeout:     Maximum wall-clock seconds allowed.

        Returns:
            ``ExecutionResult`` populated from the sandbox process output.
        """
        ...

    def capture_output(self) -> str:
        """Return combined stdout + stderr from the last execution."""
        ...

    def cleanup(self) -> None:
        """Close the E2B sandbox and release remote resources."""
        ...

    def health_check(self) -> None:
        """
        Verify that the E2B API key is set and the service is reachable.

        Raises:
            SandboxUnavailableError: If the API key is missing or the
                sandbox cannot be created.
        """
        ...
