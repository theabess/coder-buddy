"""
E2B cloud sandbox backend.

Uses the E2B SDK to create a remote cloud sandbox, install dependencies,
execute the script, and close the sandbox on cleanup.

Requires the ``e2b`` package and the ``E2B_API_KEY`` environment variable::

    pip install e2b
    export E2B_API_KEY=your_api_key
"""

from __future__ import annotations

import os

from coder_buddy.config import SandboxUnavailableError
from coder_buddy.sandbox.base import ExecutionResult, SandboxBackend

# e2b SDK is an optional dependency
try:
    from e2b import Sandbox  # type: ignore[import]
    _E2B_AVAILABLE = True
except ImportError:
    Sandbox = None  # type: ignore[assignment]
    _E2B_AVAILABLE = False


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

    # ---------------------------------------------------------------------- #
    # health_check
    # ---------------------------------------------------------------------- #

    def health_check(self) -> None:
        """
        Verify that the E2B API key is set and the service is reachable.

        Raises:
            SandboxUnavailableError: If the API key is missing, the e2b
                package is not installed, or the sandbox cannot be created.
        """
        if not _E2B_AVAILABLE:
            raise SandboxUnavailableError(
                "The 'e2b' package is not installed. "
                "Install it with: pip install e2b"
            )

        api_key = os.environ.get("E2B_API_KEY")
        if not api_key:
            raise SandboxUnavailableError(
                "E2B_API_KEY environment variable is not set. "
                "Set it with: export E2B_API_KEY=your_api_key"
            )

        try:
            sandbox = Sandbox.create(timeout=5)
            sandbox.close()
        except Exception as exc:  # noqa: BLE001
            raise SandboxUnavailableError(
                f"Failed to create E2B sandbox: {exc}"
            ) from exc

    # ---------------------------------------------------------------------- #
    # install_dependencies
    # ---------------------------------------------------------------------- #

    def install_dependencies(self, dependencies: list[str]) -> None:
        """
        Install *dependencies* inside the E2B sandbox via pip.

        Args:
            dependencies: List of pip-installable package names.
        """
        if not dependencies:
            return

        if self._sandbox is None:
            raise SandboxUnavailableError(
                "E2B sandbox is not initialised. Call health_check() first."
            )

        deps_str = " ".join(dependencies)
        self._sandbox.process.start_and_wait(f"pip install {deps_str}")

    # ---------------------------------------------------------------------- #
    # execute
    # ---------------------------------------------------------------------- #

    def execute(self, source_code: str, timeout: float = 10.0) -> ExecutionResult:
        """
        Write *source_code* to the sandbox filesystem and execute it.

        Args:
            source_code: Python source code to execute.
            timeout:     Maximum wall-clock seconds allowed.

        Returns:
            ``ExecutionResult`` populated from the sandbox process output.
        """
        if self._sandbox is None:
            raise SandboxUnavailableError(
                "E2B sandbox is not initialised. Call health_check() first."
            )

        self._sandbox.filesystem.write("/home/user/script.py", source_code)

        timed_out = False
        try:
            proc = self._sandbox.process.start_and_wait(
                "python /home/user/script.py",
                timeout=timeout,
            )
            stdout = proc.stdout if proc.stdout is not None else ""
            stderr = proc.stderr if proc.stderr is not None else ""
            exit_code = proc.exit_code if proc.exit_code is not None else 0
        except TimeoutError:
            timed_out = True
            stdout = ""
            stderr = ""
            exit_code = -1
        except Exception as exc:  # noqa: BLE001
            # Treat any other process error as a non-zero exit
            stdout = ""
            stderr = str(exc)
            exit_code = -1

        result = ExecutionResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            timed_out=timed_out,
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
        Close the E2B sandbox and release remote resources.

        Idempotent — safe to call multiple times.
        """
        if self._sandbox is not None:
            try:
                self._sandbox.close()
            except Exception:  # noqa: BLE001
                pass  # already closed or network gone — ignore
            self._sandbox = None

        self._last_result = None
