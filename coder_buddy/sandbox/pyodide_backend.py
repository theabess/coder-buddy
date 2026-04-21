"""
Pyodide in-process sandbox backend.

Runs Python code inside the Pyodide WebAssembly runtime using ``exec()``,
with stdout/stderr redirected via ``io.StringIO``.  Dependencies are
installed via ``micropip`` (pure-Python packages only).

Requires the ``pyodide`` package::

    pip install pyodide
"""

from __future__ import annotations

from coder_buddy.sandbox.base import ExecutionResult, SandboxBackend

# pyodide is an optional dependency
try:
    import pyodide  # type: ignore[import]
    import micropip  # type: ignore[import]
except ImportError:
    pyodide = None  # type: ignore[assignment]
    micropip = None  # type: ignore[assignment]


class PyodideBackend(SandboxBackend):
    """
    Sandbox backend that executes code inside the Pyodide runtime.

    Algorithm:
    1. ``health_check()``: ``import pyodide``; verify version compatibility.
    2. ``install_dependencies(deps)``:
       a. ``await micropip.install(deps)`` — pure-Python packages only.
    3. ``execute(source_code, timeout)``:
       a. Run *source_code* via ``exec()`` in the Pyodide runtime.
       b. Capture stdout/stderr via ``io.StringIO`` redirect.
    4. ``cleanup()``: reset the Pyodide namespace.
    """

    def __init__(self) -> None:
        self._namespace: dict = {}
        self._last_result: ExecutionResult | None = None

    def install_dependencies(self, dependencies: list[str]) -> None:
        """
        Install *dependencies* via ``micropip`` (pure-Python packages only).

        Args:
            dependencies: List of pip-installable package names.
        """
        ...

    def execute(self, source_code: str, timeout: float = 10.0) -> ExecutionResult:
        """
        Execute *source_code* inside the Pyodide runtime.

        Stdout and stderr are captured via ``io.StringIO`` redirection.

        Args:
            source_code: Python source code to execute.
            timeout:     Maximum wall-clock seconds allowed (best-effort).

        Returns:
            ``ExecutionResult`` populated from the captured output.
        """
        ...

    def capture_output(self) -> str:
        """Return combined stdout + stderr from the last execution."""
        ...

    def cleanup(self) -> None:
        """Reset the Pyodide execution namespace."""
        ...

    def health_check(self) -> None:
        """
        Verify that the Pyodide runtime is importable and compatible.

        Raises:
            SandboxUnavailableError: If Pyodide is not installed or
                incompatible.
        """
        ...
