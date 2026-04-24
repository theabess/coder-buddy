"""
Pyodide in-process sandbox backend.

Runs Python code inside the Pyodide WebAssembly runtime using ``exec()``,
with stdout/stderr redirected via ``io.StringIO``.  Dependencies are
installed via ``micropip`` (pure-Python packages only).

Requires the ``pyodide`` package::

    pip install pyodide

Note: Pyodide is designed to run inside a browser/WASM context.  When
running outside of Pyodide (e.g. in a regular CPython environment for
testing), the backend falls back to in-process ``exec()`` with
``io.StringIO`` capture — the same mechanism Pyodide itself would use.
"""

from __future__ import annotations

import inspect
import io
import sys
import threading
import traceback

from coder_buddy.config import SandboxUnavailableError
from coder_buddy.sandbox.base import ExecutionResult, SandboxBackend

# pyodide and micropip are optional dependencies
try:
    import pyodide  # type: ignore[import]
    _PYODIDE_AVAILABLE = True
except ImportError:
    pyodide = None  # type: ignore[assignment]
    _PYODIDE_AVAILABLE = False

try:
    import micropip  # type: ignore[import]
    _MICROPIP_AVAILABLE = True
except ImportError:
    micropip = None  # type: ignore[assignment]
    _MICROPIP_AVAILABLE = False


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
    4. ``cleanup()``: reset the Pyodide execution namespace.
    """

    def __init__(self) -> None:
        self._namespace: dict = {}
        self._last_result: ExecutionResult | None = None

    # ---------------------------------------------------------------------- #
    # health_check
    # ---------------------------------------------------------------------- #

    def health_check(self) -> None:
        """
        Verify that the Pyodide runtime is importable and compatible.

        Raises:
            SandboxUnavailableError: If Pyodide is not installed or
                incompatible.
        """
        if not _PYODIDE_AVAILABLE:
            raise SandboxUnavailableError(
                "The 'pyodide' package is not installed. "
                "Install it with: pip install pyodide"
            )

        # Verify version attribute is accessible (basic compatibility check)
        try:
            version = getattr(pyodide, "__version__", None)
            if version is None:
                raise SandboxUnavailableError(
                    "Pyodide is installed but version information is unavailable. "
                    "The installation may be incomplete or incompatible."
                )
        except Exception as exc:  # noqa: BLE001
            raise SandboxUnavailableError(
                f"Pyodide version check failed: {exc}"
            ) from exc

    # ---------------------------------------------------------------------- #
    # install_dependencies
    # ---------------------------------------------------------------------- #

    def install_dependencies(self, dependencies: list[str]) -> None:
        """
        Install *dependencies* via ``micropip`` (pure-Python packages only).

        In a real Pyodide environment, ``micropip.install`` is a coroutine
        and must be awaited.  This method handles both the async (Pyodide)
        and sync (testing) cases gracefully.

        Args:
            dependencies: List of pip-installable package names.
        """
        if not dependencies:
            return

        if not _MICROPIP_AVAILABLE:
            raise SandboxUnavailableError(
                "The 'micropip' package is not available. "
                "micropip is only available inside the Pyodide runtime."
            )

        try:
            result = micropip.install(dependencies)
            # micropip.install returns a coroutine in real Pyodide; handle it
            if inspect.isawaitable(result):
                # We are inside Pyodide's event loop — run the coroutine
                # synchronously using the Pyodide-provided mechanism.
                # In a real Pyodide environment, this is handled by the
                # browser's event loop.  For compatibility, we attempt to
                # run it via asyncio if available.
                import asyncio  # noqa: PLC0415
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Inside an already-running event loop (e.g. Jupyter/Pyodide)
                        # Schedule the coroutine; it will complete asynchronously.
                        loop.create_task(result)
                    else:
                        loop.run_until_complete(result)
                except RuntimeError:
                    # No event loop available — run a new one
                    asyncio.run(result)
        except Exception as exc:  # noqa: BLE001
            raise SandboxUnavailableError(
                f"micropip.install failed for {dependencies}: {exc}"
            ) from exc

    # ---------------------------------------------------------------------- #
    # execute
    # ---------------------------------------------------------------------- #

    def execute(self, source_code: str, timeout: float = 10.0) -> ExecutionResult:
        """
        Execute *source_code* inside the Pyodide runtime.

        Stdout and stderr are captured via ``io.StringIO`` redirection.
        The code runs in a controlled namespace (``self._namespace``) so
        that successive executions share state until ``cleanup()`` is called.

        A best-effort timeout is enforced via a daemon thread: if the
        execution thread does not finish within *timeout* seconds, the
        result is marked as timed out.

        Args:
            source_code: Python source code to execute.
            timeout:     Maximum wall-clock seconds allowed (best-effort).

        Returns:
            ``ExecutionResult`` populated from the captured output.
        """
        exec_result: dict = {}

        def _run() -> None:
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()

            original_stdout = sys.stdout
            original_stderr = sys.stderr

            try:
                sys.stdout = stdout_capture
                sys.stderr = stderr_capture

                exec(source_code, self._namespace)  # noqa: S102

                exec_result["stdout"] = stdout_capture.getvalue()
                exec_result["stderr"] = stderr_capture.getvalue()
                exec_result["exit_code"] = 0
            except SystemExit as exc:
                exec_result["stdout"] = stdout_capture.getvalue()
                exec_result["stderr"] = stderr_capture.getvalue()
                # SystemExit.code can be None, int, or str
                code = exc.code
                if code is None:
                    exec_result["exit_code"] = 0
                elif isinstance(code, int):
                    exec_result["exit_code"] = code
                else:
                    # Non-integer exit code — treat as error
                    exec_result["exit_code"] = 1
                    exec_result["stderr"] = (
                        exec_result["stderr"] + str(code)
                    ).strip()
            except Exception:  # noqa: BLE001
                exec_result["stdout"] = stdout_capture.getvalue()
                # Append the traceback to stderr
                tb = traceback.format_exc()
                exec_result["stderr"] = stderr_capture.getvalue() + tb
                exec_result["exit_code"] = 1
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            # Execution timed out — thread is still running in the background
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
        Reset the Pyodide execution namespace and clear the last result.

        Idempotent — safe to call multiple times.
        """
        self._namespace = {}
        self._last_result = None
