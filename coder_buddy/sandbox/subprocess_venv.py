"""
subprocess+venv sandbox backend.

Creates a temporary directory, builds a fresh Python virtual environment,
installs dependencies via pip, and executes the script as a subprocess.
The temporary directory is removed on ``cleanup()``.

This is the default backend — it requires only the Python standard library
and a working ``python3`` installation on the host.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from coder_buddy.config import SandboxUnavailableError
from coder_buddy.sandbox.base import ExecutionResult, SandboxBackend


def _venv_python(tmpdir: str) -> str:
    """Return the path to the venv Python executable (cross-platform)."""
    if sys.platform == "win32":
        return str(Path(tmpdir) / "venv" / "Scripts" / "python.exe")
    return str(Path(tmpdir) / "venv" / "bin" / "python")


def _venv_pip(tmpdir: str) -> str:
    """Return the path to the venv pip executable (cross-platform)."""
    if sys.platform == "win32":
        return str(Path(tmpdir) / "venv" / "Scripts" / "pip.exe")
    return str(Path(tmpdir) / "venv" / "bin" / "pip")


class SubprocessVenvBackend(SandboxBackend):
    """
    Sandbox backend that uses a temporary ``venv`` and ``subprocess``.

    Algorithm:
    1. ``health_check()``: verify ``python3 -m venv --help`` exits 0.
    2. ``install_dependencies(deps)``:
       a. Create temp dir via ``tempfile.mkdtemp()``.
       b. Run ``python3 -m venv {tmpdir}/venv``.
       c. Run ``{tmpdir}/venv/bin/pip install {deps} --quiet``.
    3. ``execute(source_code, timeout)``:
       a. Write *source_code* to ``{tmpdir}/script.py``.
       b. Run ``{tmpdir}/venv/bin/python {tmpdir}/script.py``
          with ``subprocess.run(..., timeout=timeout, capture_output=True)``.
       c. On ``TimeoutExpired``: kill process, set ``timed_out=True``.
    4. ``cleanup()``: ``shutil.rmtree(tmpdir, ignore_errors=True)``.
    """

    def __init__(self) -> None:
        self._tmpdir: str | None = None
        self._last_result: ExecutionResult | None = None

    def _ensure_tmpdir(self) -> str:
        """Create the temp directory if it does not exist yet, and return it."""
        if self._tmpdir is None:
            self._tmpdir = tempfile.mkdtemp()
        return self._tmpdir

    def install_dependencies(self, dependencies: list[str]) -> None:
        """
        Create a fresh venv in a temp directory and pip-install *dependencies*.

        Args:
            dependencies: List of pip-installable package names.
        """
        tmpdir = self._ensure_tmpdir()

        # Create the virtual environment
        subprocess.run(
            [sys.executable, "-m", "venv", str(Path(tmpdir) / "venv")],
            check=True,
            capture_output=True,
            text=True,
        )

        # Install dependencies if any were requested
        if dependencies:
            subprocess.run(
                [_venv_pip(tmpdir), "install", *dependencies, "--quiet"],
                check=True,
                capture_output=True,
                text=True,
            )

    def _ensure_venv(self) -> str:
        """
        Ensure the virtual environment exists inside the temp directory.

        Creates the venv if it has not been created yet (i.e. when
        ``install_dependencies`` was not called because there were no
        dependencies to install).

        Returns:
            The path to the temp directory containing the venv.
        """
        tmpdir = self._ensure_tmpdir()
        venv_dir = Path(tmpdir) / "venv"
        if not venv_dir.exists():
            subprocess.run(
                [sys.executable, "-m", "venv", str(venv_dir)],
                check=True,
                capture_output=True,
                text=True,
            )
        return tmpdir

    def execute(self, source_code: str, timeout: float = 10.0) -> ExecutionResult:
        """
        Write *source_code* to a temp file and execute it inside the venv.

        Ensures the venv exists before execution (creating it if
        ``install_dependencies`` was not called because there were no
        dependencies).

        Args:
            source_code: Python source code to execute.
            timeout:     Maximum wall-clock seconds allowed.

        Returns:
            ``ExecutionResult`` populated from the subprocess outcome.
        """
        tmpdir = self._ensure_venv()
        script_path = Path(tmpdir) / "script.py"
        script_path.write_text(source_code, encoding="utf-8")

        try:
            proc = subprocess.run(
                [_venv_python(tmpdir), str(script_path)],
                timeout=timeout,
                capture_output=True,
                text=True,
            )
            result = ExecutionResult(
                stdout=proc.stdout,
                stderr=proc.stderr,
                exit_code=proc.returncode,
                timed_out=False,
            )
        except subprocess.TimeoutExpired as exc:
            # Kill the process if it is still running
            if exc.process is not None:
                exc.process.kill()
            result = ExecutionResult(
                stdout=exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
                stderr=exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or ""),
                exit_code=-1,
                timed_out=True,
            )

        self._last_result = result
        return result

    def capture_output(self) -> str:
        """Return combined stdout + stderr from the last execution."""
        if self._last_result is None:
            return ""
        return self._last_result.combined_output

    def cleanup(self) -> None:
        """Remove the temporary directory (venv + script)."""
        if self._tmpdir is not None:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = None
            self._last_result = None

    def health_check(self) -> None:
        """
        Verify that ``python3 -m venv`` is available on the host.

        Raises:
            SandboxUnavailableError: If the venv module is not available.
        """
        try:
            result = subprocess.run(
                [sys.executable, "-m", "venv", "--help"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise SandboxUnavailableError(
                    f"python3 -m venv is not available (exit code {result.returncode}): "
                    f"{result.stderr.strip()}"
                )
        except FileNotFoundError as exc:
            raise SandboxUnavailableError(
                f"Python executable not found: {exc}"
            ) from exc
