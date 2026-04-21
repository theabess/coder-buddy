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

from coder_buddy.sandbox.base import ExecutionResult, SandboxBackend


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

    def install_dependencies(self, dependencies: list[str]) -> None:
        """
        Create a fresh venv in a temp directory and pip-install *dependencies*.

        Args:
            dependencies: List of pip-installable package names.
        """
        ...

    def execute(self, source_code: str, timeout: float = 10.0) -> ExecutionResult:
        """
        Write *source_code* to a temp file and execute it inside the venv.

        Args:
            source_code: Python source code to execute.
            timeout:     Maximum wall-clock seconds allowed.

        Returns:
            ``ExecutionResult`` populated from the subprocess outcome.
        """
        ...

    def capture_output(self) -> str:
        """Return combined stdout + stderr from the last execution."""
        ...

    def cleanup(self) -> None:
        """Remove the temporary directory (venv + script)."""
        ...

    def health_check(self) -> None:
        """
        Verify that ``python3 -m venv`` is available on the host.

        Raises:
            SandboxUnavailableError: If the venv module is not available.
        """
        ...
