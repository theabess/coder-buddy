"""
SandboxBackend abstract interface and ExecutionResult dataclass.

All concrete sandbox backends must subclass ``SandboxBackend`` and
implement every abstract method.  ``Execute_Node`` depends only on this
interface — never on a concrete implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ExecutionResult:
    """
    The result of a single sandbox code execution.

    Attributes:
        stdout:    Captured standard output from the executed script.
        stderr:    Captured standard error from the executed script.
        exit_code: Process exit code (``0`` indicates success).
        timed_out: ``True`` if the execution exceeded the timeout limit.
    """

    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool

    @property
    def combined_output(self) -> str:
        """
        Concatenate stdout, stderr, and (if applicable) a timeout notice.

        Returns:
            A single string with all output sections joined by newlines.
            Sections that are empty are omitted.
        """
        parts: list[str] = []
        if self.stdout:
            parts.append(self.stdout)
        if self.stderr:
            parts.append(self.stderr)
        if self.timed_out:
            parts.append("[TIMEOUT: execution exceeded time limit]")
        return "\n".join(parts)

    @property
    def has_errors(self) -> bool:
        """
        Return ``True`` if the execution produced errors.

        An execution is considered erroneous if the exit code is non-zero
        or if it timed out.
        """
        return self.exit_code != 0 or self.timed_out


class SandboxBackend(ABC):
    """
    Abstract base class for all sandbox execution backends.

    Concrete implementations must provide all five abstract methods.
    The strategy pattern ensures ``Execute_Node`` is decoupled from any
    specific execution environment.
    """

    @abstractmethod
    def install_dependencies(self, dependencies: list[str]) -> None:
        """
        Install *dependencies* into the sandbox environment.

        Args:
            dependencies: List of pip-installable package names.
        """
        ...

    @abstractmethod
    def execute(self, source_code: str, timeout: float = 10.0) -> ExecutionResult:
        """
        Execute *source_code* inside the sandbox.

        Args:
            source_code: Python source code to execute.
            timeout:     Maximum wall-clock seconds allowed.

        Returns:
            ``ExecutionResult`` with stdout, stderr, exit_code, timed_out.

        Raises:
            SandboxUnavailableError: If the backend cannot be reached.
        """
        ...

    @abstractmethod
    def capture_output(self) -> str:
        """
        Return the combined stdout + stderr from the last execution.

        Returns:
            Combined output string.
        """
        ...

    @abstractmethod
    def cleanup(self) -> None:
        """
        Release all temporary resources (venv, container, remote sandbox).

        Must be idempotent — calling it multiple times must not raise.
        """
        ...

    @abstractmethod
    def health_check(self) -> None:
        """
        Verify that the backend is available and ready.

        Called once at agent startup.

        Raises:
            SandboxUnavailableError: With a descriptive message if the
                backend is not available.
        """
        ...
