"""
Test_Node — LLM-based test generation and execution node.

Generates a pytest suite via the LLM, executes it in the sandbox, and
retries up to 3 times on failure.  If all retries fail, a warning is
stored in the state but ``source_code`` is left unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from coder_buddy.logging_utils import log_node_event
from coder_buddy.models import CodeArtifact

if TYPE_CHECKING:
    from coder_buddy.config import AgentConfig
    from coder_buddy.llm.client import LLMClient
    from coder_buddy.sandbox.base import SandboxBackend
    from coder_buddy.state import AgentState


# Maximum number of test-generation retry attempts
_MAX_TEST_RETRIES = 3


def make_test_node(
    sandbox: "SandboxBackend",
    llm_client: "LLMClient",
    config: "AgentConfig",
):
    """
    Factory that returns a ``test_node`` closure bound to *sandbox*,
    *llm_client*, and *config*.

    Args:
        sandbox:    Configured ``SandboxBackend`` instance.
        llm_client: Configured ``LLMClient`` instance.
        config:     ``AgentConfig`` controlling test generation behaviour.

    Returns:
        A ``test_node(state) -> dict`` function suitable for use as a
        LangGraph node.
    """

    def test_node(state: "AgentState") -> dict:
        """
        Generate and execute a pytest suite for ``state["current_code"]``.

        Steps:
        1. If ``config.test_generation_enabled`` is ``False``, return
           ``{test_code: None, test_logs: None}`` immediately.
        2. Call ``LLMClient.generate`` to produce a pytest suite as a
           ``CodeArtifact`` whose ``source_code`` is the test file content.
        3. Execute the test suite in the sandbox.
        4. On failure (``error_status``), retry up to 3 times, including
           the failure logs in the revised prompt.
        5. If all retries fail, return the last test code with a warning
           message; ``source_code`` is NOT modified.
        6. On success, return ``{test_code, test_logs}`` with no warning.

        Emits ``log_node_event`` at the start and end of node execution.

        Returns:
            Partial ``AgentState`` dict with keys:
            ``test_code``, ``test_logs``, ``token_usage``.
            May also include ``warning`` if all test retries fail.
        """
        retry_count: int = state["retry_count"]
        current_code: str = state["current_code"]
        file_name: str = state["file_name"]
        dependencies: list[str] = state["dependencies"]
        token_usage = state["token_usage"]

        log_node_event(
            node="test_node",
            event="start",
            retry_count=retry_count,
        )

        # --- Feature flag: skip test generation if disabled ---
        if not config.test_generation_enabled:
            log_node_event(
                node="test_node",
                event="end",
                retry_count=retry_count,
                outcome="disabled",
            )
            return {
                "test_code": None,
                "test_logs": None,
                "token_usage": token_usage,
            }

        # --- Initial test generation ---
        prompt = _build_test_prompt(
            source_code=current_code,
            file_name=file_name,
            failure_logs=None,
        )
        artifact, token_record = llm_client.generate(prompt, CodeArtifact)
        token_usage = token_usage.model_copy(update={"test_node": token_record})

        test_code: str = artifact.source_code
        # Merge source dependencies with any extra deps declared by the test
        # artifact (e.g. "pytest" when the runner script needs it).
        test_dependencies: list[str] = list(
            dict.fromkeys(dependencies + artifact.dependencies)
        )

        # --- Execute and retry loop ---
        test_logs: str = ""
        last_failure_logs: str = ""

        for attempt in range(_MAX_TEST_RETRIES):
            # Build the runner script: the test code + a self-executing block
            runner_script = _build_runner_script(test_code, current_code, file_name)

            try:
                if test_dependencies:
                    sandbox.install_dependencies(test_dependencies)

                result = sandbox.execute(runner_script, config.sandbox_timeout_seconds)
                test_logs = result.combined_output

                if not result.has_errors:
                    # Tests passed — success path
                    log_node_event(
                        node="test_node",
                        event="end",
                        retry_count=retry_count,
                        outcome="success",
                        extra={"test_attempts": attempt + 1},
                    )
                    return {
                        "test_code": test_code,
                        "test_logs": test_logs,
                        "token_usage": token_usage,
                    }

                # Tests failed — record logs for retry prompt
                last_failure_logs = test_logs

            finally:
                sandbox.cleanup()

            # Retry: ask the LLM to fix the test suite
            if attempt < _MAX_TEST_RETRIES - 1:
                retry_prompt = _build_test_prompt(
                    source_code=current_code,
                    file_name=file_name,
                    failure_logs=last_failure_logs,
                )
                artifact, token_record = llm_client.generate(retry_prompt, CodeArtifact)
                token_usage = token_usage.model_copy(update={"test_node": token_record})
                test_code = artifact.source_code
                # Re-merge dependencies in case the revised artifact adds new ones
                test_dependencies = list(
                    dict.fromkeys(dependencies + artifact.dependencies)
                )

        # --- All retries exhausted ---
        warning_msg = (
            f"Test generation failed after {_MAX_TEST_RETRIES} attempt(s). "
            "The generated test suite could not be executed successfully. "
            "Please review the test code and logs manually."
        )

        log_node_event(
            node="test_node",
            event="end",
            retry_count=retry_count,
            outcome="failed",
            extra={"test_attempts": _MAX_TEST_RETRIES},
        )

        return {
            "test_code": test_code,
            "test_logs": last_failure_logs,
            "token_usage": token_usage,
            "warning": warning_msg,
        }

    return test_node


def _build_test_prompt(
    source_code: str,
    file_name: str,
    failure_logs: "str | None",
) -> str:
    """
    Build the LLM prompt requesting a pytest test suite for *source_code*.

    When *failure_logs* is provided, the prompt includes the failure output
    and asks the LLM to revise the tests to fix the issues.

    The generated test file must be self-executing: it must include a
    ``if __name__ == "__main__":`` block that calls
    ``pytest.main([__file__, "-v"])`` so the sandbox can run it directly
    as a Python script.

    Args:
        source_code:  The source code to generate tests for.
        file_name:    The filename of the source code (for context).
        failure_logs: Combined stdout/stderr from a failed test run, or
                      ``None`` for the initial generation request.

    Returns:
        The complete prompt string to send to the LLM.
    """
    if failure_logs:
        intro = (
            f"The following pytest test suite for `{file_name}` failed to execute "
            "successfully. Please revise the tests to fix the issues shown in the "
            "failure output below.\n\n"
            "Failure output:\n"
            f"```\n{failure_logs}\n```\n\n"
            "Revised test suite requirements:\n"
        )
    else:
        intro = (
            f"Generate a comprehensive pytest test suite for the following Python "
            f"script (`{file_name}`).\n\n"
            "Test suite requirements:\n"
        )

    requirements = (
        "1. Use pytest as the testing framework.\n"
        "2. Write tests that cover the main functionality and important edge cases.\n"
        "3. Each test function must start with `test_`.\n"
        "4. The test file MUST end with the following self-executing block so it "
        "can be run directly as a Python script:\n"
        "   ```python\n"
        "   if __name__ == '__main__':\n"
        "       import pytest\n"
        "       pytest.main([__file__, '-v'])\n"
        "   ```\n"
        "5. Do NOT use any external dependencies beyond pytest and the standard library "
        "(unless the source code itself requires them).\n"
        "6. Return the test suite as a CodeArtifact where `source_code` contains the "
        "complete test file content, `file_name` is the test filename (e.g. "
        f"`test_{file_name}`), `dependencies` lists any required packages, and "
        "`language` is `\"python\"`.\n\n"
        "Source code to test:\n"
        f"```python\n{source_code}\n```"
    )

    return intro + requirements


def _build_runner_script(
    test_code: str,
    source_code: str,
    file_name: str,
) -> str:
    """
    Build a self-contained runner script that embeds both the source code
    and the test suite, then executes the tests via ``pytest.main``.

    The sandbox's ``execute()`` method writes the script to a temp file and
    runs it.  By embedding the source code as a module and the tests inline,
    we avoid any filesystem path issues.

    Args:
        test_code:   The pytest test suite source code.
        source_code: The original source code being tested.
        file_name:   The filename of the source code (used as the module name).

    Returns:
        A complete Python script that, when executed, runs the test suite.
    """
    # Derive a valid Python module name from the file_name
    module_name = file_name.replace(".py", "").replace("-", "_").replace(" ", "_")

    # Build a runner that:
    # 1. Writes the source code to a temp file so it can be imported
    # 2. Writes the test code to a temp file
    # 3. Runs pytest on the test file
    runner = (
        "import sys\n"
        "import os\n"
        "import tempfile\n"
        "import textwrap\n"
        "\n"
        "# Write source code to a temp file so tests can import it\n"
        "_tmpdir = tempfile.mkdtemp()\n"
        f"_source_path = os.path.join(_tmpdir, {file_name!r})\n"
        "_test_path = os.path.join(_tmpdir, 'test_generated.py')\n"
        "\n"
        "# Embed source code\n"
        "_source_code = " + repr(source_code) + "\n"
        "with open(_source_path, 'w', encoding='utf-8') as _f:\n"
        "    _f.write(_source_code)\n"
        "\n"
        "# Embed test code\n"
        "_test_code = " + repr(test_code) + "\n"
        "with open(_test_path, 'w', encoding='utf-8') as _f:\n"
        "    _f.write(_test_code)\n"
        "\n"
        "# Add the temp dir to sys.path so tests can import the source module\n"
        "sys.path.insert(0, _tmpdir)\n"
        "\n"
        "# Run pytest on the test file\n"
        "import pytest\n"
        "_exit_code = pytest.main([_test_path, '-v', '--tb=short'])\n"
        "\n"
        "# Clean up\n"
        "import shutil\n"
        "shutil.rmtree(_tmpdir, ignore_errors=True)\n"
        "\n"
        "# Exit with pytest's exit code so the sandbox detects failures\n"
        "sys.exit(_exit_code)\n"
    )
    return runner
