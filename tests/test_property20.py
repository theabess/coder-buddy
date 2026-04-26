"""
Property-based test for Property 20 (Task 19.6).

Kept in a separate file so it can be run without loading the expensive
module-level setup in test_integration.py (which spins up real venvs).

Feature: coder-buddy
Property 20: for any successful run with test_generation_enabled=True,
             AgentResponse.test_code is a non-empty string.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from hypothesis import given, settings
from hypothesis import strategies as st

from coder_buddy.config import AgentConfig
from coder_buddy.models import TokenUsage

# ---------------------------------------------------------------------------
# Strategy: lowercase letters, digits, spaces, and punctuation — guaranteed
# not to trigger the unsupported-language early-exit in CoderBuddy.run(),
# which checks prompt.lower() against keywords like "scala ", "kotlin", etc.
# We avoid those exact substrings by excluding the letters that form them
# entirely — using only digits and safe punctuation keeps it simple and fast.
# ---------------------------------------------------------------------------
_SAFE_PROMPT_STRATEGY = st.text(
    alphabet="0123456789 .,!?-_abcdefghijmnopquvwxyz",  # excludes: r,s,t,l,k,y,h (partial)
    min_size=1,
    max_size=80,
).filter(
    lambda p: p.strip()  # must not be all-whitespace
    and not any(
        kw in p.lower()
        for kw in (
            "javascript", "typescript", "java ", "in java", "ruby",
            "golang", "in go ", "rust ", "in rust", "c++ ", "in c++",
            "c# ", "in c#", "php ", "in php", "swift ", "in swift",
            "kotlin", "scala ", "in scala", "haskell", "r script", "in r ",
        )
    )
)

# ---------------------------------------------------------------------------
# Shared final state returned by the mocked graph on every call.
# The test_code field is always a non-empty string.
# ---------------------------------------------------------------------------
_PROP20_FINAL_STATE: dict = {
    "current_code": 'print("Hello, World!")\n',
    "file_name": "main.py",
    "dependencies": [],
    "execution_logs": "Hello, World!\n",
    "error_status": False,
    "retry_count": 0,
    "explanation": None,
    "test_code": "def test_hello_world():\n    assert True\n",
    "test_logs": "1 passed in 0.01s\n",
    "confidence_score": 4,
    "refactor_diff": "",
    "token_usage": TokenUsage(),
    "warning": None,
    "max_retries": 3,
    "pre_refactor_code": None,
    "session_history": [],
    "user_prompt": "",  # overwritten per example inside the test body
    "language": "python",
    "_route": None,
}


def _build_prop20_agent():
    """
    Build a CoderBuddy with all I/O mocked out.

    The sandbox, LLM client, and graph are all replaced with MagicMocks so
    no real venv, subprocess, or LLM call occurs.  The agent is built once
    at module level; only agent.run(prompt) is called per Hypothesis example.
    """
    from coder_buddy.agent import CoderBuddy

    config = AgentConfig(
        llm_backend="gemini-2.5-flash",
        sandbox_backend="subprocess+venv",
        max_retries=3,
        explanation_enabled=False,
        test_generation_enabled=True,
        diff_view_enabled=False,
    )

    mock_sandbox = MagicMock()
    mock_sandbox.health_check.return_value = None

    mock_graph = MagicMock()
    mock_graph.invoke.return_value = _PROP20_FINAL_STATE

    with (
        patch("coder_buddy.agent._make_sandbox", return_value=mock_sandbox),
        patch("coder_buddy.agent.LLMClient"),
        patch("coder_buddy.agent.build_graph", return_value=mock_graph),
    ):
        agent = CoderBuddy(config)

    # Attach the mock graph so the test body can update invoke() per example
    agent._graph = mock_graph
    return agent


# Built once — no per-example construction cost.
_prop20_agent = _build_prop20_agent()


# Feature: coder-buddy, Property 20: for any successful run with
# test_generation_enabled=True, AgentResponse.test_code is a non-empty string.
@given(prompt=_SAFE_PROMPT_STRATEGY)
@settings(max_examples=20)
def test_property20_test_code_is_non_empty_string_when_enabled(prompt: str) -> None:
    """
    Property 20: For any successful run with ``test_generation_enabled=True``,
    ``AgentResponse.test_code`` SHALL be a non-empty string.

    **Validates: Requirements 1.2**

    The agent is built once at module level with all I/O mocked.  Per
    example, only ``agent.run(prompt)`` is called — no venv, no LLM, no
    graph compilation — so each example completes in milliseconds.
    """
    # Update the mocked graph to return a state with the current prompt
    state = {**_PROP20_FINAL_STATE, "user_prompt": prompt}
    _prop20_agent._graph.invoke.return_value = state
    _prop20_agent.reset()

    result = _prop20_agent.run(prompt)

    assert result.success is True, (
        f"Expected success=True for prompt={prompt!r}, "
        f"got success={result.success}, failure_reason={result.failure_reason!r}"
    )
    assert result.test_code is not None, (
        f"Expected test_code to be non-None when test_generation_enabled=True, "
        f"got test_code={result.test_code!r} for prompt={prompt!r}"
    )
    assert isinstance(result.test_code, str), (
        f"Expected test_code to be a str, "
        f"got {type(result.test_code).__name__!r} for prompt={prompt!r}"
    )
    assert result.test_code.strip(), (
        f"Expected test_code to be a non-empty string, "
        f"got {result.test_code!r} for prompt={prompt!r}"
    )
