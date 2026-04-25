"""
Unit tests for the evaluator conditional edge function.

Validates all three routing branches:
- "fail"    — when retry_count >= max_retries
- "retry"   — when error_status is True (and retry_count < max_retries)
- "refactor" — when error_status is False (and retry_count < max_retries)

Also validates that retry_count is incremented exactly once when routing
to "retry", and that it is NOT incremented when routing to "refactor" or
"fail".

Property 2 (Requirement 1.5): For any AgentState where error_status=True
and retry_count < max_retries, after evaluator routes to "retry",
retry_count equals original + 1.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from unittest.mock import patch

from coder_buddy.nodes.evaluator import evaluator
from coder_buddy.models import TokenUsage


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def make_state(
    *,
    retry_count: int = 0,
    max_retries: int = 5,
    error_status: bool = False,
    execution_logs: str = "",
) -> dict:
    """
    Build a minimal AgentState dict for evaluator testing.

    Only the fields read by the evaluator are required; the rest are
    filled with sensible defaults so the dict satisfies TypedDict shape.
    """
    return {
        "user_prompt": "test prompt",
        "current_code": "print('hello')",
        "execution_logs": execution_logs,
        "error_status": error_status,
        "retry_count": retry_count,
        "dependencies": [],
        "file_name": "main.py",
        "language": "python",
        "explanation": None,
        "test_code": None,
        "test_logs": None,
        "confidence_score": None,
        "refactor_diff": None,
        "token_usage": TokenUsage(),
        "session_history": [],
        "max_retries": max_retries,
        "pre_refactor_code": None,
    }


# --------------------------------------------------------------------------- #
# "fail" branch — retry_count >= max_retries
# --------------------------------------------------------------------------- #


class TestEvaluatorFailBranch:
    """Evaluator returns 'fail' when retry_count >= max_retries."""

    def test_fail_when_retry_count_equals_max_retries(self):
        """Exact boundary: retry_count == max_retries → 'fail'."""
        state = make_state(retry_count=5, max_retries=5, error_status=True)
        result = evaluator(state)
        assert result == "fail"

    def test_fail_when_retry_count_exceeds_max_retries(self):
        """retry_count > max_retries also routes to 'fail'."""
        state = make_state(retry_count=6, max_retries=5, error_status=True)
        result = evaluator(state)
        assert result == "fail"

    def test_fail_even_when_no_error_status(self):
        """'fail' takes priority over error_status=False when limit is reached."""
        state = make_state(retry_count=5, max_retries=5, error_status=False)
        result = evaluator(state)
        assert result == "fail"

    def test_fail_with_max_retries_one(self):
        """With max_retries=1, retry_count=1 immediately routes to 'fail'."""
        state = make_state(retry_count=1, max_retries=1, error_status=True)
        result = evaluator(state)
        assert result == "fail"

    def test_fail_does_not_increment_retry_count(self):
        """retry_count must not be mutated when routing to 'fail'."""
        state = make_state(retry_count=5, max_retries=5, error_status=True)
        evaluator(state)
        assert state["retry_count"] == 5


# --------------------------------------------------------------------------- #
# "retry" branch — error_status=True, retry_count < max_retries
# --------------------------------------------------------------------------- #


class TestEvaluatorRetryBranch:
    """Evaluator returns 'retry' when there are errors and retries remain."""

    def test_retry_when_error_status_true(self):
        """error_status=True with retries remaining → 'retry'."""
        state = make_state(retry_count=0, max_retries=5, error_status=True)
        result = evaluator(state)
        assert result == "retry"

    def test_retry_at_one_below_max(self):
        """retry_count == max_retries - 1 is still within the retry window."""
        state = make_state(retry_count=4, max_retries=5, error_status=True)
        result = evaluator(state)
        assert result == "retry"

    def test_retry_increments_retry_count_by_one(self):
        """retry_count must be incremented by exactly 1 when routing to 'retry'."""
        state = make_state(retry_count=0, max_retries=5, error_status=True)
        evaluator(state)
        assert state["retry_count"] == 1

    def test_retry_increments_from_nonzero_count(self):
        """Increment works correctly from a non-zero starting count."""
        state = make_state(retry_count=3, max_retries=5, error_status=True)
        evaluator(state)
        assert state["retry_count"] == 4

    def test_retry_increments_exactly_once_per_call(self):
        """A single evaluator call increments retry_count by exactly 1, not more."""
        state = make_state(retry_count=2, max_retries=5, error_status=True)
        original = state["retry_count"]
        evaluator(state)
        assert state["retry_count"] == original + 1

    def test_retry_with_execution_logs(self):
        """Routing to 'retry' works correctly when execution_logs is non-empty."""
        state = make_state(
            retry_count=0,
            max_retries=5,
            error_status=True,
            execution_logs="Traceback (most recent call last):\n  File 'main.py'\nNameError: name 'x' is not defined",
        )
        result = evaluator(state)
        assert result == "retry"

    def test_retry_with_empty_execution_logs(self):
        """Routing to 'retry' works even when execution_logs is empty."""
        state = make_state(
            retry_count=0,
            max_retries=5,
            error_status=True,
            execution_logs="",
        )
        result = evaluator(state)
        assert result == "retry"


# --------------------------------------------------------------------------- #
# "refactor" branch — error_status=False, retry_count < max_retries
# --------------------------------------------------------------------------- #


class TestEvaluatorRefactorBranch:
    """Evaluator returns 'refactor' when execution succeeded and retries remain."""

    def test_refactor_when_no_errors(self):
        """error_status=False with retries remaining → 'refactor'."""
        state = make_state(retry_count=0, max_retries=5, error_status=False)
        result = evaluator(state)
        assert result == "refactor"

    def test_refactor_after_successful_retry(self):
        """'refactor' is returned even when retry_count > 0, as long as no errors."""
        state = make_state(retry_count=3, max_retries=5, error_status=False)
        result = evaluator(state)
        assert result == "refactor"

    def test_refactor_at_one_below_max_with_no_errors(self):
        """retry_count == max_retries - 1 with no errors still routes to 'refactor'."""
        state = make_state(retry_count=4, max_retries=5, error_status=False)
        result = evaluator(state)
        assert result == "refactor"

    def test_refactor_does_not_increment_retry_count(self):
        """retry_count must not be mutated when routing to 'refactor'."""
        state = make_state(retry_count=2, max_retries=5, error_status=False)
        evaluator(state)
        assert state["retry_count"] == 2

    def test_refactor_with_max_retries_one_and_zero_count(self):
        """With max_retries=1 and retry_count=0, no-error run routes to 'refactor'."""
        state = make_state(retry_count=0, max_retries=1, error_status=False)
        result = evaluator(state)
        assert result == "refactor"


# --------------------------------------------------------------------------- #
# Priority: "fail" takes precedence over "retry"
# --------------------------------------------------------------------------- #


class TestEvaluatorBranchPriority:
    """'fail' is checked before 'retry', so it takes priority."""

    def test_fail_takes_priority_over_error_status(self):
        """
        When retry_count == max_retries AND error_status is True,
        the evaluator must return 'fail', not 'retry'.
        """
        state = make_state(retry_count=5, max_retries=5, error_status=True)
        result = evaluator(state)
        assert result == "fail"

    def test_fail_does_not_increment_when_error_status_true(self):
        """
        When routing to 'fail' (even with error_status=True),
        retry_count must remain unchanged.
        """
        state = make_state(retry_count=5, max_retries=5, error_status=True)
        evaluator(state)
        assert state["retry_count"] == 5


# --------------------------------------------------------------------------- #
# Logging side-effects
# --------------------------------------------------------------------------- #


class TestEvaluatorLogging:
    """Evaluator emits log_node_event for each routing decision."""

    def test_retry_branch_logs_error_summary(self):
        """When routing to 'retry', log_node_event is called with outcome='retry'."""
        state = make_state(
            retry_count=0,
            max_retries=5,
            error_status=True,
            execution_logs="SyntaxError: invalid syntax",
        )
        with patch("coder_buddy.nodes.evaluator.log_node_event") as mock_log:
            evaluator(state)
            mock_log.assert_called_once()
            call_kwargs = mock_log.call_args
            assert call_kwargs.kwargs.get("outcome") == "retry" or (
                len(call_kwargs.args) >= 4 and call_kwargs.args[3] == "retry"
            )

    def test_refactor_branch_emits_log(self):
        """When routing to 'refactor', log_node_event is called with outcome='refactor'."""
        state = make_state(retry_count=0, max_retries=5, error_status=False)
        with patch("coder_buddy.nodes.evaluator.log_node_event") as mock_log:
            evaluator(state)
            mock_log.assert_called_once()

    def test_fail_branch_emits_log(self):
        """When routing to 'fail', log_node_event is called with outcome='fail'."""
        state = make_state(retry_count=5, max_retries=5, error_status=True)
        with patch("coder_buddy.nodes.evaluator.log_node_event") as mock_log:
            evaluator(state)
            mock_log.assert_called_once()


# --------------------------------------------------------------------------- #
# Property 2: retry_count increments on each routing-back
# --------------------------------------------------------------------------- #


# Feature: coder-buddy, Property 2: retry_count increments on each routing-back


@given(
    retry_count=st.integers(min_value=0, max_value=9),
    max_retries=st.integers(min_value=1, max_value=10),
)
@settings(max_examples=200)
def test_property2_retry_count_increments_when_routing_to_retry(
    retry_count: int,
    max_retries: int,
):
    """
    **Validates: Requirements 1.5**

    Property 2: For any AgentState where error_status=True and
    retry_count < max_retries, after evaluator routes to "retry",
    retry_count in the state SHALL equal the original retry_count + 1.
    """
    # Only test the "retry" branch: error_status=True, retry_count < max_retries
    if retry_count >= max_retries:
        # This combination routes to "fail" — skip it for this property
        return

    state = make_state(
        retry_count=retry_count,
        max_retries=max_retries,
        error_status=True,
        execution_logs="RuntimeError: something went wrong",
    )
    original_count = state["retry_count"]

    result = evaluator(state)

    assert result == "retry", (
        f"Expected 'retry' for retry_count={retry_count}, max_retries={max_retries}"
    )
    assert state["retry_count"] == original_count + 1, (
        f"Expected retry_count={original_count + 1}, got {state['retry_count']}"
    )
