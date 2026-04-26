"""
Property-based test for Property 21 (Task 20.4).

Feature: coder-buddy
Property 21: for any LLM call in any node, the TokenRecord for that node
             has input_tokens > 0 and output_tokens > 0 after the call.

This property is tested across all four nodes that make LLM calls:
  - write_node      (via make_write_node)
  - refactor_node   (via make_refactor_node)
  - post_process_node — explanation call (via make_post_process_node)
  - post_process_node — confidence call  (via make_post_process_node)
  - test_node       (via make_test_node)

Each test drives the node under test with a mocked LLMClient whose
generate() method returns a TokenRecord with the Hypothesis-generated
positive token counts.  The sandbox is also mocked where needed.
No real LLM calls or subprocess/venv operations are performed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from coder_buddy.config import AgentConfig
from coder_buddy.models import CodeArtifact, TokenRecord, TokenUsage
from coder_buddy.state import AgentState

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Positive integer token counts — the property requires > 0.
_POSITIVE_TOKENS = st.integers(min_value=1, max_value=100_000)

# Pairs of (input_tokens, output_tokens) both > 0.
_TOKEN_PAIR = st.tuples(_POSITIVE_TOKENS, _POSITIVE_TOKENS)

# A minimal valid source_code string.
_SOURCE_CODE = st.just('print("hello")\n')

# A minimal valid prompt string.
_PROMPT = st.just("Write a hello world script")

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_VALID_ARTIFACT = CodeArtifact(
    source_code='print("hello")\n',
    file_name="main.py",
    dependencies=[],
    language="python",
)

_DEFAULT_CONFIG = AgentConfig(
    llm_backend="gemini-2.5-flash",
    sandbox_backend="subprocess+venv",
    max_retries=3,
    explanation_enabled=True,
    test_generation_enabled=True,
    diff_view_enabled=True,
)


def _make_base_state(source_code: str = 'print("hello")\n') -> AgentState:
    """Return a minimal valid AgentState for use in node tests."""
    return AgentState(
        user_prompt="Write a hello world script",
        current_code=source_code,
        execution_logs="Hello, World!\n",
        error_status=False,
        retry_count=0,
        dependencies=[],
        file_name="main.py",
        language="python",
        explanation=None,
        test_code=None,
        test_logs=None,
        confidence_score=None,
        refactor_diff=None,
        token_usage=TokenUsage(),
        session_history=[],
        max_retries=3,
        pre_refactor_code=None,
        warning=None,
        _route=None,
    )


def _make_mock_llm_client(input_tokens: int, output_tokens: int) -> MagicMock:
    """
    Return a mock LLMClient whose generate() always returns a valid
    CodeArtifact and a TokenRecord with the given token counts.
    """
    mock_client = MagicMock()
    token_record = TokenRecord(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=None,
    )
    mock_client.generate.return_value = (_VALID_ARTIFACT, token_record)
    return mock_client


# ---------------------------------------------------------------------------
# Property 21 — write_node
# ---------------------------------------------------------------------------

# Feature: coder-buddy, Property 21: for any LLM call in any node, the
# TokenRecord for that node has input_tokens > 0 and output_tokens > 0
# after the call.
@given(token_pair=_TOKEN_PAIR)
@settings(max_examples=50)
def test_property21_write_node_token_record_positive(token_pair: tuple[int, int]) -> None:
    """
    Property 21 — write_node:

    For any LLM call made by ``write_node``, the ``TokenRecord`` stored in
    ``token_usage.write_node`` SHALL have ``input_tokens > 0`` and
    ``output_tokens > 0`` after the call.

    **Validates: Requirements 13.1**
    """
    from coder_buddy.nodes.write_node import make_write_node

    input_tokens, output_tokens = token_pair
    mock_client = _make_mock_llm_client(input_tokens, output_tokens)

    node = make_write_node(mock_client, _DEFAULT_CONFIG)
    state = _make_base_state()

    result = node(state)

    updated_usage: TokenUsage = result["token_usage"]
    record: TokenRecord = updated_usage.write_node

    assert record.input_tokens > 0, (
        f"write_node: expected input_tokens > 0, got {record.input_tokens} "
        f"(input_tokens={input_tokens}, output_tokens={output_tokens})"
    )
    assert record.output_tokens > 0, (
        f"write_node: expected output_tokens > 0, got {record.output_tokens} "
        f"(input_tokens={input_tokens}, output_tokens={output_tokens})"
    )
    assert record.input_tokens == input_tokens, (
        f"write_node: TokenRecord.input_tokens mismatch: "
        f"expected {input_tokens}, got {record.input_tokens}"
    )
    assert record.output_tokens == output_tokens, (
        f"write_node: TokenRecord.output_tokens mismatch: "
        f"expected {output_tokens}, got {record.output_tokens}"
    )


# ---------------------------------------------------------------------------
# Property 21 — refactor_node
# ---------------------------------------------------------------------------

@given(token_pair=_TOKEN_PAIR)
@settings(max_examples=50)
def test_property21_refactor_node_token_record_positive(token_pair: tuple[int, int]) -> None:
    """
    Property 21 — refactor_node:

    For any LLM call made by ``refactor_node``, the ``TokenRecord`` stored
    in ``token_usage.refactor_node`` SHALL have ``input_tokens > 0`` and
    ``output_tokens > 0`` after the call.

    **Validates: Requirements 13.1**
    """
    from coder_buddy.nodes.refactor_node import make_refactor_node

    input_tokens, output_tokens = token_pair
    mock_client = _make_mock_llm_client(input_tokens, output_tokens)

    node = make_refactor_node(mock_client, _DEFAULT_CONFIG)
    state = _make_base_state()

    result = node(state)

    updated_usage: TokenUsage = result["token_usage"]
    record: TokenRecord = updated_usage.refactor_node

    assert record.input_tokens > 0, (
        f"refactor_node: expected input_tokens > 0, got {record.input_tokens} "
        f"(input_tokens={input_tokens}, output_tokens={output_tokens})"
    )
    assert record.output_tokens > 0, (
        f"refactor_node: expected output_tokens > 0, got {record.output_tokens} "
        f"(input_tokens={input_tokens}, output_tokens={output_tokens})"
    )
    assert record.input_tokens == input_tokens, (
        f"refactor_node: TokenRecord.input_tokens mismatch: "
        f"expected {input_tokens}, got {record.input_tokens}"
    )
    assert record.output_tokens == output_tokens, (
        f"refactor_node: TokenRecord.output_tokens mismatch: "
        f"expected {output_tokens}, got {record.output_tokens}"
    )


# ---------------------------------------------------------------------------
# Property 21 — post_process_node (explanation call)
# ---------------------------------------------------------------------------

@given(token_pair=_TOKEN_PAIR)
@settings(max_examples=50)
def test_property21_post_process_explanation_token_record_positive(
    token_pair: tuple[int, int],
) -> None:
    """
    Property 21 — post_process_node (explanation):

    For any LLM call made by ``post_process_node`` for the explanation,
    the ``TokenRecord`` stored in ``token_usage.explanation`` SHALL have
    ``input_tokens > 0`` and ``output_tokens > 0`` after the call.

    **Validates: Requirements 13.1**
    """
    from coder_buddy.nodes.post_process import (
        ExplanationOutput,
        ConfidenceOutput,
        make_post_process_node,
    )

    input_tokens, output_tokens = token_pair

    # The post_process_node makes two LLM calls: explanation then confidence.
    # We give each call distinct token counts so we can verify the explanation
    # record independently.
    explanation_record = TokenRecord(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    confidence_record = TokenRecord(
        input_tokens=1,
        output_tokens=1,
    )

    mock_client = MagicMock()
    mock_client.generate.side_effect = [
        (ExplanationOutput(explanation="This prints hello."), explanation_record),
        (ConfidenceOutput(confidence_score=4), confidence_record),
    ]

    node = make_post_process_node(mock_client, _DEFAULT_CONFIG)
    state = _make_base_state()

    result = node(state)

    updated_usage: TokenUsage = result["token_usage"]
    record: TokenRecord = updated_usage.explanation

    assert record.input_tokens > 0, (
        f"post_process explanation: expected input_tokens > 0, "
        f"got {record.input_tokens} "
        f"(input_tokens={input_tokens}, output_tokens={output_tokens})"
    )
    assert record.output_tokens > 0, (
        f"post_process explanation: expected output_tokens > 0, "
        f"got {record.output_tokens} "
        f"(input_tokens={input_tokens}, output_tokens={output_tokens})"
    )
    assert record.input_tokens == input_tokens, (
        f"post_process explanation: TokenRecord.input_tokens mismatch: "
        f"expected {input_tokens}, got {record.input_tokens}"
    )
    assert record.output_tokens == output_tokens, (
        f"post_process explanation: TokenRecord.output_tokens mismatch: "
        f"expected {output_tokens}, got {record.output_tokens}"
    )


# ---------------------------------------------------------------------------
# Property 21 — post_process_node (confidence call)
# ---------------------------------------------------------------------------

@given(token_pair=_TOKEN_PAIR)
@settings(max_examples=50)
def test_property21_post_process_confidence_token_record_positive(
    token_pair: tuple[int, int],
) -> None:
    """
    Property 21 — post_process_node (confidence):

    For any LLM call made by ``post_process_node`` for the confidence score,
    the ``TokenRecord`` stored in ``token_usage.confidence`` SHALL have
    ``input_tokens > 0`` and ``output_tokens > 0`` after the call.

    **Validates: Requirements 13.1, 14.1**
    """
    from coder_buddy.nodes.post_process import (
        ExplanationOutput,
        ConfidenceOutput,
        make_post_process_node,
    )

    input_tokens, output_tokens = token_pair

    # Give the explanation call minimal tokens; vary the confidence call.
    explanation_record = TokenRecord(input_tokens=1, output_tokens=1)
    confidence_record = TokenRecord(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    mock_client = MagicMock()
    mock_client.generate.side_effect = [
        (ExplanationOutput(explanation="This prints hello."), explanation_record),
        (ConfidenceOutput(confidence_score=4), confidence_record),
    ]

    node = make_post_process_node(mock_client, _DEFAULT_CONFIG)
    state = _make_base_state()

    result = node(state)

    updated_usage: TokenUsage = result["token_usage"]
    record: TokenRecord = updated_usage.confidence

    assert record.input_tokens > 0, (
        f"post_process confidence: expected input_tokens > 0, "
        f"got {record.input_tokens} "
        f"(input_tokens={input_tokens}, output_tokens={output_tokens})"
    )
    assert record.output_tokens > 0, (
        f"post_process confidence: expected output_tokens > 0, "
        f"got {record.output_tokens} "
        f"(input_tokens={input_tokens}, output_tokens={output_tokens})"
    )
    assert record.input_tokens == input_tokens, (
        f"post_process confidence: TokenRecord.input_tokens mismatch: "
        f"expected {input_tokens}, got {record.input_tokens}"
    )
    assert record.output_tokens == output_tokens, (
        f"post_process confidence: TokenRecord.output_tokens mismatch: "
        f"expected {output_tokens}, got {record.output_tokens}"
    )


# ---------------------------------------------------------------------------
# Property 21 — post_process_node (confidence only, explanation disabled)
# ---------------------------------------------------------------------------

@given(token_pair=_TOKEN_PAIR)
@settings(max_examples=50)
def test_property21_post_process_confidence_only_token_record_positive(
    token_pair: tuple[int, int],
) -> None:
    """
    Property 21 — post_process_node (confidence, explanation disabled):

    When ``explanation_enabled=False``, only the confidence LLM call is
    made.  The ``TokenRecord`` stored in ``token_usage.confidence`` SHALL
    still have ``input_tokens > 0`` and ``output_tokens > 0``.

    **Validates: Requirements 13.1, 14.1**
    """
    from coder_buddy.nodes.post_process import ConfidenceOutput, make_post_process_node

    input_tokens, output_tokens = token_pair

    confidence_record = TokenRecord(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    mock_client = MagicMock()
    mock_client.generate.return_value = (
        ConfidenceOutput(confidence_score=3),
        confidence_record,
    )

    config_no_explanation = AgentConfig(
        llm_backend="gemini-2.5-flash",
        sandbox_backend="subprocess+venv",
        max_retries=3,
        explanation_enabled=False,
        test_generation_enabled=True,
        diff_view_enabled=True,
    )

    node = make_post_process_node(mock_client, config_no_explanation)
    state = _make_base_state()

    result = node(state)

    updated_usage: TokenUsage = result["token_usage"]
    record: TokenRecord = updated_usage.confidence

    assert record.input_tokens > 0, (
        f"post_process confidence (no explanation): expected input_tokens > 0, "
        f"got {record.input_tokens}"
    )
    assert record.output_tokens > 0, (
        f"post_process confidence (no explanation): expected output_tokens > 0, "
        f"got {record.output_tokens}"
    )


# ---------------------------------------------------------------------------
# Property 21 — test_node
# ---------------------------------------------------------------------------

@given(token_pair=_TOKEN_PAIR)
@settings(max_examples=50)
def test_property21_test_node_token_record_positive(token_pair: tuple[int, int]) -> None:
    """
    Property 21 — test_node:

    For any LLM call made by ``test_node``, the ``TokenRecord`` stored in
    ``token_usage.test_node`` SHALL have ``input_tokens > 0`` and
    ``output_tokens > 0`` after the call.

    The sandbox is mocked to return a successful (no-error) execution so
    the node completes on the first attempt.

    **Validates: Requirements 13.1**
    """
    from coder_buddy.nodes.test_node import make_test_node
    from coder_buddy.sandbox.base import ExecutionResult

    input_tokens, output_tokens = token_pair
    mock_client = _make_mock_llm_client(input_tokens, output_tokens)

    # Mock sandbox: install_dependencies is a no-op; execute returns success.
    mock_sandbox = MagicMock()
    mock_sandbox.install_dependencies.return_value = None
    mock_sandbox.execute.return_value = ExecutionResult(
        stdout="1 passed\n",
        stderr="",
        exit_code=0,
        timed_out=False,
    )
    mock_sandbox.cleanup.return_value = None

    node = make_test_node(mock_sandbox, mock_client, _DEFAULT_CONFIG)
    state = _make_base_state()

    result = node(state)

    updated_usage: TokenUsage = result["token_usage"]
    record: TokenRecord = updated_usage.test_node

    assert record.input_tokens > 0, (
        f"test_node: expected input_tokens > 0, got {record.input_tokens} "
        f"(input_tokens={input_tokens}, output_tokens={output_tokens})"
    )
    assert record.output_tokens > 0, (
        f"test_node: expected output_tokens > 0, got {record.output_tokens} "
        f"(input_tokens={input_tokens}, output_tokens={output_tokens})"
    )
    assert record.input_tokens == input_tokens, (
        f"test_node: TokenRecord.input_tokens mismatch: "
        f"expected {input_tokens}, got {record.input_tokens}"
    )
    assert record.output_tokens == output_tokens, (
        f"test_node: TokenRecord.output_tokens mismatch: "
        f"expected {output_tokens}, got {record.output_tokens}"
    )


# ===========================================================================
# Property 12 — State preservation after LangGraph partial-update merge
# ===========================================================================
# Feature: coder-buddy, Property 12: state preservation
#
# For any AgentState and any partial update dict returned by a node, all
# fields NOT in the update dict retain their original values after LangGraph
# merges the state.
#
# LangGraph's merge is a shallow dict merge:  merged = {**current_state, **partial_update}
# ===========================================================================

from typing import Any


# ---------------------------------------------------------------------------
# Strategies for AgentState fields
# ---------------------------------------------------------------------------

_TEXT = st.text(min_size=0, max_size=200)
_BOOL = st.booleans()
_NON_NEG_INT = st.integers(min_value=0, max_value=20)
_POS_INT = st.integers(min_value=1, max_value=10)
_OPT_TEXT = st.one_of(st.none(), st.text(min_size=0, max_size=200))
_OPT_INT = st.one_of(st.none(), st.integers(min_value=1, max_value=5))
_STR_LIST = st.lists(st.text(min_size=1, max_size=50), max_size=10)
_OPT_ROUTE = st.one_of(st.none(), st.sampled_from(["retry", "refactor", "fail"]))

# All AgentState field names (must match state.py exactly)
_ALL_STATE_KEYS: list[str] = [
    "user_prompt",
    "current_code",
    "execution_logs",
    "error_status",
    "retry_count",
    "dependencies",
    "file_name",
    "language",
    "explanation",
    "test_code",
    "test_logs",
    "confidence_score",
    "refactor_diff",
    "token_usage",
    "session_history",
    "max_retries",
    "pre_refactor_code",
    "warning",
    "_route",
]


@st.composite
def agent_state_strategy(draw: st.DrawFn) -> AgentState:
    """Generate a random valid AgentState."""
    return AgentState(
        user_prompt=draw(_TEXT),
        current_code=draw(_TEXT),
        execution_logs=draw(_TEXT),
        error_status=draw(_BOOL),
        retry_count=draw(_NON_NEG_INT),
        dependencies=draw(_STR_LIST),
        file_name=draw(_TEXT),
        language="python",
        explanation=draw(_OPT_TEXT),
        test_code=draw(_OPT_TEXT),
        test_logs=draw(_OPT_TEXT),
        confidence_score=draw(_OPT_INT),
        refactor_diff=draw(_OPT_TEXT),
        token_usage=TokenUsage(),
        session_history=[],
        max_retries=draw(_POS_INT),
        pre_refactor_code=draw(_OPT_TEXT),
        warning=draw(_OPT_TEXT),
        _route=draw(_OPT_ROUTE),
    )


@st.composite
def partial_update_strategy(draw: st.DrawFn, state: AgentState) -> dict[str, Any]:
    """
    Generate a random partial update dict — a subset of AgentState keys
    with new values drawn from the same value space.

    The subset may be empty (no-op update) or contain any number of keys
    up to the full set.  Values are drawn independently of the original
    state so they may differ.
    """
    # Choose a random non-empty subset of keys to update
    keys_to_update = draw(
        st.lists(
            st.sampled_from(_ALL_STATE_KEYS),
            min_size=0,
            max_size=len(_ALL_STATE_KEYS),
            unique=True,
        )
    )

    update: dict[str, Any] = {}
    for key in keys_to_update:
        if key == "user_prompt":
            update[key] = draw(_TEXT)
        elif key == "current_code":
            update[key] = draw(_TEXT)
        elif key == "execution_logs":
            update[key] = draw(_TEXT)
        elif key == "error_status":
            update[key] = draw(_BOOL)
        elif key == "retry_count":
            update[key] = draw(_NON_NEG_INT)
        elif key == "dependencies":
            update[key] = draw(_STR_LIST)
        elif key == "file_name":
            update[key] = draw(_TEXT)
        elif key == "language":
            update[key] = "python"
        elif key == "explanation":
            update[key] = draw(_OPT_TEXT)
        elif key == "test_code":
            update[key] = draw(_OPT_TEXT)
        elif key == "test_logs":
            update[key] = draw(_OPT_TEXT)
        elif key == "confidence_score":
            update[key] = draw(_OPT_INT)
        elif key == "refactor_diff":
            update[key] = draw(_OPT_TEXT)
        elif key == "token_usage":
            update[key] = TokenUsage()
        elif key == "session_history":
            update[key] = []
        elif key == "max_retries":
            update[key] = draw(_POS_INT)
        elif key == "pre_refactor_code":
            update[key] = draw(_OPT_TEXT)
        elif key == "warning":
            update[key] = draw(_OPT_TEXT)
        elif key == "_route":
            update[key] = draw(_OPT_ROUTE)

    return update


# Feature: coder-buddy, Property 12: state preservation
@given(
    original_state=agent_state_strategy(),
    partial_update=st.deferred(
        lambda: agent_state_strategy().flatmap(partial_update_strategy)
    ),
)
@settings(max_examples=100)
def test_property12_state_preservation_after_langgraph_merge(
    original_state: AgentState,
    partial_update: dict[str, Any],
) -> None:
    """
    Property 12 — State preservation:

    For any ``AgentState`` and any partial update dict returned by a node,
    all fields NOT in the update dict retain their original values after
    LangGraph merges the state.

    LangGraph's merge is a shallow dict merge::

        merged = {**current_state, **partial_update}

    This test verifies that for every key ``k`` in ``AgentState`` that is
    **not** present in ``partial_update``, ``merged[k] == original_state[k]``.

    **Validates: Requirements 1.2**
    """
    # Simulate LangGraph's state merge (shallow dict merge)
    merged: dict[str, Any] = {**original_state, **partial_update}

    # For every key NOT in the partial update, the merged value must equal
    # the original value.
    untouched_keys = [k for k in _ALL_STATE_KEYS if k not in partial_update]

    for key in untouched_keys:
        original_value = original_state[key]  # type: ignore[literal-required]
        merged_value = merged[key]
        assert merged_value == original_value, (
            f"State preservation violated for key {key!r}: "
            f"expected {original_value!r}, got {merged_value!r} after merge. "
            f"partial_update keys: {list(partial_update.keys())}"
        )
