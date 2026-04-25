"""
Unit tests for post_process_node — confidence score validation.

Validates the confidence score clamping and warning behaviour:
- Scores outside [1, 5] are clamped to the nearest boundary.
- A warning is included in the result when confidence_score <= 2.
- No warning is included when confidence_score >= 3.
- Explanation is populated when explanation_enabled=True.
- Explanation is None when explanation_enabled=False.

Also covers:
- Property 24: any confidence_score outside [1, 5] is clamped; stored value
  always satisfies 1 <= confidence_score <= 5.
- Property 25: for any confidence_score in {1, 2}, AgentResponse.warning is
  a non-None, non-empty string.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from coder_buddy.models import TokenRecord, TokenUsage
from coder_buddy.nodes.post_process import (
    ConfidenceOutput,
    ExplanationOutput,
    make_post_process_node,
)


# --------------------------------------------------------------------------- #
# Helpers / fixtures
# --------------------------------------------------------------------------- #


def _make_state(
    *,
    current_code: str = "print('hello')\n",
    file_name: str = "main.py",
    retry_count: int = 0,
    execution_logs: str = "",
    token_usage: TokenUsage | None = None,
) -> dict:
    """Build a minimal AgentState dict for post_process_node testing."""
    return {
        "user_prompt": "Write a hello world script",
        "current_code": current_code,
        "execution_logs": execution_logs,
        "error_status": False,
        "retry_count": retry_count,
        "dependencies": [],
        "file_name": file_name,
        "language": "python",
        "explanation": None,
        "test_code": None,
        "test_logs": None,
        "confidence_score": None,
        "refactor_diff": None,
        "token_usage": token_usage if token_usage is not None else TokenUsage(),
        "session_history": [],
        "max_retries": 5,
        "pre_refactor_code": None,
    }


def _make_mock_llm_client(
    *,
    confidence_score: int = 4,
    explanation_text: str = "This script prints hello world.",
) -> MagicMock:
    """
    Return a mock LLMClient whose generate() returns appropriate structured
    output based on the requested output_type.
    """
    explanation_output = ExplanationOutput(explanation=explanation_text)
    confidence_output = ConfidenceOutput(confidence_score=confidence_score)
    explanation_token = TokenRecord(input_tokens=100, output_tokens=50)
    confidence_token = TokenRecord(input_tokens=80, output_tokens=10)

    def _generate(prompt: str, output_type: type, **kwargs):
        if output_type is ExplanationOutput:
            return (explanation_output, explanation_token)
        if output_type is ConfidenceOutput:
            return (confidence_output, confidence_token)
        raise ValueError(f"Unexpected output_type: {output_type}")

    mock_client = MagicMock()
    mock_client.generate.side_effect = _generate
    return mock_client


def _make_mock_config(*, explanation_enabled: bool = True) -> MagicMock:
    """Return a mock AgentConfig with configurable explanation_enabled."""
    config = MagicMock()
    config.explanation_enabled = explanation_enabled
    return config


# --------------------------------------------------------------------------- #
# Explanation behaviour
# --------------------------------------------------------------------------- #


class TestExplanationBehaviour:
    """Explanation is populated when enabled and None when disabled."""

    def test_explanation_populated_when_enabled(self):
        """When explanation_enabled=True, explanation is a non-empty string."""
        mock_client = _make_mock_llm_client(explanation_text="Prints hello world.")
        config = _make_mock_config(explanation_enabled=True)
        node = make_post_process_node(mock_client, config)
        state = _make_state()

        with patch("coder_buddy.nodes.post_process.log_node_event"):
            result = node(state)

        assert result["explanation"] is not None
        assert isinstance(result["explanation"], str)
        assert len(result["explanation"]) > 0

    def test_explanation_is_none_when_disabled(self):
        """When explanation_enabled=False, explanation is None."""
        mock_client = _make_mock_llm_client()
        config = _make_mock_config(explanation_enabled=False)
        node = make_post_process_node(mock_client, config)
        state = _make_state()

        with patch("coder_buddy.nodes.post_process.log_node_event"):
            result = node(state)

        assert result["explanation"] is None

    def test_explanation_llm_not_called_when_disabled(self):
        """When explanation_enabled=False, the LLM is called only once (confidence)."""
        mock_client = _make_mock_llm_client()
        config = _make_mock_config(explanation_enabled=False)
        node = make_post_process_node(mock_client, config)
        state = _make_state()

        with patch("coder_buddy.nodes.post_process.log_node_event"):
            node(state)

        # Only the confidence call should be made
        assert mock_client.generate.call_count == 1

    def test_explanation_llm_called_twice_when_enabled(self):
        """When explanation_enabled=True, the LLM is called twice (explanation + confidence)."""
        mock_client = _make_mock_llm_client()
        config = _make_mock_config(explanation_enabled=True)
        node = make_post_process_node(mock_client, config)
        state = _make_state()

        with patch("coder_buddy.nodes.post_process.log_node_event"):
            node(state)

        assert mock_client.generate.call_count == 2

    def test_explanation_text_matches_llm_output(self):
        """The explanation in the result matches the text returned by the LLM."""
        expected_text = "This script computes the Fibonacci sequence."
        mock_client = _make_mock_llm_client(explanation_text=expected_text)
        config = _make_mock_config(explanation_enabled=True)
        node = make_post_process_node(mock_client, config)
        state = _make_state()

        with patch("coder_buddy.nodes.post_process.log_node_event"):
            result = node(state)

        assert result["explanation"] == expected_text


# --------------------------------------------------------------------------- #
# Confidence score clamping
# --------------------------------------------------------------------------- #


class TestConfidenceScoreClamping:
    """Confidence scores outside [1, 5] are clamped to the nearest boundary."""

    def test_score_within_range_unchanged(self):
        """A score of 3 (within [1, 5]) is returned unchanged."""
        mock_client = _make_mock_llm_client(confidence_score=3)
        config = _make_mock_config(explanation_enabled=False)
        node = make_post_process_node(mock_client, config)
        state = _make_state()

        with patch("coder_buddy.nodes.post_process.log_node_event"):
            result = node(state)

        assert result["confidence_score"] == 3

    def test_score_of_1_unchanged(self):
        """A score of 1 (lower boundary) is returned unchanged."""
        mock_client = _make_mock_llm_client(confidence_score=1)
        config = _make_mock_config(explanation_enabled=False)
        node = make_post_process_node(mock_client, config)
        state = _make_state()

        with patch("coder_buddy.nodes.post_process.log_node_event"):
            result = node(state)

        assert result["confidence_score"] == 1

    def test_score_of_5_unchanged(self):
        """A score of 5 (upper boundary) is returned unchanged."""
        mock_client = _make_mock_llm_client(confidence_score=5)
        config = _make_mock_config(explanation_enabled=False)
        node = make_post_process_node(mock_client, config)
        state = _make_state()

        with patch("coder_buddy.nodes.post_process.log_node_event"):
            result = node(state)

        assert result["confidence_score"] == 5

    def test_score_below_1_clamped_to_1(self):
        """A score of 0 (below lower boundary) is clamped to 1."""
        mock_client = _make_mock_llm_client(confidence_score=0)
        config = _make_mock_config(explanation_enabled=False)
        node = make_post_process_node(mock_client, config)
        state = _make_state()

        with patch("coder_buddy.nodes.post_process.log_node_event"):
            result = node(state)

        assert result["confidence_score"] == 1

    def test_negative_score_clamped_to_1(self):
        """A negative score is clamped to 1."""
        mock_client = _make_mock_llm_client(confidence_score=-10)
        config = _make_mock_config(explanation_enabled=False)
        node = make_post_process_node(mock_client, config)
        state = _make_state()

        with patch("coder_buddy.nodes.post_process.log_node_event"):
            result = node(state)

        assert result["confidence_score"] == 1

    def test_score_above_5_clamped_to_5(self):
        """A score of 6 (above upper boundary) is clamped to 5."""
        mock_client = _make_mock_llm_client(confidence_score=6)
        config = _make_mock_config(explanation_enabled=False)
        node = make_post_process_node(mock_client, config)
        state = _make_state()

        with patch("coder_buddy.nodes.post_process.log_node_event"):
            result = node(state)

        assert result["confidence_score"] == 5

    def test_very_large_score_clamped_to_5(self):
        """A very large score is clamped to 5."""
        mock_client = _make_mock_llm_client(confidence_score=100)
        config = _make_mock_config(explanation_enabled=False)
        node = make_post_process_node(mock_client, config)
        state = _make_state()

        with patch("coder_buddy.nodes.post_process.log_node_event"):
            result = node(state)

        assert result["confidence_score"] == 5


# --------------------------------------------------------------------------- #
# Warning behaviour
# --------------------------------------------------------------------------- #


class TestConfidenceScoreWarning:
    """Warning is present when confidence_score <= 2 and absent when >= 3."""

    def test_warning_present_when_score_is_1(self):
        """confidence_score=1 → warning is a non-None, non-empty string."""
        mock_client = _make_mock_llm_client(confidence_score=1)
        config = _make_mock_config(explanation_enabled=False)
        node = make_post_process_node(mock_client, config)
        state = _make_state()

        with patch("coder_buddy.nodes.post_process.log_node_event"):
            result = node(state)

        assert result.get("warning") is not None
        assert isinstance(result["warning"], str)
        assert len(result["warning"]) > 0

    def test_warning_present_when_score_is_2(self):
        """confidence_score=2 → warning is a non-None, non-empty string."""
        mock_client = _make_mock_llm_client(confidence_score=2)
        config = _make_mock_config(explanation_enabled=False)
        node = make_post_process_node(mock_client, config)
        state = _make_state()

        with patch("coder_buddy.nodes.post_process.log_node_event"):
            result = node(state)

        assert result.get("warning") is not None
        assert isinstance(result["warning"], str)
        assert len(result["warning"]) > 0

    def test_no_warning_when_score_is_3(self):
        """confidence_score=3 → no warning key in result (or warning is None/absent)."""
        mock_client = _make_mock_llm_client(confidence_score=3)
        config = _make_mock_config(explanation_enabled=False)
        node = make_post_process_node(mock_client, config)
        state = _make_state()

        with patch("coder_buddy.nodes.post_process.log_node_event"):
            result = node(state)

        assert result.get("warning") is None

    def test_no_warning_when_score_is_4(self):
        """confidence_score=4 → no warning in result."""
        mock_client = _make_mock_llm_client(confidence_score=4)
        config = _make_mock_config(explanation_enabled=False)
        node = make_post_process_node(mock_client, config)
        state = _make_state()

        with patch("coder_buddy.nodes.post_process.log_node_event"):
            result = node(state)

        assert result.get("warning") is None

    def test_no_warning_when_score_is_5(self):
        """confidence_score=5 → no warning in result."""
        mock_client = _make_mock_llm_client(confidence_score=5)
        config = _make_mock_config(explanation_enabled=False)
        node = make_post_process_node(mock_client, config)
        state = _make_state()

        with patch("coder_buddy.nodes.post_process.log_node_event"):
            result = node(state)

        assert result.get("warning") is None

    def test_warning_mentions_score(self):
        """The warning message references the actual confidence score value."""
        mock_client = _make_mock_llm_client(confidence_score=1)
        config = _make_mock_config(explanation_enabled=False)
        node = make_post_process_node(mock_client, config)
        state = _make_state()

        with patch("coder_buddy.nodes.post_process.log_node_event"):
            result = node(state)

        # The warning should mention the score so the user knows why they're warned
        assert "1" in result["warning"]

    def test_warning_present_when_out_of_range_score_clamped_to_1(self):
        """A score of -5 is clamped to 1, which triggers the warning."""
        mock_client = _make_mock_llm_client(confidence_score=-5)
        config = _make_mock_config(explanation_enabled=False)
        node = make_post_process_node(mock_client, config)
        state = _make_state()

        with patch("coder_buddy.nodes.post_process.log_node_event"):
            result = node(state)

        assert result["confidence_score"] == 1
        assert result.get("warning") is not None
        assert len(result["warning"]) > 0


# --------------------------------------------------------------------------- #
# Result structure
# --------------------------------------------------------------------------- #


class TestResultStructure:
    """The result dict always contains the required keys."""

    def test_result_contains_explanation_key(self):
        """Result always contains 'explanation' key."""
        mock_client = _make_mock_llm_client()
        config = _make_mock_config(explanation_enabled=False)
        node = make_post_process_node(mock_client, config)
        state = _make_state()

        with patch("coder_buddy.nodes.post_process.log_node_event"):
            result = node(state)

        assert "explanation" in result

    def test_result_contains_confidence_score_key(self):
        """Result always contains 'confidence_score' key."""
        mock_client = _make_mock_llm_client()
        config = _make_mock_config(explanation_enabled=False)
        node = make_post_process_node(mock_client, config)
        state = _make_state()

        with patch("coder_buddy.nodes.post_process.log_node_event"):
            result = node(state)

        assert "confidence_score" in result

    def test_result_contains_token_usage_key(self):
        """Result always contains 'token_usage' key."""
        mock_client = _make_mock_llm_client()
        config = _make_mock_config(explanation_enabled=False)
        node = make_post_process_node(mock_client, config)
        state = _make_state()

        with patch("coder_buddy.nodes.post_process.log_node_event"):
            result = node(state)

        assert "token_usage" in result

    def test_token_usage_updated_with_confidence_record(self):
        """token_usage in the result has the confidence record updated."""
        mock_client = _make_mock_llm_client(confidence_score=4)
        config = _make_mock_config(explanation_enabled=False)
        node = make_post_process_node(mock_client, config)
        state = _make_state()

        with patch("coder_buddy.nodes.post_process.log_node_event"):
            result = node(state)

        assert result["token_usage"].confidence.input_tokens == 80
        assert result["token_usage"].confidence.output_tokens == 10

    def test_token_usage_updated_with_explanation_record_when_enabled(self):
        """When explanation is enabled, token_usage has the explanation record updated."""
        mock_client = _make_mock_llm_client(confidence_score=4)
        config = _make_mock_config(explanation_enabled=True)
        node = make_post_process_node(mock_client, config)
        state = _make_state()

        with patch("coder_buddy.nodes.post_process.log_node_event"):
            result = node(state)

        assert result["token_usage"].explanation.input_tokens == 100
        assert result["token_usage"].explanation.output_tokens == 50


# --------------------------------------------------------------------------- #
# Property 24: confidence_score always in [1, 5] after clamping
# --------------------------------------------------------------------------- #

# Feature: coder-buddy, Property 24: any confidence_score outside [1, 5] is
# rejected or clamped; stored value always satisfies 1 <= confidence_score <= 5


class TestProperty24ConfidenceScoreAlwaysInRange:
    """
    **Validates: Requirements 14.2**

    Property 24: For any integer returned by the LLM as confidence_score,
    the value stored in the result SHALL satisfy 1 <= confidence_score <= 5.
    """

    @settings(max_examples=200)
    @given(raw_score=st.integers(min_value=-1000, max_value=1000))
    def test_confidence_score_always_in_range_after_clamping(self, raw_score: int):
        """
        For any integer raw_score, the stored confidence_score is always
        in the range [1, 5] after clamping.
        """
        mock_client = _make_mock_llm_client(confidence_score=raw_score)
        config = _make_mock_config(explanation_enabled=False)
        node = make_post_process_node(mock_client, config)
        state = _make_state()

        with patch("coder_buddy.nodes.post_process.log_node_event"):
            result = node(state)

        score = result["confidence_score"]
        assert isinstance(score, int), f"Expected int, got {type(score)}"
        assert 1 <= score <= 5, (
            f"confidence_score={score} is outside [1, 5] for raw_score={raw_score}"
        )


# --------------------------------------------------------------------------- #
# Property 25: warning is non-None and non-empty for scores in {1, 2}
# --------------------------------------------------------------------------- #

# Feature: coder-buddy, Property 25: for any confidence_score in {1, 2},
# AgentResponse.warning is a non-None, non-empty string


class TestProperty25LowScoreAlwaysHasWarning:
    """
    **Validates: Requirements 14.3**

    Property 25: For any confidence_score in {1, 2} (after clamping),
    the result SHALL contain a non-None, non-empty warning string.
    """

    @settings(max_examples=200)
    @given(raw_score=st.integers(min_value=-1000, max_value=2))
    def test_low_score_always_produces_warning(self, raw_score: int):
        """
        For any raw_score that clamps to 1 or 2, the result contains a
        non-None, non-empty warning string.
        """
        mock_client = _make_mock_llm_client(confidence_score=raw_score)
        config = _make_mock_config(explanation_enabled=False)
        node = make_post_process_node(mock_client, config)
        state = _make_state()

        with patch("coder_buddy.nodes.post_process.log_node_event"):
            result = node(state)

        # After clamping, score must be 1 or 2 (since raw_score <= 2)
        clamped_score = result["confidence_score"]
        assert clamped_score in (1, 2), (
            f"Expected clamped score in {{1, 2}}, got {clamped_score}"
        )

        warning = result.get("warning")
        assert warning is not None, (
            f"Expected non-None warning for confidence_score={clamped_score}"
        )
        assert isinstance(warning, str), (
            f"Expected warning to be a string, got {type(warning)}"
        )
        assert len(warning) > 0, (
            f"Expected non-empty warning for confidence_score={clamped_score}"
        )

    @settings(max_examples=200)
    @given(raw_score=st.integers(min_value=3, max_value=1000))
    def test_high_score_never_produces_warning(self, raw_score: int):
        """
        For any raw_score that clamps to 3, 4, or 5, the result SHALL NOT
        contain a warning.
        """
        mock_client = _make_mock_llm_client(confidence_score=raw_score)
        config = _make_mock_config(explanation_enabled=False)
        node = make_post_process_node(mock_client, config)
        state = _make_state()

        with patch("coder_buddy.nodes.post_process.log_node_event"):
            result = node(state)

        # After clamping, score must be 3, 4, or 5 (since raw_score >= 3)
        clamped_score = result["confidence_score"]
        assert clamped_score in (3, 4, 5), (
            f"Expected clamped score in {{3, 4, 5}}, got {clamped_score}"
        )

        warning = result.get("warning")
        assert warning is None, (
            f"Expected no warning for confidence_score={clamped_score}, got: {warning!r}"
        )
