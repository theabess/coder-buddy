"""
Unit tests for LLMClient.generate.

Validates:
- Successful parse: LLM returns valid CodeArtifact JSON on first call
- Parse retry (1 failure then success): mock LLM returns invalid JSON once, then valid
- Parse retry (2 failures then success): mock LLM returns invalid JSON twice, then valid
- ParseError after 3 failures: mock LLM always returns invalid JSON, ParseError raised
- LLMUnavailableError on HTTP error: mock LLM raises ModelHTTPError
- LLMUnavailableError on API error: mock LLM raises ModelAPIError
- LLMUnavailableError on network/connection error
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, call
import pytest

from coder_buddy.config import LLMUnavailableError, ParseError
from coder_buddy.llm.client import LLMClient
from coder_buddy.models import CodeArtifact, TokenRecord


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

VALID_ARTIFACT = CodeArtifact(
    source_code="print('hello')",
    file_name="hello.py",
    dependencies=[],
    language="python",
)

VALID_PROMPT = "Write a hello world script"


def _make_run_usage(input_tokens: int = 100, output_tokens: int = 50) -> MagicMock:
    """Return a mock RunUsage object."""
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    return usage


def _make_run_result(artifact: CodeArtifact, input_tokens: int = 100, output_tokens: int = 50) -> MagicMock:
    """Return a mock result object as returned by agent.run_sync()."""
    result = MagicMock()
    result.output = artifact
    result.usage.return_value = _make_run_usage(input_tokens, output_tokens)
    return result


# ---------------------------------------------------------------------------
# Test: Successful parse on first call
# ---------------------------------------------------------------------------


class TestSuccessfulParse:
    def test_returns_artifact_and_token_record(self):
        """LLM returns valid CodeArtifact on first call — returns (artifact, TokenRecord)."""
        mock_result = _make_run_result(VALID_ARTIFACT, input_tokens=200, output_tokens=80)

        with patch("coder_buddy.llm.client.Agent") as MockAgent:
            mock_agent_instance = MagicMock()
            MockAgent.return_value = mock_agent_instance
            mock_agent_instance.run_sync.return_value = mock_result

            client = LLMClient(model="gemini-2.5-flash")
            artifact, token_record = client.generate(VALID_PROMPT, CodeArtifact)

        assert artifact is VALID_ARTIFACT
        assert isinstance(token_record, TokenRecord)
        assert token_record.input_tokens == 200
        assert token_record.output_tokens == 80

    def test_agent_created_with_correct_model_string(self):
        """Agent is constructed with the pydantic-ai model string for gemini-1.5-pro."""
        mock_result = _make_run_result(VALID_ARTIFACT)

        with patch("coder_buddy.llm.client.Agent") as MockAgent:
            mock_agent_instance = MagicMock()
            MockAgent.return_value = mock_agent_instance
            mock_agent_instance.run_sync.return_value = mock_result

            client = LLMClient(model="gemini-2.5-flash")
            client.generate(VALID_PROMPT, CodeArtifact, max_retries=3)

        MockAgent.assert_called_once_with(
            model="google-gla:gemini-2.5-flash",
            output_type=CodeArtifact,
            retries=3,
        )

    def test_agent_created_with_gpt4o_model_string(self):
        """Agent is constructed with the pydantic-ai model string for gpt-4o."""
        mock_result = _make_run_result(VALID_ARTIFACT)

        with patch("coder_buddy.llm.client.Agent") as MockAgent:
            mock_agent_instance = MagicMock()
            MockAgent.return_value = mock_agent_instance
            mock_agent_instance.run_sync.return_value = mock_result

            client = LLMClient(model="gpt-4o")
            client.generate(VALID_PROMPT, CodeArtifact)

        MockAgent.assert_called_once_with(
            model="openai:gpt-4o",
            output_type=CodeArtifact,
            retries=3,
        )

    def test_token_record_has_estimated_cost(self):
        """TokenRecord includes estimated_cost_usd when model is in KNOWN_PRICES."""
        mock_result = _make_run_result(VALID_ARTIFACT, input_tokens=1000, output_tokens=500)

        with patch("coder_buddy.llm.client.Agent") as MockAgent:
            mock_agent_instance = MagicMock()
            MockAgent.return_value = mock_agent_instance
            mock_agent_instance.run_sync.return_value = mock_result

            client = LLMClient(model="gemini-2.5-flash")
            _, token_record = client.generate(VALID_PROMPT, CodeArtifact)

        # gemini-2.5-flash: 0.00015/1k input, 0.0006/1k output
        # cost = (1000 * 0.00015/1000) + (500 * 0.0006/1000) = 0.00015 + 0.0003 = 0.00045
        assert token_record.estimated_cost_usd == pytest.approx(0.00045)

    def test_run_sync_called_with_prompt(self):
        """agent.run_sync is called with the provided prompt."""
        mock_result = _make_run_result(VALID_ARTIFACT)

        with patch("coder_buddy.llm.client.Agent") as MockAgent:
            mock_agent_instance = MagicMock()
            MockAgent.return_value = mock_agent_instance
            mock_agent_instance.run_sync.return_value = mock_result

            client = LLMClient(model="gemini-2.5-flash")
            client.generate(VALID_PROMPT, CodeArtifact)

        mock_agent_instance.run_sync.assert_called_once_with(VALID_PROMPT)

    def test_zero_tokens_handled_gracefully(self):
        """When usage returns None/0 tokens, TokenRecord defaults to 0."""
        mock_result = MagicMock()
        mock_result.output = VALID_ARTIFACT
        usage = MagicMock()
        usage.input_tokens = None
        usage.output_tokens = None
        mock_result.usage.return_value = usage

        with patch("coder_buddy.llm.client.Agent") as MockAgent:
            mock_agent_instance = MagicMock()
            MockAgent.return_value = mock_agent_instance
            mock_agent_instance.run_sync.return_value = mock_result

            client = LLMClient(model="gemini-2.5-flash")
            _, token_record = client.generate(VALID_PROMPT, CodeArtifact)

        assert token_record.input_tokens == 0
        assert token_record.output_tokens == 0


# ---------------------------------------------------------------------------
# Test: ParseError after exhausting retries
# ---------------------------------------------------------------------------


class TestParseErrorAfterRetries:
    def test_parse_error_raised_after_3_failures(self):
        """UnexpectedModelBehavior from pydantic-ai is converted to ParseError."""
        from pydantic_ai.exceptions import UnexpectedModelBehavior

        with patch("coder_buddy.llm.client.Agent") as MockAgent:
            mock_agent_instance = MagicMock()
            MockAgent.return_value = mock_agent_instance
            mock_agent_instance.run_sync.side_effect = UnexpectedModelBehavior(
                "Model failed to produce valid output after 3 retries"
            )

            client = LLMClient(model="gemini-2.5-flash")
            with pytest.raises(ParseError) as exc_info:
                client.generate(VALID_PROMPT, CodeArtifact, max_retries=3)

        assert "CodeArtifact" in str(exc_info.value)
        assert "3" in str(exc_info.value)

    def test_parse_error_message_includes_output_type_name(self):
        """ParseError message includes the output type name."""
        from pydantic_ai.exceptions import UnexpectedModelBehavior

        with patch("coder_buddy.llm.client.Agent") as MockAgent:
            mock_agent_instance = MagicMock()
            MockAgent.return_value = mock_agent_instance
            mock_agent_instance.run_sync.side_effect = UnexpectedModelBehavior("bad output")

            client = LLMClient(model="gpt-4o")
            with pytest.raises(ParseError, match="CodeArtifact"):
                client.generate(VALID_PROMPT, CodeArtifact, max_retries=3)

    def test_parse_error_chained_from_unexpected_model_behavior(self):
        """ParseError is chained from the original UnexpectedModelBehavior."""
        from pydantic_ai.exceptions import UnexpectedModelBehavior

        original_exc = UnexpectedModelBehavior("parse failed")

        with patch("coder_buddy.llm.client.Agent") as MockAgent:
            mock_agent_instance = MagicMock()
            MockAgent.return_value = mock_agent_instance
            mock_agent_instance.run_sync.side_effect = original_exc

            client = LLMClient(model="gemini-2.5-flash")
            with pytest.raises(ParseError) as exc_info:
                client.generate(VALID_PROMPT, CodeArtifact)

        assert exc_info.value.__cause__ is original_exc

    def test_agent_created_with_max_retries_passed_through(self):
        """The max_retries parameter is forwarded to Agent as retries=."""
        from pydantic_ai.exceptions import UnexpectedModelBehavior

        with patch("coder_buddy.llm.client.Agent") as MockAgent:
            mock_agent_instance = MagicMock()
            MockAgent.return_value = mock_agent_instance
            mock_agent_instance.run_sync.side_effect = UnexpectedModelBehavior("fail")

            client = LLMClient(model="gemini-2.5-flash")
            with pytest.raises(ParseError):
                client.generate(VALID_PROMPT, CodeArtifact, max_retries=5)

        MockAgent.assert_called_once_with(
            model="google-gla:gemini-2.5-flash",
            output_type=CodeArtifact,
            retries=5,
        )


# ---------------------------------------------------------------------------
# Test: Parse retry — pydantic-ai handles retries internally
# ---------------------------------------------------------------------------


class TestParseRetryBehavior:
    """
    pydantic-ai's Agent handles parse retries internally (via the retries= param).
    From LLMClient's perspective, a successful run after retries looks identical
    to a first-attempt success — run_sync returns a valid result.
    We verify that LLMClient correctly passes max_retries to Agent and that
    a successful result (after internal retries) is handled correctly.
    """

    def test_retry_1_failure_then_success(self):
        """
        Simulate 1 internal parse failure then success:
        run_sync is called once and returns a valid result (pydantic-ai
        handles the retry internally before returning).
        """
        mock_result = _make_run_result(VALID_ARTIFACT, input_tokens=150, output_tokens=60)

        with patch("coder_buddy.llm.client.Agent") as MockAgent:
            mock_agent_instance = MagicMock()
            MockAgent.return_value = mock_agent_instance
            # After 1 internal retry, pydantic-ai returns a valid result
            mock_agent_instance.run_sync.return_value = mock_result

            client = LLMClient(model="gemini-2.5-flash")
            artifact, token_record = client.generate(VALID_PROMPT, CodeArtifact, max_retries=3)

        assert artifact is VALID_ARTIFACT
        assert token_record.input_tokens == 150
        assert token_record.output_tokens == 60
        # Agent was created with retries=3 so pydantic-ai can retry up to 3 times
        MockAgent.assert_called_once_with(
            model="google-gla:gemini-2.5-flash",
            output_type=CodeArtifact,
            retries=3,
        )

    def test_retry_2_failures_then_success(self):
        """
        Simulate 2 internal parse failures then success:
        run_sync returns a valid result after pydantic-ai's internal retries.
        """
        mock_result = _make_run_result(VALID_ARTIFACT, input_tokens=300, output_tokens=120)

        with patch("coder_buddy.llm.client.Agent") as MockAgent:
            mock_agent_instance = MagicMock()
            MockAgent.return_value = mock_agent_instance
            mock_agent_instance.run_sync.return_value = mock_result

            client = LLMClient(model="claude-3-5-sonnet")
            artifact, token_record = client.generate(VALID_PROMPT, CodeArtifact, max_retries=3)

        assert artifact is VALID_ARTIFACT
        assert token_record.input_tokens == 300
        # Agent was created with retries=3 allowing up to 3 internal retries
        MockAgent.assert_called_once_with(
            model="anthropic:claude-3-5-sonnet-latest",
            output_type=CodeArtifact,
            retries=3,
        )

    def test_max_retries_1_raises_parse_error_on_first_failure(self):
        """With max_retries=1, a single parse failure raises ParseError."""
        from pydantic_ai.exceptions import UnexpectedModelBehavior

        with patch("coder_buddy.llm.client.Agent") as MockAgent:
            mock_agent_instance = MagicMock()
            MockAgent.return_value = mock_agent_instance
            mock_agent_instance.run_sync.side_effect = UnexpectedModelBehavior("fail")

            client = LLMClient(model="gemini-2.5-flash")
            with pytest.raises(ParseError):
                client.generate(VALID_PROMPT, CodeArtifact, max_retries=1)

        MockAgent.assert_called_once_with(
            model="google-gla:gemini-2.5-flash",
            output_type=CodeArtifact,
            retries=1,
        )

    def test_max_retries_2_raises_parse_error_after_2_failures(self):
        """With max_retries=2, exhausting 2 retries raises ParseError."""
        from pydantic_ai.exceptions import UnexpectedModelBehavior

        with patch("coder_buddy.llm.client.Agent") as MockAgent:
            mock_agent_instance = MagicMock()
            MockAgent.return_value = mock_agent_instance
            mock_agent_instance.run_sync.side_effect = UnexpectedModelBehavior("fail after 2")

            client = LLMClient(model="gemini-2.5-flash")
            with pytest.raises(ParseError):
                client.generate(VALID_PROMPT, CodeArtifact, max_retries=2)

        MockAgent.assert_called_once_with(
            model="google-gla:gemini-2.5-flash",
            output_type=CodeArtifact,
            retries=2,
        )


# ---------------------------------------------------------------------------
# Test: LLMUnavailableError on HTTP errors
# ---------------------------------------------------------------------------


class TestLLMUnavailableErrorOnHTTPError:
    def test_model_http_error_raises_llm_unavailable(self):
        """ModelHTTPError from pydantic-ai is converted to LLMUnavailableError."""
        from pydantic_ai.exceptions import ModelHTTPError

        http_error = ModelHTTPError(status_code=429, model_name="gemini-2.5-flash", body="rate limited")

        with patch("coder_buddy.llm.client.Agent") as MockAgent:
            mock_agent_instance = MagicMock()
            MockAgent.return_value = mock_agent_instance
            mock_agent_instance.run_sync.side_effect = http_error

            client = LLMClient(model="gemini-2.5-flash")
            with pytest.raises(LLMUnavailableError) as exc_info:
                client.generate(VALID_PROMPT, CodeArtifact)

        assert "429" in str(exc_info.value)

    def test_model_http_error_chained(self):
        """LLMUnavailableError is chained from the original ModelHTTPError."""
        from pydantic_ai.exceptions import ModelHTTPError

        original_exc = ModelHTTPError(status_code=503, model_name="gpt-4o", body="service unavailable")

        with patch("coder_buddy.llm.client.Agent") as MockAgent:
            mock_agent_instance = MagicMock()
            MockAgent.return_value = mock_agent_instance
            mock_agent_instance.run_sync.side_effect = original_exc

            client = LLMClient(model="gpt-4o")
            with pytest.raises(LLMUnavailableError) as exc_info:
                client.generate(VALID_PROMPT, CodeArtifact)

        assert exc_info.value.__cause__ is original_exc

    def test_model_api_error_raises_llm_unavailable(self):
        """ModelAPIError from pydantic-ai is converted to LLMUnavailableError."""
        from pydantic_ai.exceptions import ModelAPIError

        api_error = ModelAPIError("claude-3-5-sonnet", "Authentication failed: invalid API key")

        with patch("coder_buddy.llm.client.Agent") as MockAgent:
            mock_agent_instance = MagicMock()
            MockAgent.return_value = mock_agent_instance
            mock_agent_instance.run_sync.side_effect = api_error

            client = LLMClient(model="claude-3-5-sonnet")
            with pytest.raises(LLMUnavailableError) as exc_info:
                client.generate(VALID_PROMPT, CodeArtifact)

        assert "LLM API error" in str(exc_info.value)

    def test_model_api_error_chained(self):
        """LLMUnavailableError is chained from the original ModelAPIError."""
        from pydantic_ai.exceptions import ModelAPIError

        original_exc = ModelAPIError("gpt-4o", "auth failed")

        with patch("coder_buddy.llm.client.Agent") as MockAgent:
            mock_agent_instance = MagicMock()
            MockAgent.return_value = mock_agent_instance
            mock_agent_instance.run_sync.side_effect = original_exc

            client = LLMClient(model="gpt-4o")
            with pytest.raises(LLMUnavailableError) as exc_info:
                client.generate(VALID_PROMPT, CodeArtifact)

        assert exc_info.value.__cause__ is original_exc

    def test_http_401_unauthorized_raises_llm_unavailable(self):
        """HTTP 401 (auth failure) raises LLMUnavailableError."""
        from pydantic_ai.exceptions import ModelHTTPError

        with patch("coder_buddy.llm.client.Agent") as MockAgent:
            mock_agent_instance = MagicMock()
            MockAgent.return_value = mock_agent_instance
            mock_agent_instance.run_sync.side_effect = ModelHTTPError(
                status_code=401, model_name="gpt-4o", body="unauthorized"
            )

            client = LLMClient(model="gpt-4o")
            with pytest.raises(LLMUnavailableError):
                client.generate(VALID_PROMPT, CodeArtifact)

    def test_http_500_server_error_raises_llm_unavailable(self):
        """HTTP 500 (server error) raises LLMUnavailableError."""
        from pydantic_ai.exceptions import ModelHTTPError

        with patch("coder_buddy.llm.client.Agent") as MockAgent:
            mock_agent_instance = MagicMock()
            MockAgent.return_value = mock_agent_instance
            mock_agent_instance.run_sync.side_effect = ModelHTTPError(
                status_code=500, model_name="gemini-2.5-flash", body="internal server error"
            )

            client = LLMClient(model="gemini-2.5-flash")
            with pytest.raises(LLMUnavailableError) as exc_info:
                client.generate(VALID_PROMPT, CodeArtifact)

        assert "500" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test: LLMUnavailableError on network/connection errors
# ---------------------------------------------------------------------------


class TestLLMUnavailableErrorOnNetworkError:
    def test_connection_error_raises_llm_unavailable(self):
        """A ConnectionError is converted to LLMUnavailableError."""
        with patch("coder_buddy.llm.client.Agent") as MockAgent:
            mock_agent_instance = MagicMock()
            MockAgent.return_value = mock_agent_instance
            mock_agent_instance.run_sync.side_effect = ConnectionError("Connection refused")

            client = LLMClient(model="gemini-2.5-flash")
            with pytest.raises(LLMUnavailableError):
                client.generate(VALID_PROMPT, CodeArtifact)

    def test_timeout_error_raises_llm_unavailable(self):
        """A TimeoutError is converted to LLMUnavailableError."""
        with patch("coder_buddy.llm.client.Agent") as MockAgent:
            mock_agent_instance = MagicMock()
            MockAgent.return_value = mock_agent_instance
            mock_agent_instance.run_sync.side_effect = TimeoutError("Request timed out")

            client = LLMClient(model="gpt-4o")
            with pytest.raises(LLMUnavailableError):
                client.generate(VALID_PROMPT, CodeArtifact)


# ---------------------------------------------------------------------------
# Test: LLMClient initialisation
# ---------------------------------------------------------------------------


class TestLLMClientInit:
    def test_init_sets_model(self):
        """LLMClient stores the model name."""
        client = LLMClient(model="gemini-2.5-flash")
        assert client._model == "gemini-2.5-flash"

    def test_init_resolves_model_string_gemini(self):
        """gemini-2.5-flash resolves to google-gla:gemini-2.5-flash."""
        client = LLMClient(model="gemini-2.5-flash")
        assert client._model_string == "google-gla:gemini-2.5-flash"

    def test_init_resolves_model_string_gpt4o(self):
        """gpt-4o resolves to openai:gpt-4o."""
        client = LLMClient(model="gpt-4o")
        assert client._model_string == "openai:gpt-4o"

    def test_init_resolves_model_string_claude(self):
        """claude-3-5-sonnet resolves to anthropic:claude-3-5-sonnet-latest."""
        client = LLMClient(model="claude-3-5-sonnet")
        assert client._model_string == "anthropic:claude-3-5-sonnet-latest"

    def test_init_unknown_model_uses_raw_name(self):
        """Unknown model names fall back to the raw model name."""
        client = LLMClient(model="some-unknown-model")
        assert client._model_string == "some-unknown-model"

    def test_init_sets_api_key_in_env(self, monkeypatch):
        """When api_key is provided, it is set in the environment."""
        import os
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        LLMClient(model="gemini-2.5-flash", api_key="test-key-123")
        assert os.environ.get("GEMINI_API_KEY") == "test-key-123"

    def test_init_no_api_key_does_not_set_env(self, monkeypatch):
        """When api_key is None, the environment variable is not modified."""
        import os
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        LLMClient(model="gemini-2.5-flash", api_key=None)
        assert os.environ.get("GEMINI_API_KEY") is None


# ---------------------------------------------------------------------------
# Test: TokenRecord has input_tokens > 0 and output_tokens > 0 (Task 20.1)
# ---------------------------------------------------------------------------


class TestTokenRecordPositiveTokens:
    """
    Verify that every successful LLMClient.generate() call returns a
    TokenRecord with input_tokens > 0 and output_tokens > 0.
    """

    def test_token_record_input_tokens_positive(self):
        """input_tokens > 0 when LLM returns non-zero usage."""
        mock_result = _make_run_result(VALID_ARTIFACT, input_tokens=150, output_tokens=60)

        with patch("coder_buddy.llm.client.Agent") as MockAgent:
            mock_agent_instance = MagicMock()
            MockAgent.return_value = mock_agent_instance
            mock_agent_instance.run_sync.return_value = mock_result

            client = LLMClient(model="gemini-2.5-flash")
            _, token_record = client.generate(VALID_PROMPT, CodeArtifact)

        assert token_record.input_tokens > 0, (
            f"Expected input_tokens > 0, got {token_record.input_tokens}"
        )

    def test_token_record_output_tokens_positive(self):
        """output_tokens > 0 when LLM returns non-zero usage."""
        mock_result = _make_run_result(VALID_ARTIFACT, input_tokens=150, output_tokens=60)

        with patch("coder_buddy.llm.client.Agent") as MockAgent:
            mock_agent_instance = MagicMock()
            MockAgent.return_value = mock_agent_instance
            mock_agent_instance.run_sync.return_value = mock_result

            client = LLMClient(model="gemini-2.5-flash")
            _, token_record = client.generate(VALID_PROMPT, CodeArtifact)

        assert token_record.output_tokens > 0, (
            f"Expected output_tokens > 0, got {token_record.output_tokens}"
        )

    def test_token_record_both_tokens_positive_gpt4o(self):
        """Both input_tokens and output_tokens > 0 for gpt-4o model."""
        mock_result = _make_run_result(VALID_ARTIFACT, input_tokens=500, output_tokens=200)

        with patch("coder_buddy.llm.client.Agent") as MockAgent:
            mock_agent_instance = MagicMock()
            MockAgent.return_value = mock_agent_instance
            mock_agent_instance.run_sync.return_value = mock_result

            client = LLMClient(model="gpt-4o")
            _, token_record = client.generate(VALID_PROMPT, CodeArtifact)

        assert token_record.input_tokens > 0
        assert token_record.output_tokens > 0

    def test_token_record_both_tokens_positive_claude(self):
        """Both input_tokens and output_tokens > 0 for claude-3-5-sonnet model."""
        mock_result = _make_run_result(VALID_ARTIFACT, input_tokens=300, output_tokens=120)

        with patch("coder_buddy.llm.client.Agent") as MockAgent:
            mock_agent_instance = MagicMock()
            MockAgent.return_value = mock_agent_instance
            mock_agent_instance.run_sync.return_value = mock_result

            client = LLMClient(model="claude-3-5-sonnet")
            _, token_record = client.generate(VALID_PROMPT, CodeArtifact)

        assert token_record.input_tokens > 0
        assert token_record.output_tokens > 0

    def test_token_record_reflects_exact_usage_values(self):
        """TokenRecord accurately reflects the exact token counts from LLM usage."""
        input_tokens = 1234
        output_tokens = 567
        mock_result = _make_run_result(
            VALID_ARTIFACT, input_tokens=input_tokens, output_tokens=output_tokens
        )

        with patch("coder_buddy.llm.client.Agent") as MockAgent:
            mock_agent_instance = MagicMock()
            MockAgent.return_value = mock_agent_instance
            mock_agent_instance.run_sync.return_value = mock_result

            client = LLMClient(model="gemini-2.5-flash")
            _, token_record = client.generate(VALID_PROMPT, CodeArtifact)

        assert token_record.input_tokens == input_tokens
        assert token_record.output_tokens == output_tokens
        assert token_record.input_tokens > 0
        assert token_record.output_tokens > 0

    def test_token_record_minimum_positive_values(self):
        """TokenRecord with minimum positive values (1 each) satisfies > 0."""
        mock_result = _make_run_result(VALID_ARTIFACT, input_tokens=1, output_tokens=1)

        with patch("coder_buddy.llm.client.Agent") as MockAgent:
            mock_agent_instance = MagicMock()
            MockAgent.return_value = mock_agent_instance
            mock_agent_instance.run_sync.return_value = mock_result

            client = LLMClient(model="gemini-2.5-flash")
            _, token_record = client.generate(VALID_PROMPT, CodeArtifact)

        assert token_record.input_tokens > 0
        assert token_record.output_tokens > 0


# ---------------------------------------------------------------------------
# Test: pydantic-ai not installed
# ---------------------------------------------------------------------------


class TestPydanticAINotInstalled:
    def test_raises_llm_unavailable_when_agent_is_none(self):
        """When pydantic-ai is not installed (Agent=None), LLMUnavailableError is raised."""
        with patch("coder_buddy.llm.client.Agent", None):
            client = LLMClient(model="gemini-2.5-flash")
            with pytest.raises(LLMUnavailableError, match="pydantic-ai is not installed"):
                client.generate(VALID_PROMPT, CodeArtifact)
