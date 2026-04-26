"""
Unit tests for write_node prompt construction.

Validates four scenarios:
1. Retry includes logs: When retry_count > 0, the LLM prompt contains execution_logs.
2. No-retry omits logs: When retry_count == 0, the LLM prompt does NOT contain execution_logs.
3. Reference injection when keywords present: When the user prompt contains a reference
   keyword AND session_history is non-empty, the [Reference code] block is injected.
4. No injection when history empty: When session_history is empty, the [Reference code]
   block is NOT injected even if the prompt contains reference keywords.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from coder_buddy.models import CodeArtifact, HistoryEntry, TokenRecord, TokenUsage
from coder_buddy.nodes.write_node import _build_prompt, _has_prior_reference, make_write_node


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_history_entry(
    prompt: str = "Write a hello world script",
    source_code: str = "print('hello')",
    file_name: str = "hello.py",
    dependencies: list[str] | None = None,
) -> HistoryEntry:
    """Return a minimal HistoryEntry for testing."""
    return HistoryEntry(
        prompt=prompt,
        source_code=source_code,
        file_name=file_name,
        dependencies=dependencies or [],
        timestamp=datetime(2024, 1, 1, 12, 0, 0),
    )


def _make_state(
    *,
    user_prompt: str = "Write a script",
    session_history: list | None = None,
    execution_logs: str = "",
    retry_count: int = 0,
    max_retries: int = 5,
) -> dict:
    """Build a minimal AgentState dict for write_node testing."""
    return {
        "user_prompt": user_prompt,
        "current_code": "",
        "execution_logs": execution_logs,
        "error_status": False,
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
        "session_history": session_history if session_history is not None else [],
        "max_retries": max_retries,
        "pre_refactor_code": None,
    }


def _make_mock_llm_client(source_code: str = "print('hello')") -> MagicMock:
    """Return a mock LLMClient whose generate() returns a valid CodeArtifact."""
    artifact = CodeArtifact(
        source_code=source_code,
        file_name="main.py",
        dependencies=[],
        language="python",
    )
    token_record = TokenRecord(input_tokens=100, output_tokens=50)
    mock_client = MagicMock()
    mock_client.generate.return_value = (artifact, token_record)
    return mock_client


def _make_mock_config(session_history_context_n: int = 5) -> MagicMock:
    """Return a mock AgentConfig."""
    config = MagicMock()
    config.session_history_context_n = session_history_context_n
    return config


# ---------------------------------------------------------------------------
# Tests for _build_prompt directly
# ---------------------------------------------------------------------------


class TestBuildPromptRetryLogs:
    """Scenario 1: When retry_count > 0, execution_logs appear in the prompt."""

    def test_retry_count_1_includes_logs(self):
        """retry_count=1 causes execution_logs to be included in the prompt."""
        logs = "Traceback (most recent call last):\n  NameError: name 'x' is not defined"
        prompt = _build_prompt(
            user_prompt="Fix the script",
            session_history=[],
            execution_logs=logs,
            retry_count=1,
            session_history_context_n=5,
        )
        assert logs in prompt

    def test_retry_count_3_includes_logs(self):
        """retry_count=3 also causes execution_logs to be included."""
        logs = "SyntaxError: invalid syntax at line 5"
        prompt = _build_prompt(
            user_prompt="Fix the script",
            session_history=[],
            execution_logs=logs,
            retry_count=3,
            session_history_context_n=5,
        )
        assert logs in prompt

    def test_retry_prompt_contains_fix_instruction(self):
        """When retrying, the prompt includes the 'fix the issues' instruction."""
        logs = "RuntimeError: division by zero"
        prompt = _build_prompt(
            user_prompt="Write a calculator",
            session_history=[],
            execution_logs=logs,
            retry_count=1,
            session_history_context_n=5,
        )
        assert "fix" in prompt.lower()
        assert logs in prompt

    def test_retry_logs_appear_verbatim(self):
        """The execution_logs content appears verbatim (not truncated or modified)."""
        logs = "Line 1\nLine 2\nLine 3\nTraceback:\n  ValueError: bad value"
        prompt = _build_prompt(
            user_prompt="Fix it",
            session_history=[],
            execution_logs=logs,
            retry_count=2,
            session_history_context_n=5,
        )
        assert logs in prompt


class TestBuildPromptNoRetryOmitsLogs:
    """Scenario 2: When retry_count == 0, execution_logs are NOT in the prompt."""

    def test_retry_count_0_omits_logs(self):
        """retry_count=0 means execution_logs are excluded from the prompt."""
        logs = "SomeError: this should not appear"
        prompt = _build_prompt(
            user_prompt="Write a script",
            session_history=[],
            execution_logs=logs,
            retry_count=0,
            session_history_context_n=5,
        )
        assert logs not in prompt

    def test_retry_count_0_with_nonempty_logs_still_omits(self):
        """Even with non-empty logs, retry_count=0 keeps them out of the prompt."""
        logs = "Traceback (most recent call last):\n  File 'main.py', line 1\nNameError"
        prompt = _build_prompt(
            user_prompt="Write a hello world script",
            session_history=[],
            execution_logs=logs,
            retry_count=0,
            session_history_context_n=5,
        )
        assert logs not in prompt

    def test_retry_count_0_prompt_contains_user_request(self):
        """With retry_count=0, the prompt still contains the user request."""
        user_prompt = "Write a Fibonacci function"
        prompt = _build_prompt(
            user_prompt=user_prompt,
            session_history=[],
            execution_logs="some logs",
            retry_count=0,
            session_history_context_n=5,
        )
        assert user_prompt in prompt

    def test_empty_logs_with_retry_count_1_omits_logs_section(self):
        """When execution_logs is empty and retry_count=1, no logs section is added."""
        prompt = _build_prompt(
            user_prompt="Fix the script",
            session_history=[],
            execution_logs="",
            retry_count=1,
            session_history_context_n=5,
        )
        # The "fix the issues" instruction should not appear when logs are empty
        assert "fix the issues" not in prompt.lower()


class TestBuildPromptReferenceInjection:
    """Scenario 3: [Reference code] block injected when keywords present + history non-empty."""

    def test_reference_injected_for_keyword_the_script(self):
        """'the script' keyword triggers [Reference code] injection."""
        entry = _make_history_entry(source_code="x = 1\nprint(x)")
        prompt = _build_prompt(
            user_prompt="Optimise the script",
            session_history=[entry],
            execution_logs="",
            retry_count=0,
            session_history_context_n=5,
        )
        assert "[Reference code" in prompt
        assert "x = 1\nprint(x)" in prompt

    def test_reference_injected_for_keyword_you_just_wrote(self):
        """'you just wrote' keyword triggers [Reference code] injection."""
        entry = _make_history_entry(source_code="def foo(): pass")
        prompt = _build_prompt(
            user_prompt="Add docstrings to the code you just wrote",
            session_history=[entry],
            execution_logs="",
            retry_count=0,
            session_history_context_n=5,
        )
        assert "[Reference code" in prompt
        assert "def foo(): pass" in prompt

    def test_reference_injected_for_keyword_previous(self):
        """'previous' keyword triggers [Reference code] injection."""
        entry = _make_history_entry(source_code="result = 42")
        prompt = _build_prompt(
            user_prompt="Improve the previous script",
            session_history=[entry],
            execution_logs="",
            retry_count=0,
            session_history_context_n=5,
        )
        assert "[Reference code" in prompt
        assert "result = 42" in prompt

    def test_reference_uses_most_recent_history_entry(self):
        """The [Reference code] block uses the most recent (last) history entry."""
        old_entry = _make_history_entry(source_code="old_code = True")
        new_entry = _make_history_entry(source_code="new_code = True")
        prompt = _build_prompt(
            user_prompt="Refactor the script",
            session_history=[old_entry, new_entry],
            execution_logs="",
            retry_count=0,
            session_history_context_n=5,
        )
        assert "new_code = True" in prompt
        # The reference block should use the most recent entry
        ref_start = prompt.find("[Reference code")
        ref_end = prompt.find("[End reference code]")
        reference_section = prompt[ref_start:ref_end]
        assert "new_code = True" in reference_section

    def test_reference_block_has_end_marker(self):
        """The injected reference block includes the [End reference code] marker."""
        entry = _make_history_entry(source_code="pass")
        prompt = _build_prompt(
            user_prompt="Improve the code",
            session_history=[entry],
            execution_logs="",
            retry_count=0,
            session_history_context_n=5,
        )
        assert "[End reference code]" in prompt

    def test_all_reference_keywords_trigger_injection(self):
        """Each keyword in REFERENCE_KEYWORDS triggers injection when history is present."""
        from coder_buddy.nodes.write_node import REFERENCE_KEYWORDS

        entry = _make_history_entry(source_code="x = 1")
        for keyword in REFERENCE_KEYWORDS:
            prompt = _build_prompt(
                user_prompt=f"Please update {keyword} to be faster",
                session_history=[entry],
                execution_logs="",
                retry_count=0,
                session_history_context_n=5,
            )
            assert "[Reference code" in prompt, (
                f"Expected [Reference code] block for keyword '{keyword}'"
            )


class TestBuildPromptNoInjectionWhenHistoryEmpty:
    """Scenario 4: [Reference code] block NOT injected when session_history is empty."""

    def test_no_reference_when_history_empty_with_keyword(self):
        """Even with a reference keyword, empty history means no [Reference code] block."""
        prompt = _build_prompt(
            user_prompt="Optimise the script",
            session_history=[],
            execution_logs="",
            retry_count=0,
            session_history_context_n=5,
        )
        assert "[Reference code" not in prompt

    def test_no_reference_when_history_empty_with_you_just_wrote(self):
        """'you just wrote' keyword with empty history does not inject reference block."""
        prompt = _build_prompt(
            user_prompt="Add tests to the code you just wrote",
            session_history=[],
            execution_logs="",
            retry_count=0,
            session_history_context_n=5,
        )
        assert "[Reference code" not in prompt

    def test_no_reference_when_history_empty_with_previous(self):
        """'previous' keyword with empty history does not inject reference block."""
        prompt = _build_prompt(
            user_prompt="Improve the previous script",
            session_history=[],
            execution_logs="",
            retry_count=0,
            session_history_context_n=5,
        )
        assert "[Reference code" not in prompt

    def test_no_reference_when_no_keyword_but_history_present(self):
        """Without a reference keyword, no [Reference code] block even with history."""
        entry = _make_history_entry(source_code="print('hello')")
        prompt = _build_prompt(
            user_prompt="Write a new sorting algorithm",
            session_history=[entry],
            execution_logs="",
            retry_count=0,
            session_history_context_n=5,
        )
        assert "[Reference code" not in prompt

    def test_no_reference_block_end_marker_when_history_empty(self):
        """[End reference code] marker is absent when history is empty."""
        prompt = _build_prompt(
            user_prompt="Refactor the script",
            session_history=[],
            execution_logs="",
            retry_count=0,
            session_history_context_n=5,
        )
        assert "[End reference code]" not in prompt


# ---------------------------------------------------------------------------
# Tests for _has_prior_reference
# ---------------------------------------------------------------------------


class TestHasPriorReference:
    """Unit tests for the _has_prior_reference keyword detection helper."""

    def test_returns_true_for_the_script(self):
        assert _has_prior_reference("Optimise the script") is True

    def test_returns_true_for_you_just_wrote(self):
        assert _has_prior_reference("Add tests to the code you just wrote") is True

    def test_returns_true_for_previous(self):
        assert _has_prior_reference("Improve the previous version") is True

    def test_returns_true_case_insensitive(self):
        assert _has_prior_reference("OPTIMISE THE SCRIPT") is True

    def test_returns_false_for_unrelated_prompt(self):
        assert _has_prior_reference("Write a new sorting algorithm") is False

    def test_returns_false_for_empty_string(self):
        assert _has_prior_reference("") is False


# ---------------------------------------------------------------------------
# Integration tests via make_write_node (mocking LLMClient.generate)
# ---------------------------------------------------------------------------


class TestWriteNodePromptViaNode:
    """
    Integration tests that invoke the write_node closure and capture the
    prompt passed to llm_client.generate via a mock.
    """

    def test_retry_node_passes_logs_to_llm(self):
        """When retry_count > 0, the prompt passed to llm_client.generate contains logs."""
        logs = "NameError: name 'foo' is not defined"
        mock_client = _make_mock_llm_client()
        config = _make_mock_config()

        node = make_write_node(mock_client, config)
        state = _make_state(
            user_prompt="Fix the script",
            execution_logs=logs,
            retry_count=1,
        )

        with patch("coder_buddy.nodes.write_node.log_node_event"):
            node(state)

        called_prompt = mock_client.generate.call_args[0][0]
        assert logs in called_prompt

    def test_no_retry_node_omits_logs_from_llm(self):
        """When retry_count == 0, the prompt passed to llm_client.generate omits logs."""
        logs = "SomeError: should not appear"
        mock_client = _make_mock_llm_client()
        config = _make_mock_config()

        node = make_write_node(mock_client, config)
        state = _make_state(
            user_prompt="Write a script",
            execution_logs=logs,
            retry_count=0,
        )

        with patch("coder_buddy.nodes.write_node.log_node_event"):
            node(state)

        called_prompt = mock_client.generate.call_args[0][0]
        assert logs not in called_prompt

    def test_reference_injected_via_node_when_keyword_and_history(self):
        """write_node injects [Reference code] when keyword present and history non-empty."""
        entry = _make_history_entry(source_code="def greet(): print('hi')")
        mock_client = _make_mock_llm_client()
        config = _make_mock_config()

        node = make_write_node(mock_client, config)
        state = _make_state(
            user_prompt="Add type hints to the script",
            session_history=[entry],
            retry_count=0,
        )

        with patch("coder_buddy.nodes.write_node.log_node_event"):
            node(state)

        called_prompt = mock_client.generate.call_args[0][0]
        assert "[Reference code" in called_prompt
        assert "def greet(): print('hi')" in called_prompt

    def test_no_reference_injected_via_node_when_history_empty(self):
        """write_node does NOT inject [Reference code] when session_history is empty."""
        mock_client = _make_mock_llm_client()
        config = _make_mock_config()

        node = make_write_node(mock_client, config)
        state = _make_state(
            user_prompt="Optimise the script",
            session_history=[],
            retry_count=0,
        )

        with patch("coder_buddy.nodes.write_node.log_node_event"):
            node(state)

        called_prompt = mock_client.generate.call_args[0][0]
        assert "[Reference code" not in called_prompt


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------

# Feature: coder-buddy, Property 1: execution logs are included in retry prompts
from hypothesis import given, settings
import hypothesis.strategies as st


class TestProperty14WriteNodeStateDict:
    """
    Property 14: The partial state dict returned by ``write_node`` sets
    ``current_code`` to ``artifact.source_code`` and ``dependencies`` to
    ``artifact.dependencies``.

    Validates: Requirements 2.1, 6.4
    """

    @given(
        source_code=st.text(min_size=1).filter(lambda s: s.strip()),
        file_name=st.from_regex(r"[a-z][a-z0-9_]{0,15}\.py", fullmatch=True),
        dependencies=st.lists(
            st.from_regex(r"[a-z][a-z0-9_\-]{0,19}", fullmatch=True),
            max_size=5,
        ),
    )
    @settings(max_examples=100)
    def test_property_14_current_code_and_dependencies_match_artifact(
        self,
        source_code: str,
        file_name: str,
        dependencies: list[str],
    ):
        """
        **Validates: Requirements 2.1, 6.4**

        For any valid ``CodeArtifact`` produced by the LLM, the partial state
        dict returned by ``write_node`` must satisfy:
        - ``result["current_code"] == artifact.source_code``
        - ``result["dependencies"] == artifact.dependencies``
        """
        artifact = CodeArtifact(
            source_code=source_code,
            file_name=file_name,
            dependencies=dependencies,
            language="python",
        )
        token_record = TokenRecord(input_tokens=10, output_tokens=5)

        mock_client = MagicMock()
        mock_client.generate.return_value = (artifact, token_record)
        config = _make_mock_config()

        node = make_write_node(mock_client, config)
        state = _make_state(user_prompt="Write a script")

        with patch("coder_buddy.nodes.write_node.log_node_event"):
            result = node(state)

        assert result["current_code"] == artifact.source_code, (
            f"current_code mismatch: expected {artifact.source_code!r}, "
            f"got {result['current_code']!r}"
        )
        assert result["dependencies"] == artifact.dependencies, (
            f"dependencies mismatch: expected {artifact.dependencies!r}, "
            f"got {result['dependencies']!r}"
        )


class TestProperty1ExecutionLogsInRetryPrompt:
    """
    Property 1: For any non-empty execution_logs and retry_count > 0,
    the constructed LLM prompt contains execution_logs verbatim.

    Validates: Requirements 1.4
    """

    @given(
        execution_logs=st.text(min_size=1).filter(lambda s: s.strip()),
        retry_count=st.integers(min_value=1, max_value=9),
    )
    @settings(max_examples=100)
    def test_property_1_execution_logs_verbatim_in_retry_prompt(
        self, execution_logs: str, retry_count: int
    ):
        """
        **Validates: Requirements 1.4**

        For any non-empty execution_logs string and any retry_count > 0,
        the prompt built by _build_prompt must contain execution_logs verbatim.
        """
        prompt = _build_prompt(
            user_prompt="Fix the script",
            session_history=[],
            execution_logs=execution_logs,
            retry_count=retry_count,
            session_history_context_n=5,
        )
        assert execution_logs in prompt, (
            f"execution_logs not found verbatim in prompt.\n"
            f"execution_logs={execution_logs!r}\n"
            f"retry_count={retry_count}\n"
            f"prompt={prompt!r}"
        )


# Feature: coder-buddy, Property 17: the LLM prompt contains exactly the last N session history entries when history length >= N
class TestProperty17LastNSessionHistoryEntries:
    """
    Property 17: For any session history of length L >= N and any configured
    session_history_context_n = N, the LLM prompt constructed by Write_Node
    SHALL contain the content of exactly the last N history entries and SHALL
    NOT contain entries older than position L - N.

    Validates: Requirements 10.2
    """

    @given(
        n=st.integers(min_value=1, max_value=5),
        extra=st.integers(min_value=1, max_value=5),
        user_prompt=st.text(min_size=1, max_size=50).filter(lambda s: s.strip()),
    )
    @settings(max_examples=100)
    def test_property_17_prompt_contains_exactly_last_n_history_entries(
        self,
        n: int,
        extra: int,
        user_prompt: str,
    ):
        """
        **Validates: Requirements 10.2**

        For any session history of length L = N + extra (L >= N), the LLM
        prompt built by _build_prompt must:
        - Contain the source_code and prompt of each of the last N entries.
        - NOT contain the source_code or prompt of any of the first `extra`
          entries (those older than position L - N), provided those values
          are unique and do not appear in the last N entries.
        """
        # Build L = n + extra entries with unique, distinguishable content.
        # Use a fixed prefix per index so values are unique and won't
        # accidentally appear in other entries.
        total = n + extra
        history: list[HistoryEntry] = [
            HistoryEntry(
                prompt=f"OLDER_PROMPT_{i}_UNIQUE",
                source_code=f"older_code_{i}_unique = True",
                file_name="script.py",
                dependencies=[],
                timestamp=datetime(2024, 1, 1, 0, i % 60, 0),
            )
            for i in range(extra)
        ] + [
            HistoryEntry(
                prompt=f"RECENT_PROMPT_{i}_UNIQUE",
                source_code=f"recent_code_{i}_unique = True",
                file_name="script.py",
                dependencies=[],
                timestamp=datetime(2024, 1, 2, 0, i % 60, 0),
            )
            for i in range(n)
        ]

        # The last N entries are the "recent" ones (indices extra..total-1).
        recent_entries = history[-n:]
        older_entries = history[:extra]

        prompt = _build_prompt(
            user_prompt=user_prompt,
            session_history=history,
            execution_logs="",
            retry_count=0,
            session_history_context_n=n,
        )

        # Assert: each of the last N entries' source_code and prompt appear.
        for i, entry in enumerate(recent_entries):
            assert entry.source_code in prompt, (
                f"Recent entry {i} source_code not found in prompt.\n"
                f"source_code={entry.source_code!r}\n"
                f"n={n}, extra={extra}\n"
                f"prompt={prompt!r}"
            )
            assert entry.prompt in prompt, (
                f"Recent entry {i} prompt not found in prompt.\n"
                f"entry.prompt={entry.prompt!r}\n"
                f"n={n}, extra={extra}\n"
                f"prompt={prompt!r}"
            )

        # Assert: none of the older entries' source_code or prompt appear
        # (they are unique strings that cannot appear in the recent entries).
        for i, entry in enumerate(older_entries):
            assert entry.source_code not in prompt, (
                f"Older entry {i} source_code unexpectedly found in prompt.\n"
                f"source_code={entry.source_code!r}\n"
                f"n={n}, extra={extra}\n"
                f"prompt={prompt!r}"
            )
            assert entry.prompt not in prompt, (
                f"Older entry {i} prompt unexpectedly found in prompt.\n"
                f"entry.prompt={entry.prompt!r}\n"
                f"n={n}, extra={extra}\n"
                f"prompt={prompt!r}"
            )


# ---------------------------------------------------------------------------
# Tests for token_usage accumulation in write_node
# ---------------------------------------------------------------------------


class TestWriteNodeTokenUsageAccumulation:
    """
    Verify that write_node correctly accumulates its TokenRecord into
    AgentState.token_usage using model_copy(update={"write_node": token_record}).
    """

    def test_token_usage_write_node_field_updated(self):
        """
        After write_node runs, result['token_usage'].write_node has the
        TokenRecord returned by the LLM (non-zero tokens).
        """
        mock_client = _make_mock_llm_client()
        config = _make_mock_config()
        node = make_write_node(mock_client, config)
        state = _make_state()

        with patch("coder_buddy.nodes.write_node.log_node_event"):
            result = node(state)

        assert result["token_usage"].write_node.input_tokens == 100
        assert result["token_usage"].write_node.output_tokens == 50

    def test_token_usage_other_fields_unchanged(self):
        """
        After write_node runs, all other TokenRecord fields in token_usage
        retain their prior values (zero by default).
        """
        mock_client = _make_mock_llm_client()
        config = _make_mock_config()
        node = make_write_node(mock_client, config)
        state = _make_state()

        with patch("coder_buddy.nodes.write_node.log_node_event"):
            result = node(state)

        usage = result["token_usage"]
        # Only write_node should be updated; all others remain at zero
        assert usage.refactor_node.input_tokens == 0
        assert usage.refactor_node.output_tokens == 0
        assert usage.explanation.input_tokens == 0
        assert usage.explanation.output_tokens == 0
        assert usage.test_node.input_tokens == 0
        assert usage.test_node.output_tokens == 0
        assert usage.confidence.input_tokens == 0
        assert usage.confidence.output_tokens == 0

    def test_token_usage_prior_values_preserved(self):
        """
        When the input state already has non-zero token_usage for other nodes,
        write_node only updates the write_node field and leaves others intact.
        """
        from coder_buddy.models import TokenRecord, TokenUsage

        prior_usage = TokenUsage(
            refactor_node=TokenRecord(input_tokens=200, output_tokens=100),
            test_node=TokenRecord(input_tokens=150, output_tokens=75),
        )
        mock_client = _make_mock_llm_client()
        config = _make_mock_config()
        node = make_write_node(mock_client, config)
        state = _make_state()
        state["token_usage"] = prior_usage

        with patch("coder_buddy.nodes.write_node.log_node_event"):
            result = node(state)

        usage = result["token_usage"]
        # write_node field updated with new record
        assert usage.write_node.input_tokens == 100
        assert usage.write_node.output_tokens == 50
        # Other fields preserved from prior_usage
        assert usage.refactor_node.input_tokens == 200
        assert usage.refactor_node.output_tokens == 100
        assert usage.test_node.input_tokens == 150
        assert usage.test_node.output_tokens == 75

    def test_token_usage_returned_in_result_dict(self):
        """The result dict from write_node always contains the 'token_usage' key."""
        mock_client = _make_mock_llm_client()
        config = _make_mock_config()
        node = make_write_node(mock_client, config)
        state = _make_state()

        with patch("coder_buddy.nodes.write_node.log_node_event"):
            result = node(state)

        assert "token_usage" in result

    def test_token_usage_is_new_object_not_same_reference(self):
        """
        write_node creates a new TokenUsage via model_copy, so the returned
        token_usage is a different object from the input state's token_usage.
        """
        mock_client = _make_mock_llm_client()
        config = _make_mock_config()
        node = make_write_node(mock_client, config)
        original_usage = TokenUsage()
        state = _make_state()
        state["token_usage"] = original_usage

        with patch("coder_buddy.nodes.write_node.log_node_event"):
            result = node(state)

        # model_copy creates a new instance
        assert result["token_usage"] is not original_usage
