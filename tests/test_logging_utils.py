"""
Unit tests for structured JSON logging utilities.

Validates:
- log_node_event emits valid JSON parseable by json.loads
- log_node_event output contains all required keys: ts, node, event, retry_count, outcome
- log_node_event field values match the arguments passed in
- log_node_event merges extra fields into the top-level entry
- _extract_error_summary returns the last traceback exception line when present
- _extract_error_summary returns the first non-empty line when no traceback is present
- _extract_error_summary truncates output to 500 characters
- _extract_error_summary returns "" for empty or whitespace-only input
- _extract_error_summary handles multiple tracebacks (returns last exception line)
"""

import json
import logging
from logging.handlers import MemoryHandler

import pytest

from coder_buddy.logging_utils import _extract_error_summary, log_node_event


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


class CapturingHandler(logging.Handler):
    """A logging handler that stores all emitted records in a list."""

    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    @property
    def messages(self) -> list[str]:
        return [self.format(r) for r in self.records]


def make_handler() -> CapturingHandler:
    """Attach a fresh CapturingHandler to the coder_buddy logger and return it."""
    handler = CapturingHandler()
    logger = logging.getLogger("coder_buddy")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return handler


def remove_handler(handler: CapturingHandler) -> None:
    """Detach the handler from the coder_buddy logger."""
    logging.getLogger("coder_buddy").removeHandler(handler)


# --------------------------------------------------------------------------- #
# log_node_event — JSON validity
# --------------------------------------------------------------------------- #


class TestLogNodeEventJsonValidity:
    def setup_method(self):
        self.handler = make_handler()

    def teardown_method(self):
        remove_handler(self.handler)

    def test_emitted_message_is_valid_json(self):
        """The log message emitted by log_node_event must be parseable by json.loads."""
        log_node_event("write_node", "start", retry_count=0)
        assert len(self.handler.records) == 1
        msg = self.handler.records[0].getMessage()
        parsed = json.loads(msg)  # raises if not valid JSON
        assert isinstance(parsed, dict)

    def test_emitted_message_is_valid_json_for_end_event(self):
        """End events must also produce valid JSON."""
        log_node_event("execute_node", "end", retry_count=2, outcome="retry")
        msg = self.handler.records[0].getMessage()
        parsed = json.loads(msg)
        assert isinstance(parsed, dict)


# --------------------------------------------------------------------------- #
# log_node_event — required keys
# --------------------------------------------------------------------------- #


class TestLogNodeEventRequiredKeys:
    def setup_method(self):
        self.handler = make_handler()

    def teardown_method(self):
        remove_handler(self.handler)

    def _parse_last(self) -> dict:
        msg = self.handler.records[-1].getMessage()
        return json.loads(msg)

    def test_required_keys_present(self):
        """Every emitted entry must contain ts, node, event, retry_count, outcome."""
        log_node_event("write_node", "start", retry_count=0)
        entry = self._parse_last()
        for key in ("ts", "node", "event", "retry_count", "outcome"):
            assert key in entry, f"Missing required key: {key}"

    def test_ts_is_float(self):
        """ts must be a float (Unix timestamp)."""
        log_node_event("write_node", "start", retry_count=0)
        entry = self._parse_last()
        assert isinstance(entry["ts"], float)

    def test_ts_is_positive(self):
        """ts must be a positive Unix timestamp."""
        log_node_event("write_node", "start", retry_count=0)
        entry = self._parse_last()
        assert entry["ts"] > 0


# --------------------------------------------------------------------------- #
# log_node_event — field values
# --------------------------------------------------------------------------- #


class TestLogNodeEventFieldValues:
    def setup_method(self):
        self.handler = make_handler()

    def teardown_method(self):
        remove_handler(self.handler)

    def _parse_last(self) -> dict:
        msg = self.handler.records[-1].getMessage()
        return json.loads(msg)

    def test_node_value_matches_argument(self):
        """The node field must equal the node argument passed in."""
        log_node_event("my_node", "start", retry_count=0)
        entry = self._parse_last()
        assert entry["node"] == "my_node"

    def test_event_start_value_matches(self):
        """The event field must equal 'start' when 'start' is passed."""
        log_node_event("write_node", "start", retry_count=0)
        entry = self._parse_last()
        assert entry["event"] == "start"

    def test_event_end_value_matches(self):
        """The event field must equal 'end' when 'end' is passed."""
        log_node_event("write_node", "end", retry_count=1)
        entry = self._parse_last()
        assert entry["event"] == "end"

    def test_retry_count_value_matches(self):
        """The retry_count field must equal the retry_count argument passed in."""
        log_node_event("execute_node", "start", retry_count=3)
        entry = self._parse_last()
        assert entry["retry_count"] == 3

    def test_outcome_is_none_when_not_provided(self):
        """outcome must be None (JSON null) when not explicitly provided."""
        log_node_event("write_node", "start", retry_count=0)
        entry = self._parse_last()
        assert entry["outcome"] is None

    def test_outcome_is_set_when_provided(self):
        """outcome must equal the value passed in."""
        log_node_event("evaluator", "end", retry_count=1, outcome="retry")
        entry = self._parse_last()
        assert entry["outcome"] == "retry"

    def test_outcome_refactor_value(self):
        """outcome='refactor' must be stored correctly."""
        log_node_event("evaluator", "end", retry_count=2, outcome="refactor")
        entry = self._parse_last()
        assert entry["outcome"] == "refactor"


# --------------------------------------------------------------------------- #
# log_node_event — extra fields
# --------------------------------------------------------------------------- #


class TestLogNodeEventExtraFields:
    def setup_method(self):
        self.handler = make_handler()

    def teardown_method(self):
        remove_handler(self.handler)

    def _parse_last(self) -> dict:
        msg = self.handler.records[-1].getMessage()
        return json.loads(msg)

    def test_extra_fields_merged_into_top_level(self):
        """Fields from the extra dict must appear at the top level of the entry."""
        log_node_event(
            "write_node",
            "end",
            retry_count=0,
            extra={"model": "gpt-4o", "tokens": 123},
        )
        entry = self._parse_last()
        assert entry["model"] == "gpt-4o"
        assert entry["tokens"] == 123

    def test_extra_none_does_not_add_extra_keys(self):
        """When extra=None, no unexpected keys should appear beyond the required ones."""
        log_node_event("write_node", "start", retry_count=0, extra=None)
        entry = self._parse_last()
        required = {"ts", "node", "event", "retry_count", "outcome"}
        assert set(entry.keys()) == required

    def test_extra_empty_dict_does_not_add_extra_keys(self):
        """When extra={}, no unexpected keys should appear beyond the required ones."""
        log_node_event("write_node", "start", retry_count=0, extra={})
        entry = self._parse_last()
        required = {"ts", "node", "event", "retry_count", "outcome"}
        assert set(entry.keys()) == required


# --------------------------------------------------------------------------- #
# _extract_error_summary — traceback extraction
# --------------------------------------------------------------------------- #


class TestExtractErrorSummaryTraceback:
    def test_returns_last_traceback_exception_line(self):
        """Returns the exception line (last non-empty line after traceback header)."""
        logs = (
            "Traceback (most recent call last):\n"
            "  File 'script.py', line 1, in <module>\n"
            "NameError: name 'x' is not defined\n"
        )
        result = _extract_error_summary(logs)
        assert result == "NameError: name 'x' is not defined"

    def test_returns_exception_line_for_type_error(self):
        """Works for TypeError tracebacks."""
        logs = (
            "Traceback (most recent call last):\n"
            "  File 'script.py', line 5, in <module>\n"
            "TypeError: unsupported operand type(s) for +: 'int' and 'str'\n"
        )
        result = _extract_error_summary(logs)
        assert result == "TypeError: unsupported operand type(s) for +: 'int' and 'str'"

    def test_handles_multiple_tracebacks_returns_last(self):
        """When multiple tracebacks are present, returns the exception line from the last one."""
        logs = (
            "Traceback (most recent call last):\n"
            "  File 'a.py', line 1, in <module>\n"
            "ValueError: first error\n"
            "\n"
            "Traceback (most recent call last):\n"
            "  File 'b.py', line 2, in <module>\n"
            "RuntimeError: second error\n"
        )
        result = _extract_error_summary(logs)
        assert result == "RuntimeError: second error"

    def test_traceback_with_nested_frames(self):
        """Returns the final exception line even when the traceback has many frames."""
        logs = (
            "Traceback (most recent call last):\n"
            "  File 'a.py', line 10, in outer\n"
            "  File 'b.py', line 5, in inner\n"
            "  File 'c.py', line 2, in deepest\n"
            "ZeroDivisionError: division by zero\n"
        )
        result = _extract_error_summary(logs)
        assert result == "ZeroDivisionError: division by zero"


# --------------------------------------------------------------------------- #
# _extract_error_summary — no traceback (first non-empty line)
# --------------------------------------------------------------------------- #


class TestExtractErrorSummaryNoTraceback:
    def test_returns_first_non_empty_line_when_no_traceback(self):
        """When no traceback is present, returns the first non-empty line."""
        logs = "Some output line\nAnother line\n"
        result = _extract_error_summary(logs)
        assert result == "Some output line"

    def test_skips_leading_whitespace_lines(self):
        """Whitespace-only lines before the first real content are skipped."""
        logs = "\n   \n\nActual content here\n"
        result = _extract_error_summary(logs)
        assert result == "Actual content here"

    def test_returns_empty_string_for_empty_input(self):
        """Empty string input returns empty string."""
        result = _extract_error_summary("")
        assert result == ""

    def test_returns_empty_string_for_whitespace_only_input(self):
        """Whitespace-only input returns empty string."""
        result = _extract_error_summary("   \n\n\t  \n")
        assert result == ""

    def test_single_line_no_traceback(self):
        """A single non-empty line is returned as-is."""
        result = _extract_error_summary("Hello world")
        assert result == "Hello world"


# --------------------------------------------------------------------------- #
# _extract_error_summary — truncation
# --------------------------------------------------------------------------- #


class TestExtractErrorSummaryTruncation:
    def test_truncates_traceback_line_at_500_chars(self):
        """A traceback exception line longer than 500 chars is truncated to 500."""
        long_message = "X" * 600
        logs = (
            "Traceback (most recent call last):\n"
            "  File 'script.py', line 1, in <module>\n"
            f"ValueError: {long_message}\n"
        )
        result = _extract_error_summary(logs)
        assert len(result) == 500
        assert result == f"ValueError: {long_message}"[:500]

    def test_truncates_first_line_at_500_chars_no_traceback(self):
        """A first non-empty line longer than 500 chars is truncated to 500."""
        long_line = "A" * 600
        result = _extract_error_summary(long_line)
        assert len(result) == 500
        assert result == long_line[:500]

    def test_does_not_truncate_exactly_500_chars(self):
        """A result of exactly 500 characters is returned unchanged."""
        exact_line = "B" * 500
        result = _extract_error_summary(exact_line)
        assert len(result) == 500
        assert result == exact_line

    def test_does_not_truncate_short_result(self):
        """A result shorter than 500 characters is returned in full."""
        short_line = "Short error message"
        result = _extract_error_summary(short_line)
        assert result == short_line


# --------------------------------------------------------------------------- #
# Property 16: every log_node_event call emits entries parseable by
# json.loads() with event in {"start", "end"}
# Feature: coder-buddy, Property 16: every log_node_event call emits entries parseable by json.loads() with event in {"start", "end"}
# --------------------------------------------------------------------------- #

from hypothesis import given, settings
import hypothesis.strategies as st

# Strategy for JSON-safe scalar values (strings, ints, floats, bools, None)
_json_safe_values = st.one_of(
    st.text(),
    st.integers(),
    st.floats(allow_nan=False, allow_infinity=False),
    st.booleans(),
    st.none(),
)

# Strategy for JSON-safe dicts (string keys, JSON-safe scalar values)
_json_safe_dict = st.one_of(
    st.none(),
    st.dictionaries(st.text(), _json_safe_values),
)


class TestLogNodeEventProperty16:
    """
    **Validates: Requirements 1.2**

    Property 16: every log_node_event call emits entries parseable by
    json.loads() with event in {"start", "end"}.
    """

    def setup_method(self):
        self.handler = make_handler()

    def teardown_method(self):
        remove_handler(self.handler)

    @settings(max_examples=100)
    @given(
        node=st.text(),
        retry_count=st.integers(),
        outcome=st.one_of(st.none(), st.text()),
        extra=_json_safe_dict,
    )
    def test_start_event_emits_parseable_json_with_valid_event_field(
        self, node, retry_count, outcome, extra
    ):
        """log_node_event with event='start' emits valid JSON with event in {"start", "end"}."""
        # Clear any records from previous hypothesis examples
        self.handler.records.clear()

        log_node_event(node, "start", retry_count=retry_count, outcome=outcome, extra=extra)

        assert len(self.handler.records) >= 1
        msg = self.handler.records[-1].getMessage()
        parsed = json.loads(msg)
        assert isinstance(parsed, dict)
        assert parsed["event"] in {"start", "end"}

    @settings(max_examples=100)
    @given(
        node=st.text(),
        retry_count=st.integers(),
        outcome=st.one_of(st.none(), st.text()),
        extra=_json_safe_dict,
    )
    def test_end_event_emits_parseable_json_with_valid_event_field(
        self, node, retry_count, outcome, extra
    ):
        """log_node_event with event='end' emits valid JSON with event in {"start", "end"}."""
        # Clear any records from previous hypothesis examples
        self.handler.records.clear()

        log_node_event(node, "end", retry_count=retry_count, outcome=outcome, extra=extra)

        assert len(self.handler.records) >= 1
        msg = self.handler.records[-1].getMessage()
        parsed = json.loads(msg)
        assert isinstance(parsed, dict)
        assert parsed["event"] in {"start", "end"}


# --------------------------------------------------------------------------- #
# Task 22.2 — Verify structured JSON logs are emitted and parseable
# --------------------------------------------------------------------------- #
# These tests verify:
# 1. log_node_event emits JSON with all required keys (ts, node, event,
#    retry_count) — verifying Requirement 9.1 / 9.5
# 2. The run-summary log entry emitted by agent.py is valid JSON with the
#    expected fields (event, success, retry_count, elapsed_seconds, ts,
#    confidence_score, token_usage, refactor_diff) — verifying Req 9.3 / 9.4
# 3. When a StreamHandler writing to stdout is attached, every captured log
#    line is parseable by json.loads() — verifying Req 9.5
# --------------------------------------------------------------------------- #

import io
import sys
from unittest.mock import MagicMock, patch


class TestTask22LogNodeEventJsonOutput:
    """
    Task 22.2 — Verify log_node_event emits valid JSON to stdout.

    Attaches a StreamHandler pointing to a StringIO buffer (simulating
    stdout) and confirms every emitted line is parseable by json.loads()
    and contains the required keys.
    """

    def _make_stdout_handler(self, stream: io.StringIO) -> logging.StreamHandler:
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        return handler

    def test_log_node_event_stdout_output_is_valid_json(self):
        """Each line written to stdout by log_node_event must be parseable by json.loads()."""
        buf = io.StringIO()
        handler = self._make_stdout_handler(buf)
        logger = logging.getLogger("coder_buddy")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        try:
            log_node_event("write_node", "start", retry_count=0)
            log_node_event("execute_node", "end", retry_count=1, outcome="retry")
        finally:
            logger.removeHandler(handler)

        output = buf.getvalue().strip()
        lines = [line for line in output.splitlines() if line.strip()]
        assert len(lines) == 2, f"Expected 2 log lines, got {len(lines)}: {lines!r}"
        for line in lines:
            parsed = json.loads(line)  # raises json.JSONDecodeError if not valid JSON
            assert isinstance(parsed, dict)

    def test_log_node_event_stdout_required_keys_present(self):
        """Every log line written to stdout must contain ts, node, event, retry_count."""
        buf = io.StringIO()
        handler = self._make_stdout_handler(buf)
        logger = logging.getLogger("coder_buddy")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        try:
            log_node_event("write_node", "start", retry_count=0)
            log_node_event("evaluator", "end", retry_count=2, outcome="refactor")
        finally:
            logger.removeHandler(handler)

        lines = [line for line in buf.getvalue().splitlines() if line.strip()]
        for line in lines:
            entry = json.loads(line)
            for key in ("ts", "node", "event", "retry_count"):
                assert key in entry, (
                    f"Required key '{key}' missing from log entry: {entry!r}"
                )

    def test_log_node_event_multiple_nodes_all_valid_json(self):
        """Simulate a full node lifecycle (start + end for multiple nodes) — all lines valid JSON."""
        buf = io.StringIO()
        handler = self._make_stdout_handler(buf)
        logger = logging.getLogger("coder_buddy")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        try:
            nodes = ["write_node", "execute_node", "evaluator", "refactor_node", "post_process"]
            for node in nodes:
                log_node_event(node, "start", retry_count=0)
                log_node_event(node, "end", retry_count=0, outcome="ok")
        finally:
            logger.removeHandler(handler)

        lines = [line for line in buf.getvalue().splitlines() if line.strip()]
        assert len(lines) == len(nodes) * 2, (
            f"Expected {len(nodes) * 2} log lines, got {len(lines)}"
        )
        for line in lines:
            parsed = json.loads(line)
            assert isinstance(parsed, dict)
            assert parsed["event"] in {"start", "end"}


class TestTask22RunSummaryLogEntry:
    """
    Task 22.2 — Verify the run-summary log entry emitted by agent.py is
    valid JSON with the expected fields.

    The run-summary is emitted by CoderBuddy.run() after the graph
    completes.  We mock the graph and sandbox so no real I/O occurs, then
    capture the log output and verify the final entry.
    """

    def _make_config(self, **overrides):
        from coder_buddy.config import AgentConfig
        defaults = {
            "llm_backend": "gemini-2.5-flash",
            "sandbox_backend": "subprocess+venv",
            "max_retries": 3,
            "explanation_enabled": False,
            "test_generation_enabled": False,
            "diff_view_enabled": False,
        }
        defaults.update(overrides)
        return AgentConfig(**defaults)

    def _make_success_final_state(self) -> dict:
        from coder_buddy.models import TokenUsage
        return {
            "current_code": "print('hello')",
            "file_name": "main.py",
            "dependencies": [],
            "execution_logs": "hello\n",
            "error_status": False,
            "retry_count": 1,
            "explanation": None,
            "test_code": None,
            "test_logs": None,
            "confidence_score": 4,
            "refactor_diff": "--- a\n+++ b\n",
            "token_usage": TokenUsage(),
            "warning": None,
            "max_retries": 3,
            "pre_refactor_code": None,
            "session_history": [],
            "user_prompt": "write hello world",
            "language": "python",
            "_route": None,
        }

    def _build_patched_agent(self, final_state: dict):
        """Build a CoderBuddy with all I/O mocked, return (agent, mock_graph)."""
        from coder_buddy.agent import CoderBuddy

        config = self._make_config()
        with (
            patch("coder_buddy.agent._make_sandbox") as mock_make_sandbox,
            patch("coder_buddy.agent.LLMClient"),
            patch("coder_buddy.agent.build_graph") as mock_build_graph,
        ):
            mock_sandbox = MagicMock()
            mock_sandbox.health_check.return_value = None
            mock_make_sandbox.return_value = mock_sandbox

            mock_graph = MagicMock()
            mock_graph.invoke.return_value = final_state
            mock_build_graph.return_value = mock_graph

            agent = CoderBuddy(config)

        # Patch the graph on the already-constructed agent so invoke uses our mock
        agent._graph = mock_graph
        return agent, mock_graph

    def test_run_summary_log_entry_is_valid_json(self):
        """The run-summary log entry emitted by agent.run() must be parseable by json.loads()."""
        final_state = self._make_success_final_state()
        agent, mock_graph = self._build_patched_agent(final_state)

        handler = make_handler()
        try:
            agent.run("write hello world")
        finally:
            remove_handler(handler)

        # The last record should be the run-summary entry
        assert len(handler.records) >= 1, "Expected at least one log record"
        last_msg = handler.records[-1].getMessage()
        parsed = json.loads(last_msg)  # raises if not valid JSON
        assert isinstance(parsed, dict)

    def test_run_summary_log_entry_has_required_fields(self):
        """The run-summary entry must contain: event, success, retry_count, elapsed_seconds, ts."""
        final_state = self._make_success_final_state()
        agent, mock_graph = self._build_patched_agent(final_state)

        handler = make_handler()
        try:
            agent.run("write hello world")
        finally:
            remove_handler(handler)

        last_msg = handler.records[-1].getMessage()
        entry = json.loads(last_msg)

        required_keys = ("ts", "event", "success", "retry_count", "elapsed_seconds")
        for key in required_keys:
            assert key in entry, (
                f"Required key '{key}' missing from run-summary entry: {entry!r}"
            )

    def test_run_summary_log_entry_event_is_run_complete(self):
        """The run-summary entry must have event == 'run_complete'."""
        final_state = self._make_success_final_state()
        agent, mock_graph = self._build_patched_agent(final_state)

        handler = make_handler()
        try:
            agent.run("write hello world")
        finally:
            remove_handler(handler)

        last_msg = handler.records[-1].getMessage()
        entry = json.loads(last_msg)
        assert entry["event"] == "run_complete", (
            f"Expected event='run_complete', got event={entry.get('event')!r}"
        )

    def test_run_summary_log_entry_success_field_matches_outcome(self):
        """The success field in the run-summary must match the actual run outcome."""
        final_state = self._make_success_final_state()
        agent, mock_graph = self._build_patched_agent(final_state)

        handler = make_handler()
        try:
            result = agent.run("write hello world")
        finally:
            remove_handler(handler)

        last_msg = handler.records[-1].getMessage()
        entry = json.loads(last_msg)
        assert entry["success"] == result.success, (
            f"Log entry success={entry['success']!r} does not match "
            f"AgentResponse.success={result.success!r}"
        )

    def test_run_summary_log_entry_retry_count_matches_outcome(self):
        """The retry_count in the run-summary must match the final state's retry_count."""
        final_state = self._make_success_final_state()
        agent, mock_graph = self._build_patched_agent(final_state)

        handler = make_handler()
        try:
            result = agent.run("write hello world")
        finally:
            remove_handler(handler)

        last_msg = handler.records[-1].getMessage()
        entry = json.loads(last_msg)
        assert entry["retry_count"] == result.retry_count, (
            f"Log entry retry_count={entry['retry_count']!r} does not match "
            f"AgentResponse.retry_count={result.retry_count!r}"
        )

    def test_run_summary_log_entry_has_token_usage_field(self):
        """The run-summary entry must contain a token_usage dict."""
        final_state = self._make_success_final_state()
        agent, mock_graph = self._build_patched_agent(final_state)

        handler = make_handler()
        try:
            agent.run("write hello world")
        finally:
            remove_handler(handler)

        last_msg = handler.records[-1].getMessage()
        entry = json.loads(last_msg)
        assert "token_usage" in entry, (
            f"Expected 'token_usage' key in run-summary entry: {entry!r}"
        )
        assert isinstance(entry["token_usage"], dict), (
            f"Expected token_usage to be a dict, got {type(entry['token_usage'])!r}"
        )

    def test_run_summary_log_entry_has_confidence_score_field(self):
        """The run-summary entry must contain a confidence_score field."""
        final_state = self._make_success_final_state()
        agent, mock_graph = self._build_patched_agent(final_state)

        handler = make_handler()
        try:
            agent.run("write hello world")
        finally:
            remove_handler(handler)

        last_msg = handler.records[-1].getMessage()
        entry = json.loads(last_msg)
        assert "confidence_score" in entry, (
            f"Expected 'confidence_score' key in run-summary entry: {entry!r}"
        )

    def test_run_summary_log_entry_has_refactor_diff_field(self):
        """The run-summary entry must contain a refactor_diff field."""
        final_state = self._make_success_final_state()
        agent, mock_graph = self._build_patched_agent(final_state)

        handler = make_handler()
        try:
            agent.run("write hello world")
        finally:
            remove_handler(handler)

        last_msg = handler.records[-1].getMessage()
        entry = json.loads(last_msg)
        assert "refactor_diff" in entry, (
            f"Expected 'refactor_diff' key in run-summary entry: {entry!r}"
        )

    def test_run_summary_log_entry_elapsed_seconds_is_positive(self):
        """elapsed_seconds in the run-summary must be a positive number."""
        final_state = self._make_success_final_state()
        agent, mock_graph = self._build_patched_agent(final_state)

        handler = make_handler()
        try:
            agent.run("write hello world")
        finally:
            remove_handler(handler)

        last_msg = handler.records[-1].getMessage()
        entry = json.loads(last_msg)
        assert isinstance(entry["elapsed_seconds"], (int, float)), (
            f"Expected elapsed_seconds to be numeric, got {type(entry['elapsed_seconds'])!r}"
        )
        assert entry["elapsed_seconds"] >= 0, (
            f"Expected elapsed_seconds >= 0, got {entry['elapsed_seconds']!r}"
        )

    def test_run_summary_log_entry_ts_is_positive_float(self):
        """ts in the run-summary must be a positive Unix timestamp."""
        final_state = self._make_success_final_state()
        agent, mock_graph = self._build_patched_agent(final_state)

        handler = make_handler()
        try:
            agent.run("write hello world")
        finally:
            remove_handler(handler)

        last_msg = handler.records[-1].getMessage()
        entry = json.loads(last_msg)
        assert isinstance(entry["ts"], float), (
            f"Expected ts to be a float, got {type(entry['ts'])!r}"
        )
        assert entry["ts"] > 0, f"Expected ts > 0, got {entry['ts']!r}"

    def test_all_log_entries_in_a_run_are_valid_json(self):
        """Every log entry emitted during a full agent.run() call must be valid JSON."""
        final_state = self._make_success_final_state()
        agent, mock_graph = self._build_patched_agent(final_state)

        handler = make_handler()
        try:
            agent.run("write hello world")
        finally:
            remove_handler(handler)

        assert len(handler.records) >= 1, "Expected at least one log record during run()"
        for record in handler.records:
            msg = record.getMessage()
            parsed = json.loads(msg)  # raises json.JSONDecodeError if not valid JSON
            assert isinstance(parsed, dict), (
                f"Expected log entry to be a JSON object, got {type(parsed)!r}: {msg!r}"
            )

    def test_stdout_stream_handler_produces_parseable_json_lines(self):
        """
        When a StreamHandler writing to stdout is attached to the coder_buddy
        logger, every line captured during agent.run() must be parseable by
        json.loads() — verifying Requirement 9.5.
        """
        final_state = self._make_success_final_state()
        agent, mock_graph = self._build_patched_agent(final_state)

        # Redirect stdout to a buffer to capture what would be written to stdout
        buf = io.StringIO()
        stdout_handler = logging.StreamHandler(buf)
        stdout_handler.setFormatter(logging.Formatter("%(message)s"))

        logger = logging.getLogger("coder_buddy")
        logger.addHandler(stdout_handler)
        logger.setLevel(logging.DEBUG)
        try:
            agent.run("write hello world")
        finally:
            logger.removeHandler(stdout_handler)

        output = buf.getvalue().strip()
        assert output, "Expected log output to stdout, got empty string"

        lines = [line for line in output.splitlines() if line.strip()]
        assert len(lines) >= 1, f"Expected at least 1 log line, got {len(lines)}"
        for line in lines:
            parsed = json.loads(line)  # raises json.JSONDecodeError if not valid JSON
            assert isinstance(parsed, dict), (
                f"Expected each stdout log line to be a JSON object: {line!r}"
            )
