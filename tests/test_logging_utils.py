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
