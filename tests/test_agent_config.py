"""
Unit tests for AgentConfig dataclass validation.

Validates:
- max_retries boundary values (1, 5, 10) are accepted
- max_retries below range (0, -1) raises ValueError
- max_retries above range (11, 100) raises ValueError
- Invalid sandbox_backend names raise ValueError
- Invalid llm_backend names raise ValueError
- All valid sandbox_backend names are accepted
- All valid llm_backend names are accepted
"""

import pytest

from coder_buddy.config import (
    AgentConfig,
    VALID_LLM_BACKENDS,
    VALID_SANDBOX_BACKENDS,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

VALID_DEFAULTS = {
    "llm_backend": "gemini-2.5-flash",
    "sandbox_backend": "subprocess+venv",
    "max_retries": 5,
}


def make_config(**overrides) -> AgentConfig:
    """Return an AgentConfig built from VALID_DEFAULTS merged with overrides."""
    kwargs = {**VALID_DEFAULTS, **overrides}
    return AgentConfig(**kwargs)


# --------------------------------------------------------------------------- #
# max_retries — valid range
# --------------------------------------------------------------------------- #


class TestMaxRetriesValidRange:
    def test_max_retries_lower_boundary(self):
        """max_retries=1 is the minimum allowed value."""
        cfg = make_config(max_retries=1)
        assert cfg.max_retries == 1

    def test_max_retries_midpoint(self):
        """max_retries=5 (default) is within the valid range."""
        cfg = make_config(max_retries=5)
        assert cfg.max_retries == 5

    def test_max_retries_upper_boundary(self):
        """max_retries=10 is the maximum allowed value."""
        cfg = make_config(max_retries=10)
        assert cfg.max_retries == 10


# --------------------------------------------------------------------------- #
# max_retries — below range
# --------------------------------------------------------------------------- #


class TestMaxRetriesBelowRange:
    def test_max_retries_zero_raises(self):
        """max_retries=0 is below the minimum of 1."""
        with pytest.raises(ValueError, match="max_retries"):
            make_config(max_retries=0)

    def test_max_retries_negative_raises(self):
        """max_retries=-1 is below the minimum of 1."""
        with pytest.raises(ValueError, match="max_retries"):
            make_config(max_retries=-1)


# --------------------------------------------------------------------------- #
# max_retries — above range
# --------------------------------------------------------------------------- #


class TestMaxRetriesAboveRange:
    def test_max_retries_eleven_raises(self):
        """max_retries=11 is above the maximum of 10."""
        with pytest.raises(ValueError, match="max_retries"):
            make_config(max_retries=11)

    def test_max_retries_hundred_raises(self):
        """max_retries=100 is well above the maximum of 10."""
        with pytest.raises(ValueError, match="max_retries"):
            make_config(max_retries=100)


# --------------------------------------------------------------------------- #
# sandbox_backend — invalid names
# --------------------------------------------------------------------------- #


class TestSandboxBackendInvalidNames:
    def test_unknown_sandbox_backend_raises(self):
        with pytest.raises(ValueError, match="sandbox_backend"):
            make_config(sandbox_backend="unknown-backend")

    def test_empty_sandbox_backend_raises(self):
        with pytest.raises(ValueError, match="sandbox_backend"):
            make_config(sandbox_backend="")

    def test_misspelled_sandbox_backend_raises(self):
        with pytest.raises(ValueError, match="sandbox_backend"):
            make_config(sandbox_backend="Docker")  # case-sensitive

    def test_partial_sandbox_backend_raises(self):
        with pytest.raises(ValueError, match="sandbox_backend"):
            make_config(sandbox_backend="docker-compose")


# --------------------------------------------------------------------------- #
# llm_backend — invalid names
# --------------------------------------------------------------------------- #


class TestLLMBackendInvalidNames:
    def test_unknown_llm_backend_raises(self):
        with pytest.raises(ValueError, match="llm_backend"):
            make_config(llm_backend="gpt-3")

    def test_empty_llm_backend_raises(self):
        with pytest.raises(ValueError, match="llm_backend"):
            make_config(llm_backend="")

    def test_misspelled_llm_backend_raises(self):
        with pytest.raises(ValueError, match="llm_backend"):
            make_config(llm_backend="GPT-4o")  # case-sensitive

    def test_partial_llm_backend_raises(self):
        with pytest.raises(ValueError, match="llm_backend"):
            make_config(llm_backend="gemini")


# --------------------------------------------------------------------------- #
# sandbox_backend — all valid names
# --------------------------------------------------------------------------- #


class TestSandboxBackendValidNames:
    def test_subprocess_venv(self):
        cfg = make_config(sandbox_backend="subprocess+venv")
        assert cfg.sandbox_backend == "subprocess+venv"

    def test_docker(self):
        cfg = make_config(sandbox_backend="docker")
        assert cfg.sandbox_backend == "docker"

    def test_e2b(self):
        cfg = make_config(sandbox_backend="e2b")
        assert cfg.sandbox_backend == "e2b"

    def test_pyodide(self):
        cfg = make_config(sandbox_backend="pyodide")
        assert cfg.sandbox_backend == "pyodide"

    def test_all_valid_sandbox_backends_accepted(self):
        """Every entry in VALID_SANDBOX_BACKENDS must construct without error."""
        for backend in VALID_SANDBOX_BACKENDS:
            cfg = make_config(sandbox_backend=backend)
            assert cfg.sandbox_backend == backend


# --------------------------------------------------------------------------- #
# llm_backend — all valid names
# --------------------------------------------------------------------------- #


class TestLLMBackendValidNames:
    def test_gemini_1_5_pro(self):
        cfg = make_config(llm_backend="gemini-2.5-flash")
        assert cfg.llm_backend == "gemini-2.5-flash"

    def test_gpt_4o(self):
        cfg = make_config(llm_backend="gpt-4o")
        assert cfg.llm_backend == "gpt-4o"

    def test_claude_3_5_sonnet(self):
        cfg = make_config(llm_backend="claude-3-5-sonnet")
        assert cfg.llm_backend == "claude-3-5-sonnet"

    def test_all_valid_llm_backends_accepted(self):
        """Every entry in VALID_LLM_BACKENDS must construct without error."""
        for backend in VALID_LLM_BACKENDS:
            cfg = make_config(llm_backend=backend)
            assert cfg.llm_backend == backend


# --------------------------------------------------------------------------- #
# Property-based tests
# --------------------------------------------------------------------------- #


from hypothesis import given, settings
from hypothesis import strategies as st


class TestProperty8MaxRetriesRange:
    """
    Property 8: AgentConfig(max_retries=n) succeeds for 1 ≤ n ≤ 10
    and raises ValueError for n < 1 or n > 10.

    Feature: coder-buddy, Property 8: AgentConfig(max_retries=n) succeeds
    for 1 ≤ n ≤ 10 and raises ValueError for n < 1 or n > 10.
    """

    @given(n=st.integers(min_value=1, max_value=10))
    @settings(max_examples=100)
    def test_valid_range_succeeds(self, n: int):
        """Any integer in [1, 10] must construct without error."""
        cfg = make_config(max_retries=n)
        assert cfg.max_retries == n

    @given(n=st.integers(max_value=0))
    @settings(max_examples=100)
    def test_below_range_raises(self, n: int):
        """Any integer ≤ 0 must raise ValueError."""
        with pytest.raises(ValueError, match="max_retries"):
            make_config(max_retries=n)

    @given(n=st.integers(min_value=11))
    @settings(max_examples=100)
    def test_above_range_raises(self, n: int):
        """Any integer ≥ 11 must raise ValueError."""
        with pytest.raises(ValueError, match="max_retries"):
            make_config(max_retries=n)
