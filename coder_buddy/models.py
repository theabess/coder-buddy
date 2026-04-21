"""
Pydantic models for Coder Buddy.

Defines the core data models used throughout the agent:

- ``CodeArtifact``  — structured LLM output (source code + metadata)
- ``TokenRecord``   — per-node token usage
- ``TokenUsage``    — aggregated token usage across all nodes
- ``HistoryEntry``  — a single entry in the session history deque
- ``AgentResponse`` — the public response returned by ``CoderBuddy.run()``
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, field_validator


class CodeArtifact(BaseModel):
    """
    Structured output produced by the LLM for every code-generation call.

    Validators enforce that ``source_code`` is non-empty and that
    ``language`` is ``"python"`` (the only supported language in V1).
    """

    source_code: str
    file_name: str
    dependencies: list[str]
    language: str = "python"

    @field_validator("source_code")
    @classmethod
    def source_code_not_empty(cls, v: str) -> str:
        """Reject empty or whitespace-only source code."""
        if not v.strip():
            raise ValueError("source_code must be a non-empty string")
        return v

    @field_validator("language")
    @classmethod
    def language_must_be_python(cls, v: str) -> str:
        """Reject any language other than ``"python"``."""
        if v.lower() != "python":
            raise ValueError(
                f"Language '{v}' is not supported in V1; only 'python' is allowed"
            )
        return v.lower()


class TokenRecord(BaseModel):
    """Token usage for a single LLM call."""

    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float | None = None


class TokenUsage(BaseModel):
    """
    Accumulated token usage across all nodes in a single agent run.

    Each field corresponds to a node that makes LLM calls.  Computed
    properties aggregate the totals across all nodes.
    """

    write_node: TokenRecord = TokenRecord(input_tokens=0, output_tokens=0)
    refactor_node: TokenRecord = TokenRecord(input_tokens=0, output_tokens=0)
    explanation: TokenRecord = TokenRecord(input_tokens=0, output_tokens=0)
    test_node: TokenRecord = TokenRecord(input_tokens=0, output_tokens=0)
    confidence: TokenRecord = TokenRecord(input_tokens=0, output_tokens=0)

    @property
    def total_input_tokens(self) -> int:
        """Sum of ``input_tokens`` across all node records."""
        return sum(r.input_tokens for r in self._all_records())

    @property
    def total_output_tokens(self) -> int:
        """Sum of ``output_tokens`` across all node records."""
        return sum(r.output_tokens for r in self._all_records())

    @property
    def total_estimated_cost_usd(self) -> float | None:
        """
        Sum of ``estimated_cost_usd`` across all node records that have a
        non-``None`` cost.  Returns ``None`` if no cost data is available.
        """
        costs = [
            r.estimated_cost_usd
            for r in self._all_records()
            if r.estimated_cost_usd is not None
        ]
        return sum(costs) if costs else None

    def _all_records(self) -> list[TokenRecord]:
        """Return all per-node ``TokenRecord`` instances."""
        return [
            self.write_node,
            self.refactor_node,
            self.explanation,
            self.test_node,
            self.confidence,
        ]


class HistoryEntry(BaseModel):
    """A single entry stored in the session history deque."""

    prompt: str
    source_code: str
    file_name: str
    dependencies: list[str]
    timestamp: datetime


class AgentResponse(BaseModel):
    """
    The public response object returned by ``CoderBuddy.run()``.

    Always returned regardless of success or failure.  When
    ``success=False``, ``failure_reason`` is populated with a
    human-readable explanation.
    """

    success: bool
    source_code: str
    file_name: str
    dependencies: list[str]
    execution_logs: str
    retry_count: int

    # Optional enrichment fields
    explanation: str | None
    test_code: str | None
    confidence_score: int | None
    refactor_diff: str | None

    # Observability
    token_usage: TokenUsage
    elapsed_seconds: float

    # Failure details (populated when success=False)
    failure_reason: str | None = None
    warning: str | None = None
