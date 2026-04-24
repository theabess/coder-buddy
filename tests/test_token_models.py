"""
Unit tests for TokenRecord and TokenUsage Pydantic models.

Validates:
- TokenRecord construction and field types
- TokenUsage aggregation: total_input_tokens, total_output_tokens, total_estimated_cost_usd
- Edge cases: all zeros, partial costs, no costs
"""

import pytest
from pydantic import ValidationError

from coder_buddy.models import TokenRecord, TokenUsage


class TestTokenRecord:
    def test_basic_construction(self):
        record = TokenRecord(input_tokens=100, output_tokens=50)
        assert record.input_tokens == 100
        assert record.output_tokens == 50
        assert record.estimated_cost_usd is None

    def test_with_cost(self):
        record = TokenRecord(input_tokens=100, output_tokens=50, estimated_cost_usd=0.0025)
        assert record.estimated_cost_usd == pytest.approx(0.0025)

    def test_zero_tokens(self):
        record = TokenRecord(input_tokens=0, output_tokens=0)
        assert record.input_tokens == 0
        assert record.output_tokens == 0

    def test_cost_none_by_default(self):
        record = TokenRecord(input_tokens=10, output_tokens=5)
        assert record.estimated_cost_usd is None


class TestTokenUsageDefaults:
    def test_default_construction_all_zeros(self):
        usage = TokenUsage()
        assert usage.total_input_tokens == 0
        assert usage.total_output_tokens == 0
        assert usage.total_estimated_cost_usd is None

    def test_default_node_records_are_zero(self):
        usage = TokenUsage()
        for record in [
            usage.write_node,
            usage.refactor_node,
            usage.explanation,
            usage.test_node,
            usage.confidence,
        ]:
            assert record.input_tokens == 0
            assert record.output_tokens == 0
            assert record.estimated_cost_usd is None


class TestTokenUsageAggregation:
    def test_total_input_tokens_sums_all_nodes(self):
        usage = TokenUsage(
            write_node=TokenRecord(input_tokens=100, output_tokens=0),
            refactor_node=TokenRecord(input_tokens=200, output_tokens=0),
            explanation=TokenRecord(input_tokens=50, output_tokens=0),
            test_node=TokenRecord(input_tokens=75, output_tokens=0),
            confidence=TokenRecord(input_tokens=25, output_tokens=0),
        )
        assert usage.total_input_tokens == 450

    def test_total_output_tokens_sums_all_nodes(self):
        usage = TokenUsage(
            write_node=TokenRecord(input_tokens=0, output_tokens=300),
            refactor_node=TokenRecord(input_tokens=0, output_tokens=150),
            explanation=TokenRecord(input_tokens=0, output_tokens=80),
            test_node=TokenRecord(input_tokens=0, output_tokens=60),
            confidence=TokenRecord(input_tokens=0, output_tokens=20),
        )
        assert usage.total_output_tokens == 610

    def test_total_tokens_with_all_nodes_populated(self):
        usage = TokenUsage(
            write_node=TokenRecord(input_tokens=100, output_tokens=50),
            refactor_node=TokenRecord(input_tokens=80, output_tokens=40),
            explanation=TokenRecord(input_tokens=60, output_tokens=30),
            test_node=TokenRecord(input_tokens=40, output_tokens=20),
            confidence=TokenRecord(input_tokens=20, output_tokens=10),
        )
        assert usage.total_input_tokens == 300
        assert usage.total_output_tokens == 150

    def test_total_cost_sums_all_node_costs(self):
        usage = TokenUsage(
            write_node=TokenRecord(input_tokens=100, output_tokens=50, estimated_cost_usd=0.001),
            refactor_node=TokenRecord(input_tokens=80, output_tokens=40, estimated_cost_usd=0.002),
            explanation=TokenRecord(input_tokens=60, output_tokens=30, estimated_cost_usd=0.0005),
            test_node=TokenRecord(input_tokens=40, output_tokens=20, estimated_cost_usd=0.0003),
            confidence=TokenRecord(input_tokens=20, output_tokens=10, estimated_cost_usd=0.0002),
        )
        assert usage.total_estimated_cost_usd == pytest.approx(0.004)

    def test_total_cost_none_when_no_costs_set(self):
        usage = TokenUsage(
            write_node=TokenRecord(input_tokens=100, output_tokens=50),
            refactor_node=TokenRecord(input_tokens=80, output_tokens=40),
        )
        assert usage.total_estimated_cost_usd is None

    def test_total_cost_partial_costs_only_sums_non_none(self):
        """Only nodes with cost set contribute to the total."""
        usage = TokenUsage(
            write_node=TokenRecord(input_tokens=100, output_tokens=50, estimated_cost_usd=0.005),
            refactor_node=TokenRecord(input_tokens=80, output_tokens=40),  # no cost
            explanation=TokenRecord(input_tokens=60, output_tokens=30, estimated_cost_usd=0.003),
        )
        assert usage.total_estimated_cost_usd == pytest.approx(0.008)

    def test_only_write_node_populated(self):
        usage = TokenUsage(
            write_node=TokenRecord(input_tokens=500, output_tokens=200, estimated_cost_usd=0.01),
        )
        assert usage.total_input_tokens == 500
        assert usage.total_output_tokens == 200
        assert usage.total_estimated_cost_usd == pytest.approx(0.01)

    def test_model_copy_update_preserves_other_nodes(self):
        """Verify model_copy(update=...) pattern used by nodes works correctly."""
        original = TokenUsage(
            write_node=TokenRecord(input_tokens=100, output_tokens=50),
        )
        updated = original.model_copy(
            update={"refactor_node": TokenRecord(input_tokens=80, output_tokens=40)}
        )
        # Original unchanged
        assert original.refactor_node.input_tokens == 0
        # Updated has new refactor_node
        assert updated.refactor_node.input_tokens == 80
        # write_node preserved
        assert updated.write_node.input_tokens == 100
        # Totals reflect both
        assert updated.total_input_tokens == 180
        assert updated.total_output_tokens == 90

    def test_all_records_returns_five_records(self):
        usage = TokenUsage()
        records = usage._all_records()
        assert len(records) == 5

    def test_all_records_order(self):
        """_all_records returns nodes in the expected order."""
        usage = TokenUsage(
            write_node=TokenRecord(input_tokens=1, output_tokens=0),
            refactor_node=TokenRecord(input_tokens=2, output_tokens=0),
            explanation=TokenRecord(input_tokens=3, output_tokens=0),
            test_node=TokenRecord(input_tokens=4, output_tokens=0),
            confidence=TokenRecord(input_tokens=5, output_tokens=0),
        )
        records = usage._all_records()
        assert [r.input_tokens for r in records] == [1, 2, 3, 4, 5]


# Feature: coder-buddy, Property 22: TokenUsage.total_input_tokens and total_output_tokens equal the arithmetic sum of per-node records
from hypothesis import given, settings
import hypothesis.strategies as st


@given(
    write_in=st.integers(min_value=0, max_value=10_000),
    write_out=st.integers(min_value=0, max_value=10_000),
    refactor_in=st.integers(min_value=0, max_value=10_000),
    refactor_out=st.integers(min_value=0, max_value=10_000),
    explanation_in=st.integers(min_value=0, max_value=10_000),
    explanation_out=st.integers(min_value=0, max_value=10_000),
    test_in=st.integers(min_value=0, max_value=10_000),
    test_out=st.integers(min_value=0, max_value=10_000),
    confidence_in=st.integers(min_value=0, max_value=10_000),
    confidence_out=st.integers(min_value=0, max_value=10_000),
)
def test_property_22_total_tokens_equal_arithmetic_sum(
    write_in, write_out,
    refactor_in, refactor_out,
    explanation_in, explanation_out,
    test_in, test_out,
    confidence_in, confidence_out,
):
    """**Validates: Requirements 2.12**

    Property 22: TokenUsage.total_input_tokens and total_output_tokens
    always equal the arithmetic sum of per-node TokenRecord fields.
    """
    usage = TokenUsage(
        write_node=TokenRecord(input_tokens=write_in, output_tokens=write_out),
        refactor_node=TokenRecord(input_tokens=refactor_in, output_tokens=refactor_out),
        explanation=TokenRecord(input_tokens=explanation_in, output_tokens=explanation_out),
        test_node=TokenRecord(input_tokens=test_in, output_tokens=test_out),
        confidence=TokenRecord(input_tokens=confidence_in, output_tokens=confidence_out),
    )

    expected_input = write_in + refactor_in + explanation_in + test_in + confidence_in
    expected_output = write_out + refactor_out + explanation_out + test_out + confidence_out

    assert usage.total_input_tokens == expected_input
    assert usage.total_output_tokens == expected_output


# Feature: coder-buddy, Property 23: estimated_cost_usd equals (input_tokens * price_per_input) + (output_tokens * price_per_output)
@given(
    input_tokens=st.integers(min_value=0, max_value=10_000_000),
    output_tokens=st.integers(min_value=0, max_value=10_000_000),
    price_per_input=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    price_per_output=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_property_23_estimated_cost_usd_equals_formula(
    input_tokens,
    output_tokens,
    price_per_input,
    price_per_output,
):
    """**Validates: Requirements 2.13**

    Property 23: estimated_cost_usd stored in a TokenRecord equals
    (input_tokens * price_per_input) + (output_tokens * price_per_output).
    """
    expected_cost = (input_tokens * price_per_input) + (output_tokens * price_per_output)
    record = TokenRecord(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=expected_cost,
    )

    assert record.estimated_cost_usd == pytest.approx(expected_cost)
