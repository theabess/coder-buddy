"""
Per-token cost table for supported LLM models.

Prices are expressed as cost per 1 000 tokens (input, output).
``AgentConfig.price_per_input_token`` / ``price_per_output_token`` can
override these values at runtime.
"""

from __future__ import annotations

# model_name → (price_per_1k_input_tokens, price_per_1k_output_tokens)
KNOWN_PRICES: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash": (0.00015, 0.0006),   # non-thinking tier
    "gemini-2.5-pro":   (0.00125, 0.010),
    "gpt-4o":           (0.005,   0.015),
    "claude-3-5-sonnet": (0.003,  0.015),
}


def get_price(model: str) -> tuple[float, float] | None:
    """
    Return the ``(input_price, output_price)`` tuple for *model*.

    Args:
        model: Model identifier string.

    Returns:
        A ``(price_per_1k_input, price_per_1k_output)`` tuple, or
        ``None`` if the model is not in ``KNOWN_PRICES``.
    """
    return KNOWN_PRICES.get(model)


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    price_per_input_token: float | None = None,
    price_per_output_token: float | None = None,
) -> float | None:
    """
    Estimate the cost in USD for a single LLM call.

    Uses *price_per_input_token* / *price_per_output_token* when provided;
    otherwise falls back to ``KNOWN_PRICES``.

    Args:
        model:                 Model identifier string.
        input_tokens:          Number of input tokens consumed.
        output_tokens:         Number of output tokens produced.
        price_per_input_token: Override price per input token (not per 1k).
        price_per_output_token: Override price per output token (not per 1k).

    Returns:
        Estimated cost in USD, or ``None`` if no price data is available.
    """
    if price_per_input_token is not None and price_per_output_token is not None:
        return (input_tokens * price_per_input_token) + (
            output_tokens * price_per_output_token
        )

    prices = get_price(model)
    if prices is None:
        return None

    input_price_per_token = prices[0] / 1000
    output_price_per_token = prices[1] / 1000
    return (input_tokens * input_price_per_token) + (
        output_tokens * output_price_per_token
    )
