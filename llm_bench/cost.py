"""Cost model for llm-bench.

Pricing is expressed as a per-model :class:`Price` (USD per 1M input/output
tokens), matching the way providers actually publish their rate cards. From a
single request we can derive both the concrete cost of that request and a
normalised ``$/1M tokens`` figure useful for provider-to-provider comparison.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    """USD price per 1M input and output tokens."""

    input_per_1m: float
    output_per_1m: float


def cost_for_request(
    price: Price, input_tokens: int, output_tokens: int
) -> float:
    """Cost in USD of a single request.

    Args:
        price: Per-model pricing.
        input_tokens: Prompt token count.
        output_tokens: Completion token count.

    Returns:
        The request cost in USD.
    """
    in_cost = (input_tokens / 1_000_000.0) * price.input_per_1m
    out_cost = (output_tokens / 1_000_000.0) * price.output_per_1m
    return in_cost + out_cost


def cost_per_1m(
    price: Price, input_tokens: int, output_tokens: int
) -> float:
    """Normalise a request's cost to USD per 1M total tokens.

    This lets two requests (or two providers) with different prompt/completion
    mixes be compared on a common ``$/1M-token`` basis.

    Args:
        price: Per-model pricing.
        input_tokens: Prompt token count.
        output_tokens: Completion token count.

    Returns:
        USD per 1M tokens for the given mix.

    Raises:
        ValueError: If the total token count is zero.
    """
    total = input_tokens + output_tokens
    if total == 0:
        raise ValueError("cost_per_1m requires a non-zero token count")
    return cost_for_request(price, input_tokens, output_tokens) * (
        1_000_000.0 / total
    )
