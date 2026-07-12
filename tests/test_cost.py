"""Tests for the cost model."""

from __future__ import annotations

import pytest

from llm_bench.cost import Price, cost_for_request, cost_per_1m


def test_cost_for_request_basic():
    price = Price(input_per_1m=5.0, output_per_1m=15.0)
    # 1M input tokens -> $5, 1M output -> $15
    assert cost_for_request(price, 1_000_000, 1_000_000) == 20.0
    # 500k each -> $2.5 + $7.5
    assert cost_for_request(price, 500_000, 500_000) == 10.0


def test_cost_for_request_zero_tokens():
    price = Price(input_per_1m=5.0, output_per_1m=15.0)
    assert cost_for_request(price, 0, 0) == 0.0


def test_cost_per_1m_normalises_mix():
    price = Price(input_per_1m=5.0, output_per_1m=15.0)
    # Request with 1 input + 1 output token costs 20e-6 USD.
    # Normalised to 1M tokens: 20e-6 * 1_000_000 / 2 = 10.0
    assert cost_per_1m(price, 1, 1) == 10.0


def test_cost_per_1m_zero_tokens_raises():
    price = Price(input_per_1m=5.0, output_per_1m=15.0)
    with pytest.raises(ValueError):
        cost_per_1m(price, 0, 0)
