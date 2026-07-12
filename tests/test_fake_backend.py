"""Tests for the FakeBackend and metrics integration (no network)."""

from __future__ import annotations

import time

import pytest

from llm_bench.backends import FakeBackend, TokenEvent
from llm_bench.harness import Runner
from llm_bench.metrics import compute_metrics


def test_fake_backend_yields_deterministic_tokens():
    clock = lambda: 0.0  # start at t=0 so deltas are exact
    backend = FakeBackend(tokens=["a", "b", "c"], ttft=0.2, tpot=0.1, clock=clock)
    events = list(backend.stream("hi"))
    assert [e.token for e in events] == ["a", "b", "c"]
    # TTFT: first event fires exactly ttft after start.
    assert events[0].timestamp == pytest.approx(0.2)
    # TPOT: each subsequent token is exactly tpot later.
    assert events[1].timestamp - events[0].timestamp == pytest.approx(0.1)
    assert events[2].timestamp - events[1].timestamp == pytest.approx(0.1)


def test_fake_backend_is_reproducible():
    clock = lambda: 0.0
    backend = FakeBackend(tokens=["x", "y"], ttft=0.05, tpot=0.02, clock=clock)
    first = list(backend.stream("p"))
    second = list(backend.stream("p"))
    assert [e.timestamp for e in first] == [e.timestamp for e in second]


def test_fake_backend_call_count():
    backend = FakeBackend(tokens=["a"], ttft=0.0, tpot=0.0)
    list(backend.stream("1"))
    list(backend.stream("2"))
    assert backend.call_count == 2


def test_compute_metrics_ttft_and_tpot():
    events = [
        TokenEvent(token="a", timestamp=1.0),
        TokenEvent(token="b", timestamp=1.5),
        TokenEvent(token="c", timestamp=2.0),
    ]
    start = 0.0
    m = compute_metrics(events, start)
    assert m.ttft == 1.0
    # (2.0 - 1.0) / (3 - 1) = 0.5
    assert m.tpot == 0.5
    # 3 tokens over 1.0s generation time = 3.0 tok/s
    assert m.throughput == 3.0
    assert m.total_latency == 2.0
    assert m.num_tokens == 3


def test_compute_metrics_single_token():
    events = [TokenEvent(token="a", timestamp=1.0)]
    m = compute_metrics(events, 0.0)
    assert m.ttft == 1.0
    assert m.tpot == 0.0
    assert m.throughput == 0.0
    assert m.num_tokens == 1


def test_runner_with_fake_backend_is_deterministic():
    # Fixed clock (always t=0) so every run is byte-for-byte identical and the
    # synthetic ttft/tpot survive exactly into the aggregate.
    clock = lambda: 0.0
    backend = FakeBackend(tokens=["a", "b", "c"], ttft=0.1, tpot=0.05, clock=clock)
    runner = Runner(
        backend=backend, prompts=["p1", "p2"], runs=3, clock=clock
    )
    agg, results = runner.run()
    assert agg.runs == 6
    assert agg.ttft_mean == pytest.approx(0.1)
    assert agg.tpot_mean == pytest.approx(0.05)
    assert agg.throughput_mean == pytest.approx(3.0 / (2 * 0.05))
    # All runs identical under FakeBackend.
    assert agg.ttft_p95 == pytest.approx(0.1)
    assert agg.tpot_p95 == pytest.approx(0.05)
    assert all(r.metrics.num_tokens == 3 for r in results)
