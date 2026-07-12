"""Latency and throughput metrics for llm-bench.

All metrics are derived from the per-token timestamps produced by a
:class:`~llm_bench.backends.Backend`. We measure:

* **TTFT** (time to first token): ``first_token_ts - start_ts``.
* **TPOT** (time per output token): ``(last_ts - first_ts) / (n - 1)``.
* **throughput**: output tokens per second of generation time.
* **total latency**: ``last_ts - start_ts`` (TTFT inclusive).

Timestamps are monotonic-clock seconds; ``start_ts`` is recorded by the caller
immediately before the stream begins.
"""

from __future__ import annotations

from dataclasses import dataclass

from llm_bench.backends import TokenEvent


@dataclass(frozen=True)
class Metrics:
    """Per-request latency/throughput measurements."""

    ttft: float
    tpot: float
    throughput: float
    total_latency: float
    num_tokens: int


def compute_metrics(
    events: list[TokenEvent], start_ts: float, end_ts: float | None = None
) -> Metrics:
    """Compute latency/throughput metrics from token events.

    Args:
        events: Ordered token events emitted for a single request.
        start_ts: Wall-clock time (seconds) recorded just before streaming.
        end_ts: Optional explicit end time. When omitted, the timestamp of the
            last token event is used.

    Returns:
        Metrics with TTFT, TPOT, throughput (tok/s), total latency and the
        token count.

    Raises:
        ValueError: If ``events`` is empty.
    """
    if not events:
        raise ValueError("compute_metrics requires at least one token event")

    first_ts = events[0].timestamp
    last_ts = events[-1].timestamp if end_ts is None else end_ts
    num_tokens = len(events)

    ttft = max(first_ts - start_ts, 0.0)
    gen_time = max(last_ts - first_ts, 0.0)
    tpot = gen_time / (num_tokens - 1) if num_tokens > 1 else 0.0
    throughput = num_tokens / gen_time if gen_time > 0 else 0.0
    total_latency = max(last_ts - start_ts, 0.0)

    return Metrics(
        ttft=ttft,
        tpot=tpot,
        throughput=throughput,
        total_latency=total_latency,
        num_tokens=num_tokens,
    )
