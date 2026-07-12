"""Benchmark harness: run a prompt suite and aggregate metrics.

The :class:`Runner` drives a :class:`~llm_bench.backends.Backend` over a list of
prompts, repeated ``runs`` times, collecting one :class:`RunResult` per
request. Results are then aggregated into a :class:`Aggregate` with mean /
median / p95 statistics across every metric, plus cost roll-ups.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from llm_bench.backends import Backend, TokenEvent
from llm_bench.cost import Price, cost_for_request, cost_per_1m
from llm_bench.metrics import Metrics, compute_metrics

DEFAULT_PROMPTS = [
    "Explain the trade-off between latency and throughput in LLM serving.",
    "Summarise the concept of speculative decoding in one paragraph.",
    "List three factors that drive the cost of serving an LLM API.",
]


def count_tokens(text: str) -> int:
    """Approximate token count via whitespace splitting.

    This is intentionally dependency-free (no ``tiktoken``/network) so the
    harness works offline and deterministically. It is a rough proxy suitable
    for relative benchmarking; swap in a real tokenizer for exact accounting.
    """
    return len(text.split())


@dataclass
class RunResult:
    """Outcome of a single backend request."""

    prompt: str
    model: str | None
    metrics: Metrics
    cost: float
    cost_per_million: float
    input_tokens: int
    output_tokens: int


@dataclass
class Aggregate:
    """Aggregated statistics over all :class:`RunResult` objects."""

    runs: int = 0
    ttft_mean: float = 0.0
    ttft_median: float = 0.0
    ttft_p95: float = 0.0
    tpot_mean: float = 0.0
    tpot_median: float = 0.0
    tpot_p95: float = 0.0
    throughput_mean: float = 0.0
    throughput_median: float = 0.0
    throughput_p95: float = 0.0
    total_latency_mean: float = 0.0
    total_latency_median: float = 0.0
    total_latency_p95: float = 0.0
    cost_mean: float = 0.0
    cost_total: float = 0.0
    cost_per_1m: float = 0.0
    tokens_total: int = 0


def _percentile(values: list[float], quantile: float) -> float:
    """Nearest-rank percentile with ceil rounding (e.g. p95 of 10 = 10th value)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = min(len(ordered), max(1, int(quantile * len(ordered) + 0.999999)))
    return ordered[rank - 1]


def aggregate(results: Sequence[RunResult]) -> Aggregate:
    """Compute mean / median / p95 aggregates over run results."""
    agg = Aggregate(runs=len(results))
    if not results:
        return agg

    ttfts = [r.metrics.ttft for r in results]
    tpots = [r.metrics.tpot for r in results]
    throughputs = [r.metrics.throughput for r in results]
    latencies = [r.metrics.total_latency for r in results]
    costs = [r.cost for r in results]

    agg.ttft_mean = statistics.fmean(ttfts)
    agg.ttft_median = statistics.median(ttfts)
    agg.ttft_p95 = _percentile(ttfts, 0.95)

    agg.tpot_mean = statistics.fmean(tpots)
    agg.tpot_median = statistics.median(tpots)
    agg.tpot_p95 = _percentile(tpots, 0.95)

    agg.throughput_mean = statistics.fmean(throughputs)
    agg.throughput_median = statistics.median(throughputs)
    agg.throughput_p95 = _percentile(throughputs, 0.95)

    agg.total_latency_mean = statistics.fmean(latencies)
    agg.total_latency_median = statistics.median(latencies)
    agg.total_latency_p95 = _percentile(latencies, 0.95)

    agg.cost_mean = statistics.fmean(costs)
    agg.cost_total = sum(costs)
    # Normalised cost across all tokens seen in the suite.
    total_in = sum(r.input_tokens for r in results)
    total_out = sum(r.output_tokens for r in results)
    total_tokens = total_in + total_out
    agg.cost_per_1m = (
        (agg.cost_total * 1_000_000.0 / total_tokens) if total_tokens else 0.0
    )
    agg.tokens_total = total_tokens
    return agg


class Runner:
    """Runs a benchmark suite against a backend and aggregates results.

    Args:
        backend: The backend to benchmark.
        prompts: Prompt suite to run.
        runs: Repetitions per prompt.
        model: Optional model label passed to the backend.
        price: Optional per-model pricing used for cost accounting.
        clock: Monotonic clock (seconds); injectable for tests.
    """

    def __init__(
        self,
        backend: Backend,
        prompts: Sequence[str] | None = None,
        runs: int = 1,
        model: str | None = None,
        price: Price | None = None,
        clock=time.monotonic,
    ) -> None:
        self.backend = backend
        self.prompts = list(prompts) if prompts is not None else list(DEFAULT_PROMPTS)
        self.runs = max(1, int(runs))
        self.model = model
        self.price = price
        self._clock = clock

    def _run_once(self, prompt: str) -> RunResult:
        events: list[TokenEvent] = []
        start_ts = self._clock()
        for event in self.backend.stream(prompt, model=self.model):
            events.append(event)
        # The request ends when the final token arrives; deriving end time from
        # the last event keeps synthetic (FakeBackend) and real (HttpBackend)
        # timing consistent without a second wall-clock sample.
        metrics = compute_metrics(events, start_ts)

        input_tokens = count_tokens(prompt)
        output_tokens = metrics.num_tokens
        if self.price is not None:
            cost = cost_for_request(self.price, input_tokens, output_tokens)
            try:
                per_1m = cost_per_1m(self.price, input_tokens, output_tokens)
            except ValueError:
                per_1m = 0.0
        else:
            cost = 0.0
            per_1m = 0.0

        return RunResult(
            prompt=prompt,
            model=self.model,
            metrics=metrics,
            cost=cost,
            cost_per_million=per_1m,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def run(self) -> tuple[Aggregate, list[RunResult]]:
        """Execute the full suite and return ``(aggregate, per_run_results)``."""
        results: list[RunResult] = []
        for prompt in self.prompts:
            for _ in range(self.runs):
                results.append(self._run_once(prompt))
        return aggregate(results), results
