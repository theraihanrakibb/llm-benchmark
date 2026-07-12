"""llm-bench: an LLM benchmark & eval harness.

Public API surface for the ``llm_bench`` package.
"""

from llm_bench.backends import Backend, FakeBackend, HttpBackend, TokenEvent
from llm_bench.cost import Price, cost_for_request, cost_per_1m
from llm_bench.metrics import Metrics, compute_metrics
from llm_bench.harness import RunResult, Aggregate, Runner

__version__ = "0.1.0"

__all__ = [
    "Backend",
    "FakeBackend",
    "HttpBackend",
    "TokenEvent",
    "Price",
    "cost_for_request",
    "cost_per_1m",
    "Metrics",
    "compute_metrics",
    "RunResult",
    "Aggregate",
    "Runner",
]
