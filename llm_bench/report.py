"""Report rendering for llm-bench.

A benchmark produces an :class:`~llm_bench.harness.Aggregate` plus the raw
:class:`~llm_bench.harness.RunResult` list. This module renders them as a human
readable text table, a machine readable JSON document, or a flat CSV summary.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import asdict

from llm_bench.harness import Aggregate, RunResult

_ROWS = [
    ("TTFT (s)", "ttft_mean", "ttft_median", "ttft_p95"),
    ("TPOT (s)", "tpot_mean", "tpot_median", "tpot_p95"),
    ("Throughput (tok/s)", "throughput_mean", "throughput_median", "throughput_p95"),
    ("Total latency (s)", "total_latency_mean", "total_latency_median", "total_latency_p95"),
]


def render_text(agg: Aggregate, model: str | None = None) -> str:
    """Render the aggregate as a fixed-width plain-text table."""
    lines: list[str] = []
    title = "llm-bench report"
    if model:
        title += f"  model={model}"
    lines.append(title)
    lines.append("=" * 52)
    header = f"{'metric':<22}{'mean':>11}{'median':>11}{'p95':>10}"
    lines.append(header)
    lines.append("-" * 52)
    for label, mean_k, median_k, p95_k in _ROWS:
        lines.append(
            f"{label:<22}{getattr(agg, mean_k):>11.4f}"
            f"{getattr(agg, median_k):>11.4f}{getattr(agg, p95_k):>10.4f}"
        )
    lines.append("-" * 52)
    lines.append(f"{'runs':<22}{agg.runs:>11}")
    lines.append(f"{'tokens total':<22}{agg.tokens_total:>11}")
    lines.append(f"{'cost mean ($)':<22}{agg.cost_mean:>11.6f}")
    lines.append(f"{'cost total ($)':<22}{agg.cost_total:>11.6f}")
    lines.append(f"{'cost ($/1M tok)':<22}{agg.cost_per_1m:>11.6f}")
    return "\n".join(lines)


def render_json(agg: Aggregate, results: Sequence[RunResult]) -> dict:
    """Render the full report as a JSON-serialisable dict."""
    return {
        "aggregate": asdict(agg),
        "runs": [asdict(r) for r in results],
    }


def render_csv(agg: Aggregate) -> str:
    """Render the aggregate summary as CSV (one row of values)."""
    import io

    fields = [
        "runs",
        "ttft_mean",
        "ttft_median",
        "ttft_p95",
        "tpot_mean",
        "tpot_median",
        "tpot_p95",
        "throughput_mean",
        "throughput_median",
        "throughput_p95",
        "total_latency_mean",
        "total_latency_median",
        "total_latency_p95",
        "cost_mean",
        "cost_total",
        "cost_per_1m",
        "tokens_total",
    ]

    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()
    writer.writerow({k: getattr(agg, k) for k in fields})
    return out.getvalue()


def write_report(
    agg: Aggregate,
    results: Sequence[RunResult],
    path: str,
    fmt: str = "text",
    model: str | None = None,
) -> None:
    """Write a report to ``path`` in the chosen format.

    Args:
        agg: Aggregated metrics.
        results: Per-run results (used by the JSON renderer).
        path: Destination file path.
        fmt: One of ``text``, ``json``, ``csv``.
        model: Optional model label for the text header.
    """
    fmt = fmt.lower()
    if fmt == "json":
        content = json.dumps(render_json(agg, results), indent=2)
    elif fmt == "csv":
        content = render_csv(agg)
    else:
        content = render_text(agg, model=model)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
        if not content.endswith("\n"):
            fh.write("\n")
