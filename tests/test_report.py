"""Tests for report rendering (text/JSON/CSV shapes)."""

from __future__ import annotations

from llm_bench.backends import FakeBackend
from llm_bench.cost import Price
from llm_bench.harness import Runner
from llm_bench.report import render_csv, render_json, render_text


def _run() -> tuple:
    backend = FakeBackend(tokens=["a", "b"], ttft=0.1, tpot=0.05)
    price = Price(input_per_1m=5.0, output_per_1m=15.0)
    runner = Runner(backend=backend, prompts=["p"], runs=4, price=price)
    return runner.run()


def test_render_text_contains_sections():
    agg, _ = _run()
    text = render_text(agg, model="fake")
    assert "llm-bench report" in text
    assert "TTFT" in text
    assert "cost ($/1M tok)" in text
    assert "model=fake" in text


def test_render_json_shape():
    agg, results = _run()
    doc = render_json(agg, results)
    assert set(doc.keys()) == {"aggregate", "runs"}
    assert doc["aggregate"]["runs"] == 4
    assert len(doc["runs"]) == 4
    run0 = doc["runs"][0]
    assert set(run0) == {
        "prompt",
        "model",
        "metrics",
        "cost",
        "cost_per_million",
        "input_tokens",
        "output_tokens",
    }
    assert set(run0["metrics"]) == {
        "ttft",
        "tpot",
        "throughput",
        "total_latency",
        "num_tokens",
    }
    # Cost accounting applied.
    assert run0["cost"] > 0.0


def test_render_csv_shape():
    agg, _ = _run()
    csv_text = render_csv(agg)
    lines = csv_text.strip().splitlines()
    assert lines[0].split(",")[0] == "runs"
    assert len(lines) == 2
    assert lines[1].split(",")[0] == "4"
