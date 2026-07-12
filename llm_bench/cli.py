"""Command-line interface for llm-bench.

Usage examples::

    python -m llm_bench.cli run --backend fake --runs 5
    python -m llm_bench.cli run --backend fake --prompts prompts.txt --runs 10
    python -m llm_bench.cli run --backend fake --model gpt-4o --price-in 5 --price-out 15
    python -m llm_bench.cli run --backend http --model gpt-4o \
        --base-url https://api.openai.com/v1 --api-key $OPENAI_API_KEY
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from llm_bench.backends import FakeBackend, HttpBackend
from llm_bench.cost import Price
from llm_bench.harness import Runner
from llm_bench.report import render_json, render_text, write_report

logger = logging.getLogger("llm_bench.cli")


def _read_prompts(path: str) -> list[str]:
    """Read one prompt per non-empty line from ``path``."""
    prompts: list[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                prompts.append(line)
    return prompts


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="llm-bench",
        description="LLM benchmark & eval harness: latency, throughput, cost.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run a benchmark suite")
    run.add_argument(
        "--backend",
        choices=["fake", "http"],
        default="fake",
        help="Backend to use (default: fake)",
    )
    run.add_argument(
        "--prompts",
        default=None,
        help="Path to a prompts file (one prompt per non-empty line)",
    )
    run.add_argument(
        "--runs", type=int, default=5, help="Repetitions per prompt (default: 5)"
    )
    run.add_argument("--model", default=None, help="Model label / name")
    run.add_argument(
        "--price-in",
        type=float,
        default=None,
        help="Input price in USD per 1M tokens (for cost accounting)",
    )
    run.add_argument(
        "--price-out",
        type=float,
        default=None,
        help="Output price in USD per 1M tokens (for cost accounting)",
    )
    run.add_argument(
        "--format",
        choices=["text", "json", "csv"],
        default="text",
        help="Report output format (default: text)",
    )
    run.add_argument(
        "--output",
        default=None,
        help="Write report to this file (default: print to stdout)",
    )
    run.add_argument(
        "--base-url",
        default=None,
        help="[http backend] OpenAI-compatible base URL",
    )
    run.add_argument(
        "--api-key",
        default=None,
        help="[http backend] API key (falls back to OPENAI_API_KEY env)",
    )
    return parser


def _build_backend(args: argparse.Namespace):
    """Instantiate the selected backend from parsed args."""
    if args.backend == "fake":
        return FakeBackend(tokens=None, ttft=0.1, tpot=0.05)
    # http backend
    import os

    base_url = args.base_url or "https://api.openai.com/v1"
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise SystemExit(
            "http backend requires --api-key or OPENAI_API_KEY environment variable"
        )
    model = args.model or "gpt-4o"
    return HttpBackend(base_url=base_url, api_key=api_key, model=model)


def cmd_run(args: argparse.Namespace) -> int:
    """Execute the ``run`` subcommand."""
    prompts = _read_prompts(args.prompts) if args.prompts else None
    price = None
    if args.price_in is not None or args.price_out is not None:
        price = Price(
            input_per_1m=args.price_in or 0.0,
            output_per_1m=args.price_out or 0.0,
        )

    backend = _build_backend(args)
    runner = Runner(
        backend=backend,
        prompts=prompts,
        runs=args.runs,
        model=args.model,
        price=price,
    )
    logger.info(
        "Running benchmark backend=%s runs=%d prompts=%d",
        args.backend,
        args.runs,
        len(prompts) if prompts else 3,
    )
    agg, results = runner.run()

    if args.output:
        write_report(agg, results, args.output, fmt=args.format, model=args.model)
        print(f"Wrote {args.format} report to {args.output}")
    else:
        if args.format == "json":
            import json as _json

            print(_json.dumps(render_json(agg, results), indent=2))
        else:
            print(render_text(agg, model=args.model))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return cmd_run(args)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
