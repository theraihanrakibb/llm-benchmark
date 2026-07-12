"""Tests for HttpBackend using an httpx MockTransport (no real network)."""

from __future__ import annotations

import json

import httpx

from llm_bench.backends import HttpBackend, TokenEvent
from llm_bench.metrics import compute_metrics


def _sse_handler(tokens: list[str]):
    """Build an httpx MockTransport that streams SSE token deltas."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/chat/completions"
        body_lines = []
        for tok in tokens:
            payload = {
                "choices": [{"delta": {"content": tok}, "index": 0, "finish_reason": None}]
            }
            body_lines.append(f"data: {json.dumps(payload)}\n")
        body_lines.append("data: [DONE]\n")
        # httpx streaming responses read line-by-line; split into lines.
        raw = "".join(body_lines).encode()
        return httpx.Response(200, text=raw.decode())

    return handler


def test_http_backend_parses_sse():
    tokens = ["Hello", " world", "!"]
    transport = httpx.MockTransport(_sse_handler(tokens))
    client = httpx.Client(transport=transport, base_url="https://example.test")
    backend = HttpBackend(
        base_url="https://example.test",
        api_key="test-key",
        model="gpt-4o",
        client=client,
        clock=lambda: 0.0,  # all tokens at t=0 -> TPOT 0
    )
    events = list(backend.stream("hi"))
    assert [e.token for e in events] == tokens
    assert all(isinstance(e, TokenEvent) for e in events)


def test_http_backend_metrics_from_sse():
    tokens = ["a", "b", "c"]
    transport = httpx.MockTransport(_sse_handler(tokens))
    client = httpx.Client(transport=transport, base_url="https://example.test")
    clock_values = iter([1.0, 1.5, 2.0])

    def clock():
        return next(clock_values)

    backend = HttpBackend(
        base_url="https://example.test",
        api_key="test-key",
        model="gpt-4o",
        client=client,
        clock=clock,
    )
    events = list(backend.stream("hi"))
    m = compute_metrics(events, start_ts=0.0)
    assert m.ttft == 1.0
    assert m.num_tokens == 3


def test_http_backend_requires_httpx(monkeypatch):
    """If httpx import is unavailable, HttpBackend must refuse to init."""
    import llm_bench.backends as backends

    monkeypatch.setattr(backends, "httpx", None)
    try:
        HttpBackend(base_url="x", api_key="y", model="m")
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
