"""Pluggable backend abstraction for llm-bench.

A :class:`Backend` turns a prompt into a stream of :class:`TokenEvent` objects,
each carrying the emitted token and the wall-clock timestamp at which it
arrived. All latency/throughput math downstream is derived purely from those
timestamps, so backends are trivially swappable:

* :class:`FakeBackend` synthesises deterministic tokens with a configurable
  first-token delay (TTFT) and per-token delay (TPOT). It is the default in
  tests and demos and never touches the network.
* :class:`HttpBackend` speaks the OpenAI-compatible streaming API over ``httpx``
  and parses Server-Sent Events. It is fully implemented but, unlike
  ``FakeBackend``, only makes sense against a live (or mocked) endpoint.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

try:  # httpx is only needed by HttpBackend; keep import lazy/optional.
    import httpx
except Exception:  # pragma: no cover - httpx is declared in requirements
    httpx = None  # type: ignore[assignment]


@dataclass(frozen=True)
class TokenEvent:
    """A single emitted token with its arrival timestamp.

    ``timestamp`` is a monotonic clock value (seconds) as produced by the
    backend's injected clock callable.
    """

    token: str
    timestamp: float


DEFAULT_FAKE_TOKENS = [
    "The",
    " quick",
    " brown",
    " fox",
    " jumps",
    " over",
    " the",
    " lazy",
    " dog",
    ".",
]


class Backend(ABC):
    """Abstract interface every benchmark backend must implement."""

    @abstractmethod
    def stream(
        self, prompt: str, model: str | None = None, **kwargs: object
    ) -> Iterator[TokenEvent]:
        """Yield :class:`TokenEvent` objects for ``prompt``.

        Args:
            prompt: The user prompt to complete.
            model: Optional model override; backends may ignore it.
            kwargs: Backend-specific options (ignored by the base contract).

        Yields:
            TokenEvent: One event per emitted token, in order.
        """
        raise NotImplementedError


class FakeBackend(Backend):
    """Deterministic, network-free backend used in tests and demos.

    The backend does not actually sleep. Instead it reconstructs the token
    timestamps from a configurable TTFT and TPOT using the injected ``clock``,
    which makes every run fully reproducible and fast.

    Args:
        tokens: The tokens to emit. Defaults to a fixed canned sentence so that
            runs are byte-for-byte reproducible.
        ttft: Simulated time-to-first-token in seconds.
        tpot: Simulated time-per-output-token in seconds.
        clock: Monotonic clock returning seconds. Injectable for tests.
    """

    def __init__(
        self,
        tokens: list[str] | None = None,
        ttft: float = 0.1,
        tpot: float = 0.05,
        clock=time.monotonic,
    ) -> None:
        self._tokens = list(tokens) if tokens is not None else list(DEFAULT_FAKE_TOKENS)
        self._ttft = float(ttft)
        self._tpot = float(tpot)
        self._clock = clock
        self.call_count = 0

    def stream(
        self, prompt: str, model: str | None = None, **kwargs: object
    ) -> Iterator[TokenEvent]:
        """Yield deterministic tokens with reconstructed timestamps."""
        if not self._tokens:
            return
        self.call_count += 1
        start = self._clock()
        first_ts = start + self._ttft
        yield TokenEvent(token=self._tokens[0], timestamp=first_ts)
        prev = first_ts
        for tok in self._tokens[1:]:
            prev = prev + self._tpot
            yield TokenEvent(token=tok, timestamp=prev)


class HttpBackend(Backend):
    """OpenAI-compatible streaming backend built on ``httpx``.

    Args:
        base_url: Base URL of the OpenAI-compatible API, e.g.
            ``https://api.openai.com/v1``.
        api_key: Bearer token for the ``Authorization`` header.
        model: Default model name used when a call does not override it.
        client: Optional pre-configured ``httpx.Client`` (used for tests with a
            ``MockTransport``). When omitted a client is created lazily.
        clock: Monotonic clock returning seconds; injectable for tests.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        client: "httpx.Client | None" = None,
        clock=time.monotonic,
    ) -> None:
        if httpx is None:
            raise RuntimeError("httpx is required for HttpBackend")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self._client = client
        self._clock = clock

    def _get_client(self) -> "httpx.Client":
        if self._client is not None:
            return self._client
        return httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=60.0,
        )

    def stream(
        self, prompt: str, model: str | None = None, **kwargs: object
    ) -> Iterator[TokenEvent]:
        """Stream tokens from an OpenAI-compatible ``/chat/completions`` endpoint."""
        model = model or self.model
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }
        client = self._get_client()
        with client.stream(
            "POST", "/chat/completions", json=payload
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if content:
                    yield TokenEvent(token=content, timestamp=self._clock())
