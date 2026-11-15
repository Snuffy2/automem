"""Voyage rate-limit handling: Retry-After, shared cooldown, and wait budgets.

All timing goes through a fake clock and sleep; no real network or sleeping.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional
from unittest.mock import Mock, patch

import httpx
import pytest

from automem.embedding.voyage import (
    VoyageEmbeddingProvider,
    VoyageRateLimiter,
    VoyageRateLimitError,
    parse_retry_after,
)

DIM = 1024
URL = "https://api.voyageai.com/v1/embeddings"
# 2026-01-01T00:00:00Z
WALL_NOW = 1767225600.0


class FakeClock:
    """Monotonic clock whose sleep advances time and records each call."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: List[float] = []
        self._lock = threading.Lock()

    def monotonic(self) -> float:
        return self.now

    def wall(self) -> float:
        return WALL_NOW + (self.now - 1000.0)

    def sleep(self, seconds: float) -> None:
        with self._lock:
            self.sleeps.append(seconds)
            self.now += seconds


def make_limiter(clock: FakeClock, rpm: float = 0.0) -> VoyageRateLimiter:
    return VoyageRateLimiter(
        rpm,
        clock=clock.monotonic,
        wall_clock=clock.wall,
        sleep=clock.sleep,
        rng=lambda: 0.0,
    )


def status_response(status: int, headers: Optional[Dict[str, str]] = None) -> Mock:
    request = httpx.Request("POST", URL)
    error = httpx.HTTPStatusError(
        f"HTTP {status}",
        request=request,
        response=httpx.Response(status, request=request, headers=headers or {}),
    )
    response = Mock()
    response.raise_for_status.side_effect = error
    return response


def ok_response() -> Mock:
    response = Mock()
    response.raise_for_status.return_value = None
    response.status_code = 200
    response.json.return_value = {"data": [{"embedding": [0.1] * DIM}]}
    return response


def make_provider(limiter: VoyageRateLimiter, responses: List[Mock], **kwargs):
    with patch("automem.embedding.voyage.httpx.Client") as client_cls:
        client = Mock()
        client.post.side_effect = responses
        client_cls.return_value = client
        provider = VoyageEmbeddingProvider(
            api_key="voyage-test-key", dimension=DIM, rate_limiter=limiter, **kwargs
        )
    return provider, client


# ---------------------------------------------------------------- parsing


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("7", 7.0),
        (" 2.5 ", 2.5),
        ("0", 0.0),
        ("Thu, 01 Jan 2026 00:00:30 GMT", 30.0),
        ("Wed, 31 Dec 2025 23:59:00 GMT", 0.0),  # past date clamps to 0
        (None, None),
        ("", None),
        ("soon", None),
        ("-3", None),
        ("nan", None),
        ("inf", None),
    ],
)
def test_parse_retry_after(header, expected):
    assert parse_retry_after(header, now=WALL_NOW) == expected


# ---------------------------------------------------------------- provider


def test_success_path_makes_one_request_without_waiting():
    clock = FakeClock()
    provider, client = make_provider(make_limiter(clock), [ok_response()])

    assert len(provider.generate_embedding("hello")) == DIM
    assert client.post.call_count == 1
    assert clock.sleeps == []


def test_429_honors_retry_after_seconds():
    clock = FakeClock()
    provider, client = make_provider(
        make_limiter(clock), [status_response(429, {"Retry-After": "4"}), ok_response()]
    )

    assert len(provider.generate_embedding("hello")) == DIM
    assert client.post.call_count == 2
    assert clock.sleeps == [4.0]


def test_429_honors_retry_after_http_date():
    clock = FakeClock()
    provider, _ = make_provider(
        make_limiter(clock),
        [status_response(429, {"Retry-After": "Thu, 01 Jan 2026 00:00:06 GMT"}), ok_response()],
    )

    provider.generate_embedding("hello")
    assert clock.sleeps == [6.0]


def test_retry_after_is_capped_by_max_delay():
    clock = FakeClock()
    provider, _ = make_provider(
        make_limiter(clock),
        [status_response(429, {"Retry-After": "120"}), ok_response()],
        max_retry_delay=5.0,
        max_retry_wait=60.0,
    )

    provider.generate_embedding("hello")
    assert clock.sleeps == [5.0]


@pytest.mark.parametrize("headers", [{}, {"Retry-After": "garbage"}])
def test_missing_or_invalid_retry_after_uses_exponential_backoff(headers):
    clock = FakeClock()
    provider, client = make_provider(
        make_limiter(clock),
        [status_response(503, headers), status_response(503, headers), ok_response()],
        max_retries=2,
    )

    provider.generate_embedding("hello")
    # rng=0 gives the lower jitter bound: half of 1s, then half of 2s.
    assert clock.sleeps == [0.5, 1.0]
    assert client.post.call_count == 3


def test_backoff_jitter_stays_within_step():
    clock = FakeClock()
    limiter = VoyageRateLimiter(
        clock=clock.monotonic, wall_clock=clock.wall, sleep=clock.sleep, rng=lambda: 0.999
    )
    provider, _ = make_provider(limiter, [status_response(500), ok_response()])

    provider.generate_embedding("hello")
    assert 0.5 <= clock.sleeps[0] < 1.0


def test_query_embedding_fails_fast_when_retry_after_exceeds_budget():
    clock = FakeClock()
    provider, client = make_provider(
        make_limiter(clock),
        [status_response(429, {"Retry-After": "10"}), ok_response()],
        query_max_retry_wait=3.0,
        max_retry_wait=60.0,
    )

    with pytest.raises(VoyageRateLimitError) as excinfo:
        provider.generate_query_embedding("what did I decide?")

    assert isinstance(excinfo.value.__cause__, httpx.HTTPStatusError)
    assert client.post.call_count == 1
    assert clock.sleeps == []


def test_total_wait_budget_bounds_repeated_backoff():
    clock = FakeClock()
    provider, client = make_provider(
        make_limiter(clock),
        [status_response(503, {"Retry-After": "2"})] * 5 + [ok_response()],
        max_retries=5,
        max_retry_wait=5.0,
    )

    with pytest.raises(VoyageRateLimitError):
        provider.generate_embedding("hello")

    assert sum(clock.sleeps) <= 5.0
    assert client.post.call_count == 3


def test_non_retryable_status_is_raised_immediately():
    clock = FakeClock()
    provider, client = make_provider(make_limiter(clock), [status_response(400), ok_response()])

    with pytest.raises(httpx.HTTPStatusError):
        provider.generate_embedding("hello")
    assert client.post.call_count == 1
    assert clock.sleeps == []


def test_retries_exhausted_raises_last_status_error():
    clock = FakeClock()
    provider, client = make_provider(
        make_limiter(clock),
        [status_response(429, {"Retry-After": "1"})] * 2,
        max_retries=1,
    )

    with pytest.raises(httpx.HTTPStatusError):
        provider.generate_embedding("hello")
    assert client.post.call_count == 2
    # The final 429 still holds other callers back.
    assert limiter_cooldown(provider) == pytest.approx(1.0)


def limiter_cooldown(provider: VoyageEmbeddingProvider) -> float:
    return provider.rate_limiter.cooldown_remaining()


# ---------------------------------------------------------------- sharing


def test_cooldown_from_one_provider_delays_another():
    clock = FakeClock()
    limiter = make_limiter(clock)
    first, _ = make_provider(limiter, [status_response(429, {"Retry-After": "8"}), ok_response()])
    second, second_client = make_provider(limiter, [ok_response()])

    # The first caller's 429 records the cooldown; stop it before it retries.
    first.max_retry_wait = 0.0
    with pytest.raises(VoyageRateLimitError):
        first.generate_embedding("a")
    assert limiter.cooldown_remaining() == pytest.approx(8.0)

    second.generate_embedding("b")
    assert clock.sleeps == [8.0]
    assert second_client.post.call_count == 1


def test_cooldown_makes_query_caller_fail_fast_without_request():
    clock = FakeClock()
    limiter = make_limiter(clock)
    limiter.note_rate_limited(30.0)
    provider, client = make_provider(limiter, [ok_response()], query_max_retry_wait=2.0)

    with pytest.raises(VoyageRateLimitError):
        provider.generate_query_embedding("q")
    assert client.post.call_count == 0


def test_cooldown_is_shared_across_threads():
    clock = FakeClock()
    limiter = make_limiter(clock)
    limiter.note_rate_limited(5.0)
    released: List[float] = []
    errors: List[BaseException] = []

    def worker() -> None:
        try:
            limiter.acquire(deadline=clock.monotonic() + 100.0)
            released.append(clock.monotonic())
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert not errors
    assert len(released) == 4
    # Nobody proceeded before the cooldown ended.
    assert min(released) >= 1005.0


def test_requests_per_minute_paces_calls():
    clock = FakeClock()
    limiter = make_limiter(clock, rpm=30)  # one request every 2 seconds

    for _ in range(3):
        limiter.acquire(deadline=clock.monotonic() + 100.0)

    assert clock.sleeps == [2.0, 2.0]


def test_requests_per_minute_respects_deadline():
    clock = FakeClock()
    limiter = make_limiter(clock, rpm=6)  # one request every 10 seconds
    limiter.acquire(deadline=clock.monotonic() + 1.0)

    with pytest.raises(VoyageRateLimitError):
        limiter.acquire(deadline=clock.monotonic() + 1.0)
    assert clock.sleeps == []


def test_cooldown_wait_adds_bounded_jitter():
    clock = FakeClock()
    limiter = VoyageRateLimiter(
        clock=clock.monotonic, wall_clock=clock.wall, sleep=clock.sleep, rng=lambda: 0.5
    )
    limiter.note_rate_limited(2.0)

    limiter.acquire(deadline=clock.monotonic() + 10.0)
    # 2s cooldown + 0.5 * min(1.0, 2.0 * 0.25)
    assert clock.sleeps == [pytest.approx(2.25)]
