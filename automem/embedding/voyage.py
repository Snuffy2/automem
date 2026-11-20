"""Voyage AI embedding provider using Voyage API."""

import email.utils
import logging
import math
import os
import random
import threading
import time
from datetime import timezone
from typing import Callable, List, Optional

import httpx

from automem.config import (
    VOYAGE_MAX_RETRIES,
    VOYAGE_MAX_RETRY_WAIT_SECONDS,
    VOYAGE_QUERY_MAX_RETRY_WAIT_SECONDS,
    VOYAGE_REQUESTS_PER_MINUTE,
    VOYAGE_RETRY_MAX_DELAY_SECONDS,
)
from automem.embedding.provider import EmbeddingProvider

logger = logging.getLogger(__name__)

VOYAGE_API_URL = "https://api.voyageai.com/v1/embeddings"

# First exponential-backoff step when no Retry-After header is available.
_BACKOFF_BASE_SECONDS = 1.0
# Fraction of a cooldown added as random jitter (capped at one second) so
# callers released together do not all hit Voyage in the same instant.
_COOLDOWN_JITTER_FRACTION = 0.25


class VoyageRateLimitError(RuntimeError):
    """Raised when a Voyage call cannot proceed within its wait budget."""


def parse_retry_after(value: Optional[str], now: Optional[float] = None) -> Optional[float]:
    """Parse an HTTP ``Retry-After`` header into a delay in seconds.

    Args:
        value: Raw header value: delay seconds or an HTTP-date.
        now: Current wall-clock time as a Unix timestamp; defaults to ``time.time()``.

    Returns:
        Non-negative delay in seconds, or None when the header is absent or invalid.
    """
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        seconds = float(text)
    except ValueError:
        seconds = None
    if seconds is not None:
        return seconds if math.isfinite(seconds) and seconds >= 0 else None
    try:
        when = email.utils.parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    current = time.time() if now is None else now
    return max(0.0, when.timestamp() - current)


class VoyageRateLimiter:
    """Thread-safe, process-wide gate shared by every Voyage request.

    After a 429 the limiter holds a cooldown that every caller waits out before
    its next request. An optional requests-per-minute cap spaces requests evenly.
    Clock, sleep and random sources are injectable for tests.
    """

    def __init__(
        self,
        requests_per_minute: float = 0.0,
        *,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        rng: Callable[[], float] = random.random,
    ) -> None:
        """Create a limiter.

        Args:
            requests_per_minute: Pacing cap; 0 or less disables pacing.
            clock: Monotonic clock used for cooldowns and deadlines.
            wall_clock: Unix-time clock used to resolve HTTP-date headers.
            sleep: Sleep function.
            rng: Source of uniform random numbers in [0, 1).
        """
        self._interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
        self.clock = clock
        self.wall_clock = wall_clock
        self.rng = rng
        self._sleep = sleep
        self._lock = threading.Lock()
        self._cooldown_until = 0.0
        self._next_slot = 0.0

    def cooldown_remaining(self) -> float:
        """Return the seconds left in the shared cooldown (0 when none)."""
        with self._lock:
            return max(0.0, self._cooldown_until - self.clock())

    def note_rate_limited(self, delay: float) -> None:
        """Start or extend the shared cooldown to at least ``delay`` seconds from now."""
        with self._lock:
            self._cooldown_until = max(self._cooldown_until, self.clock() + max(0.0, delay))

    def acquire(self, deadline: float) -> None:
        """Wait for the cooldown and pacing slot, then claim the slot.

        Args:
            deadline: Monotonic time after which the caller gives up.

        Raises:
            VoyageRateLimitError: If the wait would run past ``deadline``.
        """
        while True:
            with self._lock:
                now = self.clock()
                ready_at = max(self._cooldown_until, self._next_slot)
                if ready_at <= now:
                    if self._interval:
                        self._next_slot = now + self._interval
                    return
                if ready_at > deadline:
                    raise VoyageRateLimitError(
                        f"Voyage rate limit: next request allowed in {ready_at - now:.2f}s, "
                        f"beyond the remaining wait budget of {max(0.0, deadline - now):.2f}s"
                    )
                wait = ready_at - now
                if ready_at == self._cooldown_until:
                    wait += self.rng() * min(1.0, wait * _COOLDOWN_JITTER_FRACTION)
                wait = min(wait, deadline - now)
            self._sleep(wait)

    def sleep_until_deadline(self, delay: float, deadline: float) -> None:
        """Sleep ``delay`` seconds for a caller-local backoff.

        Raises:
            VoyageRateLimitError: If the sleep would run past ``deadline``.
        """
        if self.clock() + delay > deadline:
            raise VoyageRateLimitError(
                f"Voyage retry backoff of {delay:.2f}s exceeds the remaining wait budget"
            )
        self._sleep(delay)


_shared_rate_limiter = VoyageRateLimiter(VOYAGE_REQUESTS_PER_MINUTE)


def get_shared_rate_limiter() -> VoyageRateLimiter:
    """Return the limiter shared by all Voyage providers in this process."""
    return _shared_rate_limiter


class VoyageEmbeddingProvider(EmbeddingProvider):
    """Generates embeddings using Voyage AI's embedding API.

    Requires a Voyage API key and makes network requests to Voyage AI.
    Provides high-quality semantic embeddings with generous free tier.

    Supports the Voyage 4 family with shared embedding spaces:
    - voyage-4-large: Best quality, MoE architecture
    - voyage-4: Approaches voyage-3-large quality, mid-sized
    - voyage-4-lite: Optimized for latency/cost
    """

    SUPPORTED_DIMENSIONS = {256, 512, 1024, 2048}

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "voyage-4",
        dimension: int = 1024,
        timeout: float = 30.0,
        max_retries: Optional[int] = None,
        input_type: Optional[str] = None,
        query_input_type: Optional[str] = None,
        *,
        max_retry_wait: Optional[float] = None,
        query_max_retry_wait: Optional[float] = None,
        max_retry_delay: Optional[float] = None,
        rate_limiter: Optional[VoyageRateLimiter] = None,
    ) -> None:
        """Initialize Voyage embedding provider.

        Args:
            api_key: Voyage API key (falls back to VOYAGE_API_KEY env var)
            model: Voyage embedding model to use (voyage-4, voyage-4-large, voyage-4-lite)
            dimension: Number of dimensions for embeddings (256, 512, 1024, 2048)
            timeout: Request timeout in seconds (default 30)
            max_retries: Retries after the first request for 429/5xx and network
                errors (default VOYAGE_MAX_RETRIES)
            input_type: Optional input type hint ("query" or "document")
            query_input_type: Optional input type for search queries; when set,
                generate_query_embedding sends it instead of input_type
            max_retry_wait: Total seconds one call may wait on cooldowns, pacing
                and backoff (default VOYAGE_MAX_RETRY_WAIT_SECONDS)
            query_max_retry_wait: Wait budget for recall query embeddings
                (default VOYAGE_QUERY_MAX_RETRY_WAIT_SECONDS)
            max_retry_delay: Cap on any single Retry-After or backoff delay
                (default VOYAGE_RETRY_MAX_DELAY_SECONDS)
            rate_limiter: Limiter to coordinate with; defaults to the
                process-wide shared limiter

        Raises:
            ValueError: If API key is not provided or dimension is invalid
        """
        api_key_value = api_key or os.getenv("VOYAGE_API_KEY")
        if not api_key_value:
            raise ValueError("Voyage API key required (pass api_key or set VOYAGE_API_KEY)")

        if dimension not in self.SUPPORTED_DIMENSIONS:
            raise ValueError(
                f"Invalid dimension {dimension}. Must be one of: {self.SUPPORTED_DIMENSIONS}"
            )

        self.model = model
        self._dimension = dimension
        self.timeout = timeout
        self.max_retries = VOYAGE_MAX_RETRIES if max_retries is None else max(0, max_retries)
        self.max_retry_wait = (
            VOYAGE_MAX_RETRY_WAIT_SECONDS if max_retry_wait is None else max_retry_wait
        )
        self.query_max_retry_wait = (
            VOYAGE_QUERY_MAX_RETRY_WAIT_SECONDS
            if query_max_retry_wait is None
            else query_max_retry_wait
        )
        self.max_retry_delay = (
            VOYAGE_RETRY_MAX_DELAY_SECONDS if max_retry_delay is None else max_retry_delay
        )
        self.rate_limiter = rate_limiter or get_shared_rate_limiter()
        self.input_type = input_type
        self.query_input_type = query_input_type

        self.client = httpx.Client(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {api_key_value}",
                "Content-Type": "application/json",
            },
        )

        logger.info(
            "Voyage embedding provider initialized (model=%s, dimensions=%d, timeout=%.1fs, retries=%d)",
            model,
            dimension,
            timeout,
            self.max_retries,
        )

    def _retry_delay(self, response: httpx.Response, attempt: int) -> float:
        """Return the wait before the next attempt: Retry-After or jittered backoff."""
        limiter = self.rate_limiter
        delay = parse_retry_after(response.headers.get("Retry-After"), limiter.wall_clock())
        if delay is None:
            step = _BACKOFF_BASE_SECONDS * (2**attempt)
            delay = step / 2 + limiter.rng() * step / 2
        return min(delay, self.max_retry_delay)

    def _make_request(
        self,
        texts: List[str],
        input_type: Optional[str] = None,
        max_wait: Optional[float] = None,
    ) -> List[List[float]]:
        """Make embedding request to Voyage API.

        Every attempt first waits for the shared rate limiter. A 429 starts a
        process-wide cooldown (Retry-After when present, otherwise jittered
        exponential backoff); a 5xx backs off locally. Waiting is bounded by
        ``max_wait`` so callers fail fast instead of blocking.

        Args:
            texts: List of texts to embed
            input_type: Optional input type ("query" or "document")
            max_wait: Total wait budget in seconds; defaults to max_retry_wait

        Returns:
            List of embedding vectors

        Raises:
            VoyageRateLimitError: If waiting would exceed the budget
            Exception: If API call fails after retries
        """
        limiter = self.rate_limiter
        budget = self.max_retry_wait if max_wait is None else max_wait
        deadline = limiter.clock() + max(0.0, budget)

        payload = {
            "input": texts,
            "model": self.model,
            "output_dimension": self._dimension,
        }

        # Use instance input_type if not overridden
        effective_input_type = input_type or self.input_type
        if effective_input_type:
            payload["input_type"] = effective_input_type

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                limiter.acquire(deadline)
            except VoyageRateLimitError as exc:
                logger.warning("%s; giving up (model=%s)", exc, self.model)
                if last_error is not None:
                    raise exc from last_error
                raise
            try:
                response = self.client.post(VOYAGE_API_URL, json=payload)
                response.raise_for_status()
                data = response.json()

                # Defensive validation of API response structure
                if not isinstance(data, dict):
                    raise ValueError(f"Voyage API response is not an object (model={self.model})")
                if "error" in data:
                    raise ValueError(
                        f"Voyage API returned error payload (model={self.model}, status={response.status_code}): "
                        f"{data['error']}"
                    )
                if "data" not in data:
                    raise ValueError("Voyage API response missing 'data' field")
                if not isinstance(data["data"], list):
                    raise ValueError("Voyage API response 'data' field is not a list")

                embeddings = []
                for i, item in enumerate(data["data"]):
                    if not isinstance(item, dict):
                        raise ValueError(f"Voyage API response item {i} is not an object")
                    if "embedding" not in item:
                        raise ValueError(
                            f"Voyage API response item {i} " f"missing 'embedding' field"
                        )

                    embedding = item["embedding"]
                    if not isinstance(embedding, list):
                        raise ValueError(
                            f"Voyage API response item {i} 'embedding' field is not a list"
                        )
                    if any(not isinstance(v, (int, float)) for v in embedding):
                        raise ValueError(
                            f"Voyage API response item {i} 'embedding' contains non-numeric values"
                        )
                    embeddings.append(embedding)

                # Validate dimensions
                for i, emb in enumerate(embeddings):
                    if len(emb) != self._dimension:
                        raise ValueError(
                            f"Voyage embedding length {len(emb)} "
                            f"!= configured dimension "
                            f"{self._dimension} "
                            f"at index {i} (model={self.model})"
                        )

                # Validate count matches input
                if len(embeddings) != len(texts):
                    raise ValueError(
                        f"Voyage returned {len(embeddings)} "
                        f"embeddings for {len(texts)} input texts"
                    )

                return embeddings

            except httpx.HTTPStatusError as e:
                last_error = e
                status = e.response.status_code
                if status == 429 or 500 <= status < 600:
                    error_type = "rate limited" if status == 429 else "server error"
                    delay = self._retry_delay(e.response, attempt)
                    if status == 429:
                        # Shared: every Voyage caller waits out this cooldown in
                        # acquire() instead of retrying immediately.
                        limiter.note_rate_limited(delay)
                    if attempt >= self.max_retries:
                        logger.warning(
                            "Voyage %s (status %d), attempt %d/%d; no retries left",
                            error_type,
                            status,
                            attempt + 1,
                            self.max_retries + 1,
                        )
                        raise
                    logger.warning(
                        "Voyage %s (status %d), attempt %d/%d; retrying in %.2fs",
                        error_type,
                        status,
                        attempt + 1,
                        self.max_retries + 1,
                        delay,
                    )
                    if status != 429:
                        try:
                            limiter.sleep_until_deadline(delay, deadline)
                        except VoyageRateLimitError as exc:
                            logger.warning("%s; giving up (model=%s)", exc, self.model)
                            raise exc from e
                    continue
                raise
            except ValueError:
                raise
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    logger.warning(
                        "Voyage request failed, attempt %d/%d: %s",
                        attempt + 1,
                        self.max_retries + 1,
                        e,
                    )
                    continue
                raise

        raise last_error or RuntimeError("Voyage request failed after retries")

    def generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding using Voyage API.

        Args:
            text: The text to embed

        Returns:
            Embedding vector from Voyage

        Raises:
            Exception: If API call fails
        """
        embeddings = self._make_request([text])
        logger.debug("Generated Voyage embedding for text (length: %d)", len(text))
        return embeddings[0]

    def generate_query_embedding(self, text: str) -> List[float]:
        """Generate a search-query embedding on the request path.

        Sends query_input_type when configured and uses the shorter
        query_max_retry_wait budget so a rate-limited recall fails fast.
        """
        return self._make_request(
            [text], input_type=self.query_input_type, max_wait=self.query_max_retry_wait
        )[0]

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts in one API call.

        Args:
            texts: List of texts to embed (max 128 per batch for Voyage)

        Returns:
            List of embedding vectors from Voyage

        Raises:
            Exception: If API call fails
        """
        if not texts:
            return []

        # Voyage supports up to 128 texts per batch
        batch_size = 128
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = self._make_request(batch)
            all_embeddings.extend(embeddings)

        logger.info(
            "Generated %d Voyage embeddings in batch (avg length: %d)",
            len(all_embeddings),
            sum(len(t) for t in texts) // len(texts) if texts else 0,
        )
        return all_embeddings

    def dimension(self) -> int:
        """Return embedding dimensionality.

        Returns:
            The number of dimensions in the embedding vectors
        """
        return self._dimension

    def provider_name(self) -> str:
        """Return provider name.

        Returns:
            Provider identifier
        """
        return f"voyage:{self.model}"

    def close(self) -> None:
        """Close the HTTP client."""
        if hasattr(self, "client"):
            self.client.close()

    def __enter__(self) -> "VoyageEmbeddingProvider":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        """Clean up HTTP client."""
        self.close()
