from __future__ import annotations

import hashlib
import math
import random
from typing import Any, Callable, List, Optional, Sequence


def normalize_tags(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(tag, str) for tag in value):
        return value
    raise ValueError("'tags' must be a list of strings or a single string")


def coerce_importance(value: Any) -> float:
    if value is None:
        return 0.5
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("'importance' must be a number") from exc
    if score < 0 or score > 1:
        raise ValueError("'importance' must be between 0 and 1")
    return score


def coerce_embedding(value: Any, expected_dim: int) -> Optional[List[float]]:
    if value is None or value == "":
        return None
    vector: List[Any]
    if isinstance(value, list):
        vector = value
    elif isinstance(value, str):
        vector = [part.strip() for part in value.split(",") if part.strip()]
    else:
        raise ValueError("Embedding must be a list of floats or a comma-separated string")

    if len(vector) != expected_dim:
        raise ValueError(f"Embedding must contain exactly {expected_dim} values")

    try:
        return [float(component) for component in vector]
    except ValueError as exc:
        raise ValueError("Embedding must contain numeric values") from exc


def generate_placeholder_embedding(content: str, expected_dim: int) -> List[float]:
    """Generate a deterministic embedding vector from the content."""
    digest = hashlib.sha256(content.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)
    rng = random.Random(seed)
    return [rng.random() for _ in range(expected_dim)]


def is_placeholder_vector(vector: Sequence[float], content: str) -> bool:
    """Return whether ``vector`` is the deterministic placeholder for ``content``.

    Placeholder vectors are seeded from the content hash, so a stored point can be
    identified without extra payload. Comparison uses cosine similarity because a
    cosine-distance collection may store normalized vectors.

    Args:
        vector: Stored vector to inspect.
        content: Text the vector was written for.

    Returns:
        True when the vector matches the placeholder for that text.
    """
    if not vector or not isinstance(content, str):
        return False
    expected = generate_placeholder_embedding(content, len(vector))
    try:
        dot = sum(float(a) * b for a, b in zip(vector, expected))
        norm = math.sqrt(sum(float(a) * float(a) for a in vector))
    except (TypeError, ValueError):
        return False
    expected_norm = math.sqrt(sum(b * b for b in expected))
    if norm == 0 or expected_norm == 0:
        return False
    return dot / (norm * expected_norm) >= 1 - 1e-6


def generate_real_embedding(
    content: str,
    *,
    init_embedding_provider: Callable[[], None],
    state: Any,
    logger: Any,
    placeholder_embedding: Callable[[str], List[float]],
    query: bool = False,
) -> List[float]:
    """Generate an embedding using the configured provider (query=True for search queries)."""
    init_embedding_provider()

    if state.embedding_provider is None:
        logger.warning("No embedding provider available, using placeholder")
        return placeholder_embedding(content)

    expected_dim = state.effective_vector_size
    try:
        provider = state.embedding_provider
        if query and hasattr(provider, "generate_query_embedding"):
            embedding = provider.generate_query_embedding(content)
        else:
            embedding = provider.generate_embedding(content)
        if not isinstance(embedding, list) or len(embedding) != expected_dim:
            logger.warning(
                "Provider %s returned %s dims (expected %d); falling back to placeholder",
                state.embedding_provider.provider_name(),
                len(embedding) if isinstance(embedding, list) else "invalid",
                expected_dim,
            )
            return placeholder_embedding(content)
        return embedding
    except Exception as exc:
        logger.warning("Failed to generate embedding: %s", str(exc))
        return placeholder_embedding(content)


def generate_real_embeddings_batch(
    contents: List[str],
    *,
    init_embedding_provider: Callable[[], None],
    state: Any,
    logger: Any,
    placeholder_embedding: Callable[[str], List[float]],
    allow_placeholder_fallback: bool = True,
    allow_placeholder_provider: bool = False,
    allow_partial: bool = False,
) -> List[Optional[List[float]]]:
    """Generate multiple embeddings in a single batch for efficiency.

    Args:
        contents: Texts to embed.
        init_embedding_provider: Lazily initializes ``state.embedding_provider``.
        state: Service state holding the provider and effective vector size.
        logger: Logger for fallback diagnostics.
        placeholder_embedding: Deterministic placeholder generator.
        allow_placeholder_fallback: Substitute placeholder vectors when the provider
            is missing or fails. When False, failures raise instead.
        allow_placeholder_provider: In strict mode, accept a deliberately configured
            placeholder provider.
        allow_partial: In strict mode, return None for items whose per-item provider
            call failed instead of raising, so callers can keep the successful
            vectors and queue the rest for retry.

    Returns:
        One vector per input; entries are None only when ``allow_partial`` is set.
    """
    init_embedding_provider()

    if not contents:
        return []

    if state.embedding_provider is None:
        if not allow_placeholder_fallback:
            raise RuntimeError("No embedding provider available")
        logger.debug("No embedding provider available, falling back to placeholder embeddings")
        return [placeholder_embedding(c) for c in contents]

    provider = state.embedding_provider
    try:
        provider_name = provider.provider_name()
    except Exception:
        provider_name = provider.__class__.__name__ or "unknown"
    if (
        not allow_placeholder_fallback
        and not allow_placeholder_provider
        and str(provider_name).strip().lower() == "placeholder"
    ):
        raise RuntimeError("Placeholder embedding provider is not allowed in strict mode")
    expected_dim = state.effective_vector_size
    try:
        embeddings = provider.generate_embeddings_batch(contents)
        if (
            not isinstance(embeddings, list)
            or len(embeddings) != len(contents)
            or any(not isinstance(e, list) or len(e) != expected_dim for e in embeddings)
        ):
            raise ValueError(
                f"invalid batch result: expected {len(contents)} embeddings "
                f"of {expected_dim} dims"
            )
        return embeddings
    except Exception as exc:
        logger.warning(
            "Batch embedding via provider %s failed (%s); "
            "falling back to per-item embedding calls",
            provider_name,
            exc,
        )

    # Per-item fallback: single-input calls may succeed even when the batch
    # endpoint fails (e.g. Voyage hanging on multi-input requests).
    results: List[Optional[List[float]]] = []
    placeholder_count = 0
    missing_count = 0
    for content in contents:
        try:
            embedding = provider.generate_embedding(content)
            if not isinstance(embedding, list) or len(embedding) != expected_dim:
                raise ValueError(
                    f"expected {expected_dim} dims, got "
                    f"{len(embedding) if isinstance(embedding, list) else 'invalid'}"
                )
            results.append(embedding)
        except Exception as exc:
            if not allow_placeholder_fallback and allow_partial:
                logger.debug("Per-item embedding failed (%s); leaving item unembedded", exc)
                results.append(None)
                missing_count += 1
                continue
            if not allow_placeholder_fallback:
                raise RuntimeError(
                    f"Failed to generate provider embedding for item "
                    f"{len(results) + 1}/{len(contents)}"
                ) from exc
            logger.debug("Per-item embedding failed (%s); using placeholder", exc)
            results.append(placeholder_embedding(content))
            placeholder_count += 1

    if placeholder_count:
        logger.warning(
            "%d/%d items fell back to placeholder embeddings — these will be "
            "invisible to semantic search until re-embedded",
            placeholder_count,
            len(contents),
        )
    if missing_count:
        logger.warning(
            "%d/%d items could not be embedded by provider %s; callers must retry them",
            missing_count,
            len(contents),
            provider_name,
        )
    return results
