"""Find Qdrant points whose vector is a hash-based placeholder, not a provider embedding."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from automem.embedding.runtime_helpers import is_placeholder_vector

SCROLL_PAGE_SIZE = 256


def _point_vector(point: Any) -> Optional[Sequence[float]]:
    vector = getattr(point, "vector", None)
    if isinstance(vector, dict):
        # Named vectors: AutoMem writes a single unnamed vector, so only accept a
        # lone entry rather than guessing which named vector holds the embedding.
        vector = next(iter(vector.values())) if len(vector) == 1 else None
    if isinstance(vector, (list, tuple)) and vector:
        return vector
    return None


def find_placeholder_point_ids(qdrant_client: Any, collection_name: str) -> List[str]:
    """Return IDs of points whose stored vector is the placeholder for their content.

    Placeholder vectors are derived deterministically from the payload ``content``,
    so legacy points written before placeholders were refused are detected without
    any marker in the payload.

    Args:
        qdrant_client: Qdrant client exposing ``scroll``.
        collection_name: Collection to scan.

    Returns:
        Matching point IDs as strings, in scroll order.
    """
    matches: List[str] = []
    offset: Any = None
    while True:
        points, offset = qdrant_client.scroll(
            collection_name=collection_name,
            limit=SCROLL_PAGE_SIZE,
            offset=offset,
            with_payload=["content"],
            with_vectors=True,
        )
        for point in points or []:
            payload = getattr(point, "payload", None) or {}
            vector = _point_vector(point)
            content = payload.get("content")
            if vector is not None and isinstance(content, str):
                if is_placeholder_vector(vector, content):
                    matches.append(str(point.id))
        if offset is None:
            return matches
