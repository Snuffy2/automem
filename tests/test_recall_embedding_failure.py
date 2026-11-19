"""Recall must not run a vector search with a placeholder query vector.

When the configured embedding provider fails to embed the recall query (for
example a Voyage 429), the old path fell back to a hash-based placeholder
vector and searched Qdrant with it, mixing arbitrary "vector" matches into the
results. Recall now skips the vector leg, keeps the keyword leg, and reports
``vector_search.skipped_reason``.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

import app
import automem.search.runtime_recall_helpers as recall_helpers
from automem.embedding.placeholder import PlaceholderEmbeddingProvider
from automem.embedding.runtime_helpers import EmbeddingUnavailableError, generate_real_embedding
from tests.support.fake_graph import FakeGraph, FakeNode, FakeResult

DIM = 8
KEYWORD_MEMORY = {
    "id": "keyword-hit",
    "content": "voyage rate limit runbook",
    "tags": ["ops"],
    "importance": 0.7,
    "confidence": 0.9,
    "type": "Context",
    "enriched": True,
    "timestamp": "2026-05-18T00:00:00+00:00",
}


class _KeywordGraph(FakeGraph):
    """FakeGraph that answers the recall keyword query with one memory."""

    def query(self, query: str, params: Dict[str, Any] | None = None, **kwargs: Any) -> Any:
        if "kw_score" in query:
            return FakeResult([[FakeNode(dict(KEYWORD_MEMORY)), 3]])
        return super().query(query, params, **kwargs)


class _RecordingQdrant:
    def __init__(self) -> None:
        self.calls: List[List[float]] = []

    def search(self, *, collection_name, query_vector, limit, with_payload=True, query_filter=None):
        self.calls.append(list(query_vector))
        payload = {**KEYWORD_MEMORY, "id": "vector-hit", "content": "unrelated vector memory"}
        return [SimpleNamespace(id="vector-hit", payload=payload, score=0.9)]


class _FailingProvider:
    def provider_name(self) -> str:
        return "voyage:voyage-4"

    def generate_embedding(self, text: str) -> List[float]:
        raise RuntimeError("429 Too Many Requests")


class _WorkingProvider:
    def provider_name(self) -> str:
        return "voyage:voyage-4"

    def generate_embedding(self, text: str) -> List[float]:
        return [0.5] * DIM


@pytest.fixture
def recall_env(monkeypatch):
    state = app.ServiceState()
    state.memory_graph = _KeywordGraph()
    state.effective_vector_size = DIM
    qdrant = _RecordingQdrant()
    state.qdrant = qdrant

    monkeypatch.setattr(app, "state", state)
    monkeypatch.setattr(app, "init_falkordb", lambda: None)
    monkeypatch.setattr(app, "init_qdrant", lambda: None)
    monkeypatch.setattr(app, "API_TOKEN", "test-token")
    monkeypatch.setattr(app, "ADMIN_TOKEN", "test-admin-token")
    # Use the real app embedding binding so strict mode reaches the provider;
    # pin helper globals that sibling tests may reconfigure.
    monkeypatch.setattr(recall_helpers, "_generate_real_embedding", app._generate_real_embedding)
    monkeypatch.setattr(recall_helpers, "_fetch_relations", lambda *_a, **_k: [])
    monkeypatch.setattr(recall_helpers, "_logger", logging.getLogger("automem.test.recall"))
    return state, qdrant


def _recall(extra: str = "") -> Dict[str, Any]:
    resp = app.app.test_client().get(
        f"/recall?query=voyage%20rate%20limit&limit=5&current_only=false{extra}",
        headers={"Authorization": "Bearer test-token"},
    )
    assert resp.status_code == 200, resp.get_data(as_text=True)
    return resp.get_json()


def _ids(body: Dict[str, Any]) -> List[str]:
    return [r.get("id") or (r.get("memory") or {}).get("id") for r in body["results"]]


def test_query_embedding_failure_skips_vector_search(recall_env):
    state, qdrant = recall_env
    state.embedding_provider = _FailingProvider()

    body = _recall()

    assert qdrant.calls == []
    assert _ids(body) == ["keyword-hit"]
    assert body["vector_search"] == {
        "enabled": True,
        "matched": False,
        "skipped_reason": "embedding_failed",
    }


def test_query_embedding_failure_skips_scope_fallback_search(recall_env):
    state, qdrant = recall_env
    state.embedding_provider = _FailingProvider()

    body = _recall("&tags=ops&scope_fallback=true")

    assert qdrant.calls == []
    assert body["vector_search"]["skipped_reason"] == "embedding_failed"


def test_query_embedding_success_runs_vector_search(recall_env):
    state, qdrant = recall_env
    state.embedding_provider = _WorkingProvider()

    body = _recall()

    assert qdrant.calls == [[0.5] * DIM]
    assert "vector-hit" in _ids(body)
    assert body["vector_search"] == {"enabled": True, "matched": True}


def test_configured_placeholder_provider_still_searches(recall_env):
    state, qdrant = recall_env
    provider = PlaceholderEmbeddingProvider(dimension=DIM)
    state.embedding_provider = provider

    body = _recall()

    assert qdrant.calls == [provider.generate_embedding("voyage rate limit")]
    assert "skipped_reason" not in body["vector_search"]


def _embed(provider: Any, *, strict: bool) -> List[float]:
    return generate_real_embedding(
        "text",
        init_embedding_provider=lambda: None,
        state=SimpleNamespace(embedding_provider=provider, effective_vector_size=DIM),
        logger=logging.getLogger("automem.test.embedding"),
        placeholder_embedding=lambda _c: [0.0] * DIM,
        strict=strict,
    )


def test_non_strict_embedding_failure_still_returns_placeholder():
    assert _embed(_FailingProvider(), strict=False) == [0.0] * DIM


@pytest.mark.parametrize(
    "provider",
    [
        _FailingProvider(),
        SimpleNamespace(provider_name=lambda: "voyage", generate_embedding=lambda _t: [1.0]),
    ],
    ids=["provider_error", "wrong_dimension"],
)
def test_strict_embedding_failure_raises(provider):
    with pytest.raises(EmbeddingUnavailableError):
        _embed(provider, strict=True)
