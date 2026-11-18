"""Behavior tests for ``run_consolidation_tick`` persistence and event emission."""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from automem.consolidation.runtime_scheduler import run_consolidation_tick


def _forget_result(archived: int = 0, deleted: int = 0) -> Dict[str, Any]:
    """Build a result shaped like ``MemoryConsolidator.apply_controlled_forgetting`` output."""
    return {
        "mode": "forget",
        "success": True,
        "steps": {
            "forget": {
                "examined": archived + deleted + 3,
                "archived": [
                    {"id": f"a{i}", "content_preview": "old", "relevance": 0.1, "type": "Memory"}
                    for i in range(archived)
                ],
                "deleted": [
                    {"id": f"d{i}", "content_preview": "older", "relevance": 0.0, "type": "Memory"}
                    for i in range(deleted)
                ],
                "preserved": 3,
                "protected": [],
            }
        },
    }


def _decay_result(processed: int) -> Dict[str, Any]:
    """Build a result shaped like ``MemoryConsolidator._apply_decay`` output."""
    return {
        "mode": "decay",
        "success": True,
        "steps": {
            "decay": {
                "processed": processed,
                "avg_relevance_before": 0.5,
                "avg_relevance_after": 0.4,
                "distribution": {"high": 0, "medium": processed, "low": 0, "archive": 0},
            }
        },
    }


class FakeScheduler:
    """Scheduler double returning canned task results."""

    def __init__(self, results: List[Dict[str, Any]]) -> None:
        self._results = results

    def run_scheduled_tasks(self, decay_threshold: float | None = None) -> List[Dict[str, Any]]:
        return self._results

    def get_next_runs(self) -> Dict[str, str]:
        return {"forget": "In 6 day(s)", "decay": "In 23 hour(s)"}


class TickHarness:
    """Run one tick against fakes and record what was persisted and emitted."""

    def __init__(self, results: List[Dict[str, Any]]) -> None:
        self.scheduler = FakeScheduler(results)
        self.persisted: List[Dict[str, Any]] = []
        self.events: List[Dict[str, Any]] = []
        self.logger = MagicMock()

    def run(self) -> None:
        run_consolidation_tick(
            get_memory_graph_fn=lambda: object(),
            build_scheduler_from_graph_fn=lambda graph: self.scheduler,
            persist_consolidation_run_fn=lambda graph, result: self.persisted.append(result),
            decay_importance_threshold=None,
            emit_event_fn=lambda event_type, data, utc_now_fn: self.events.append(
                {"type": event_type, **data}
            ),
            utc_now_fn=lambda: "2026-11-17T00:00:00+00:00",
            perf_counter_fn=lambda: 0.0,
            logger=self.logger,
        )

    def events_by_task(self) -> Dict[str, Dict[str, Any]]:
        return {event["task_type"]: event for event in self.events}


def test_forget_result_with_lists_is_persisted_and_emitted() -> None:
    result = _forget_result(archived=2, deleted=1)
    harness = TickHarness([result])

    harness.run()

    harness.logger.exception.assert_not_called()
    assert harness.persisted == [result]
    assert len(harness.events) == 1
    event = harness.events[0]
    assert event["type"] == "consolidation.run"
    assert event["task_type"] == "forget"
    assert event["affected_count"] == 3
    assert event["success"] is True
    assert event["next_scheduled"] == "In 6 day(s)"
    assert event["steps"] == ["forget"]


def test_forget_run_that_archives_nothing_reports_zero_affected() -> None:
    harness = TickHarness([_forget_result(archived=0, deleted=0)])

    harness.run()

    harness.logger.exception.assert_not_called()
    assert [event["affected_count"] for event in harness.events] == [0]


def test_every_result_in_a_tick_is_persisted_and_emitted_when_forget_is_first() -> None:
    forget = _forget_result(archived=70)
    decay = _decay_result(processed=488)
    creative = {"mode": "creative", "success": True, "steps": {"creative": {"created": 4}}}
    results = [forget, decay, creative]
    harness = TickHarness(results)

    harness.run()

    harness.logger.exception.assert_not_called()
    assert harness.persisted == results
    events = harness.events_by_task()
    assert set(events) == {"forget", "decay", "creative"}
    assert events["forget"]["affected_count"] == 70
    assert events["decay"]["affected_count"] == 488
    assert events["creative"]["affected_count"] == 4


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (_decay_result(processed=12), 12),
        ({"mode": "creative", "steps": {"creative": {"discovered": 9, "created": 3}}}, 3),
        (
            {
                "mode": "cluster",
                "steps": {"cluster": {"clusters_found": 5, "meta_memories_created": 2}},
            },
            2,
        ),
        (
            {
                "mode": "identity",
                "steps": {
                    "identity": {
                        "identities_synthesized": 2,
                        "merges_performed": [{"canonical": "a", "alias": "b"}],
                        "merge_candidates_for_review": [{"canonical": "c", "alias": "d"}],
                    }
                },
            },
            3,
        ),
        ({"mode": "identity", "steps": {"identity": {"skipped": True, "reason": "off"}}}, 0),
    ],
    ids=["decay", "creative", "cluster", "identity", "identity-skipped"],
)
def test_affected_count_reflects_what_each_step_reports(
    result: Dict[str, Any], expected: int
) -> None:
    harness = TickHarness([result])

    harness.run()

    harness.logger.exception.assert_not_called()
    assert [event["affected_count"] for event in harness.events] == [expected]
