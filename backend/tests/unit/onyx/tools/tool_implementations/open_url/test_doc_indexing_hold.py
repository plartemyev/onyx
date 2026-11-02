"""Unit tests for the open_url indexing hold (doc_indexing)."""

from unittest.mock import MagicMock
from uuid import uuid4

from onyx.db.enums import UserFileStatus
from onyx.tools.tool_implementations.open_url.doc_indexing import (
    _OPEN_URL_INDEX_POLL_SECONDS,
    _wait_until_searchable,
)


def _session_returning(seq_statuses: list[dict]) -> MagicMock:
    """A mock db_session whose status queries return queued mappings, one per
    poll."""
    session = MagicMock()
    queue = list(seq_statuses)

    def _query(*_args: object) -> MagicMock:
        result = MagicMock()
        result.filter.return_value.all.return_value = queue.pop(0).items()
        return result

    session.query.side_effect = _query
    return session


def test_wait_returns_once_all_completed() -> None:
    ids = [uuid4(), uuid4()]
    completed = {uid: UserFileStatus.COMPLETED for uid in ids}
    session = _session_returning([completed])

    _wait_until_searchable(session, ids, budget_seconds=30)

    session.query.assert_called_once()


def test_wait_polls_until_terminal() -> None:
    id_a, id_b = uuid4(), uuid4()
    session = _session_returning(
        [
            {id_a: UserFileStatus.PROCESSING, id_b: UserFileStatus.INDEXING},
            {id_a: UserFileStatus.COMPLETED, id_b: UserFileStatus.COMPLETED},
        ]
    )

    _wait_until_searchable(session, [id_a, id_b], budget_seconds=60)

    assert session.query.call_count == 2


def test_wait_zero_budget_skips_polling() -> None:
    session = _session_returning([])
    ids = [uuid4()]

    _wait_until_searchable(session, ids, budget_seconds=0)

    session.query.assert_not_called()


def test_wait_bounded_by_budget(monkeypatch) -> None:
    """A file stuck in PROCESSING cannot hold longer than the budget."""
    now = {"t": 0.0}
    sleeps: list[float] = []

    def fake_sleep(s: float) -> None:
        sleeps.append(s)
        now["t"] += s

    monkeypatch.setattr(
        "onyx.tools.tool_implementations.open_url.doc_indexing.time.sleep",
        fake_sleep,
    )
    monkeypatch.setattr(
        "onyx.tools.tool_implementations.open_url.doc_indexing.time.monotonic",
        lambda: now["t"],
    )
    stuck = {uuid4(): UserFileStatus.PROCESSING}
    session = _session_returning([stuck] * 100)

    _wait_until_searchable(session, list(stuck), budget_seconds=5)

    # Each sleep is capped by the remaining budget, and the loop stops at the
    # deadline: two full 2s polls plus one capped 1s poll for a 5s budget.
    assert sleeps == [2.0, 2.0, 1.0]
