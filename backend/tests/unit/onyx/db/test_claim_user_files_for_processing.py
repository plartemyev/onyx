"""
Unit tests for claim_user_files_for_processing.

The beat safety net claims deferred (UPLOADED) chat attachments the send-path
trigger never claimed: it flips them to PROCESSING and returns every id that
needs a processing task (claimed UPLOADED + already-PROCESSING). Incognito
uploads are excluded from the claim — un-referenced incognito files belong to
the session sweep, not the indexing pipeline.
"""

from unittest.mock import MagicMock
from uuid import uuid4

from sqlalchemy.dialects import postgresql
from sqlalchemy.sql.expression import ClauseElement

from onyx.db.enums import UserFileStatus
from onyx.db.projects import claim_user_files_for_processing


def _compile(statement: ClauseElement) -> str:
    return str(statement.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}))


def test_claim_flips_uploaded_and_returns_full_enqueue_set() -> None:
    uploaded_id = uuid4()
    processing_id = uuid4()
    db_session = MagicMock()
    db_session.execute.return_value.all.return_value = [
        (uploaded_id, UserFileStatus.UPLOADED),
        (processing_id, UserFileStatus.PROCESSING),
    ]

    claimed = claim_user_files_for_processing(db_session)

    # Both ids are returned for enqueue; only the scan and one UPDATE ran.
    assert claimed == [uploaded_id, processing_id]
    assert db_session.execute.call_count == 2
    db_session.commit.assert_called_once()

    # The UPDATE only touches rows still UPLOADED (race guard vs the send
    # path): compile it and check the flip's SET and race guard.
    update_stmt = db_session.execute.call_args_list[1].args[0]
    update_sql = _compile(update_stmt)
    assert "UPDATE user_file" in update_sql
    assert "SET status='PROCESSING'" in update_sql
    assert "user_file.status = 'UPLOADED'" in update_sql


def test_claim_excludes_incognito_and_terminal_statuses() -> None:
    db_session = MagicMock()
    db_session.execute.return_value.all.return_value = []

    claim_user_files_for_processing(db_session)

    # Nothing claimable: no UPDATE, no commit.
    assert db_session.execute.call_count == 1
    db_session.commit.assert_not_called()

    select_sql = _compile(db_session.execute.call_args_list[0].args[0])
    # The scan covers both actionable statuses and never incognito uploads.
    assert "incognito" in select_sql
    assert UserFileStatus.UPLOADED.value in select_sql
    assert UserFileStatus.PROCESSING.value in select_sql
    assert UserFileStatus.COMPLETED.value not in select_sql
