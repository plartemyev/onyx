"""
Unit tests for trigger_user_file_processing.

Deferred (UPLOADED) chat attachments start processing when a sent message
references them: the trigger flips them to PROCESSING and hands them to the
queue. Files in any other state are ignored so repeated sends and resends
are no-ops.
"""

from unittest.mock import MagicMock, patch
from uuid import uuid4

from onyx.configs.constants import (
    CELERY_USER_FILE_PROCESSING_TASK_EXPIRES,
    OnyxCeleryQueues,
    OnyxCeleryTask,
)
from onyx.db.enums import UserFileStatus
from onyx.db.models import UserFile
from onyx.db.projects import trigger_user_file_processing


def _make_user_file(status: UserFileStatus) -> MagicMock:
    uf = MagicMock(spec=UserFile)
    uf.id = uuid4()
    uf.status = status
    return uf


@patch("onyx.db.projects.DISABLE_VECTOR_DB", False)
@patch(
    "onyx.background.celery.versioned_apps.client.app",
    new_callable=MagicMock,
)
def test_trigger_flips_uploaded_and_enqueues(
    mock_client_app: MagicMock,
) -> None:
    uploaded = _make_user_file(UserFileStatus.UPLOADED)
    db_session = MagicMock()
    db_session.query.return_value.filter.return_value.all.return_value = [uploaded]

    triggered = trigger_user_file_processing(
        db_session=db_session,
        user_file_ids=[uploaded.id],
        tenant_id="test_tenant",
    )

    assert triggered == 1
    assert uploaded.status == UserFileStatus.PROCESSING
    db_session.commit.assert_called_once()

    assert mock_client_app.send_task.call_count == 1
    call = mock_client_app.send_task.call_args
    assert call.args[0] == OnyxCeleryTask.PROCESS_SINGLE_USER_FILE
    assert call.kwargs.get("queue") == OnyxCeleryQueues.USER_FILE_PROCESSING
    assert call.kwargs.get("expires") == CELERY_USER_FILE_PROCESSING_TASK_EXPIRES


@patch("onyx.db.projects.DISABLE_VECTOR_DB", False)
@patch(
    "onyx.background.celery.versioned_apps.client.app",
    new_callable=MagicMock,
)
def test_trigger_ignores_already_processed_files(
    mock_client_app: MagicMock,
) -> None:
    db_session = MagicMock()
    db_session.query.return_value.filter.return_value.all.return_value = []

    triggered = trigger_user_file_processing(
        db_session=db_session,
        user_file_ids=[uuid4()],
        tenant_id="test_tenant",
    )

    assert triggered == 0
    db_session.commit.assert_not_called()
    mock_client_app.send_task.assert_not_called()


@patch("onyx.db.projects.DISABLE_VECTOR_DB", False)
@patch(
    "onyx.background.celery.versioned_apps.client.app",
    new_callable=MagicMock,
)
def test_trigger_no_ids_is_noop(
    mock_client_app: MagicMock,
) -> None:
    triggered = trigger_user_file_processing(
        db_session=MagicMock(),
        user_file_ids=[],
        tenant_id="test_tenant",
    )

    assert triggered == 0
    mock_client_app.send_task.assert_not_called()
