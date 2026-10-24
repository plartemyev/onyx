"""
Unit test verifying that the upload API path sends tasks with expires=.

The upload_files_to_user_files function must include expires= on every
send_task call to prevent phantom task accumulation if the worker is down or
slow. Only files already in PROCESSING (project uploads) are enqueued at
upload time; deferred chat attachments (UPLOADED) are enqueued when a sent
message references them.
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
from onyx.db.projects import upload_files_to_user_files


def _make_mock_user_file(status: UserFileStatus) -> MagicMock:
    uf = MagicMock(spec=UserFile)
    uf.id = str(uuid4())
    uf.status = status
    return uf


@patch("onyx.db.projects.get_current_tenant_id", return_value="test_tenant")
@patch("onyx.db.projects.create_user_files")
@patch(
    "onyx.background.celery.versioned_apps.client.app",
    new_callable=MagicMock,
)
def test_send_task_includes_expires(
    mock_client_app: MagicMock,
    mock_create: MagicMock,
    mock_tenant: MagicMock,  # noqa: ARG001
) -> None:
    """Every send_task call from the upload path must include expires=."""
    user_files = [
        _make_mock_user_file(UserFileStatus.PROCESSING),
        _make_mock_user_file(UserFileStatus.PROCESSING),
    ]
    mock_create.return_value = MagicMock(
        user_files=user_files,
        rejected_files=[],
        id_to_temp_id={},
        skip_indexing_filenames=set(),
        indexable_files=user_files,
    )

    mock_user = MagicMock()
    mock_db_session = MagicMock()

    upload_files_to_user_files(
        files=[],
        project_id=1,
        user=mock_user,
        temp_id_map=None,
        db_session=mock_db_session,
    )

    assert mock_client_app.send_task.call_count == len(user_files)

    for call in mock_client_app.send_task.call_args_list:
        assert call.args[0] == OnyxCeleryTask.PROCESS_SINGLE_USER_FILE
        assert call.kwargs.get("queue") == OnyxCeleryQueues.USER_FILE_PROCESSING
        assert call.kwargs.get("expires") == CELERY_USER_FILE_PROCESSING_TASK_EXPIRES, (
            "send_task must include expires= to prevent phantom task accumulation"
        )


@patch("onyx.db.projects.get_current_tenant_id", return_value="test_tenant")
@patch("onyx.db.projects.create_user_files")
@patch(
    "onyx.background.celery.versioned_apps.client.app",
    new_callable=MagicMock,
)
def test_deferred_uploads_are_not_enqueued(
    mock_client_app: MagicMock,
    mock_create: MagicMock,
    mock_tenant: MagicMock,  # noqa: ARG001
) -> None:
    """Chat attachments (UPLOADED) must not start processing at upload time."""
    user_files = [
        _make_mock_user_file(UserFileStatus.UPLOADED),
        _make_mock_user_file(UserFileStatus.SKIPPED),
    ]
    mock_create.return_value = MagicMock(
        user_files=user_files,
        rejected_files=[],
        id_to_temp_id={},
        skip_indexing_filenames=set(),
        indexable_files=user_files,
    )

    upload_files_to_user_files(
        files=[],
        project_id=None,
        user=MagicMock(),
        temp_id_map=None,
        db_session=MagicMock(),
    )

    mock_client_app.send_task.assert_not_called()
