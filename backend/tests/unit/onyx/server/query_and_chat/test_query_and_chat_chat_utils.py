from onyx.file_store.models import ChatFileType
from onyx.server.query_and_chat.chat_utils import mime_type_to_chat_file_type


def test_mime_type_to_chat_file_type_strips_parameters() -> None:
    assert (
        mime_type_to_chat_file_type("text/csv; charset=utf-8") == ChatFileType.TABULAR
    )
    assert (
        mime_type_to_chat_file_type("application/pdf; charset=binary")
        == ChatFileType.DOC
    )
    assert (
        mime_type_to_chat_file_type("image/png; name=image.png") == ChatFileType.IMAGE
    )


def test_mime_type_to_chat_file_type_normalizes_case_and_whitespace() -> None:
    assert (
        mime_type_to_chat_file_type(" Text/CSV ; charset=utf-8") == ChatFileType.TABULAR
    )


def test_mime_type_to_chat_file_type_maps_audio_and_video() -> None:
    assert mime_type_to_chat_file_type("audio/x-wav") == ChatFileType.AUDIO
    assert mime_type_to_chat_file_type("audio/mpeg") == ChatFileType.AUDIO
    assert mime_type_to_chat_file_type("Audio/MP4; codecs=mp4a") == ChatFileType.AUDIO
    assert mime_type_to_chat_file_type("video/mp4") == ChatFileType.VIDEO
    assert mime_type_to_chat_file_type("video/quicktime") == ChatFileType.VIDEO

    # Non-media types keep their existing classification.
    assert mime_type_to_chat_file_type("application/octet-stream") == (
        ChatFileType.PLAIN_TEXT
    )
