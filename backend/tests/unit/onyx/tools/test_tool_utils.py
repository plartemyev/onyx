import pytest

from onyx.llm.constants import LlmProviderNames
from onyx.tools.models import (
    AnalyzeImageToolRichResponse,
    DownloadedFile,
    DownloadToolRichResponse,
    PythonExecutionFile,
    PythonToolRichResponse,
    ToolResponse,
)
from onyx.tools.utils import (
    explicit_tool_calling_supported,
    tool_response_generated_files,
)


@pytest.mark.parametrize(
    "model_provider, model_name, expected_result",
    [
        (LlmProviderNames.ANTHROPIC, "claude-4-sonnet-20250514", True),
        (
            "another-provider",
            "claude-haiku-4-5-20251001",
            True,
        ),
        (
            LlmProviderNames.ANTHROPIC,
            "claude-3-sonnet-20240229",
            False,
        ),
        (
            LlmProviderNames.BEDROCK,
            "amazon.titan-text-express-v1",
            False,
        ),
        (LlmProviderNames.OPENAI, "gpt-4o", True),
        (LlmProviderNames.OPENAI, "gpt-3.5-turbo-instruct", False),
    ],
)
def test_explicit_tool_calling_supported(
    model_provider: str,
    model_name: str,
    expected_result: bool,
) -> None:
    """
    Anthropic models support tool calling, but
    a) will raise an error if you provide any tool messages and don't provide a list of tools.
    b) will send text before and after generating tool calls.
    We don't want to provide that list of tools because our UI doesn't support sequential
    tool calling yet for (a) and just looks bad for (b), so for now we just treat anthropic
    models as non-tool-calling.

    Additionally, for Bedrock provider, any model containing an anthropic model name as a
    substring should also return False for the same reasons.
    """
    actual_result = explicit_tool_calling_supported(model_provider, model_name)
    assert actual_result == expected_result


def _downloaded_file(file_id: str, filename: str) -> DownloadedFile:
    return DownloadedFile(
        url=f"https://example.com/{filename}",
        file_id=file_id,
        file_url=f"https://example.com/api/chat/file/{file_id}",
        filename=filename,
        mime_type="image/png",
    )


def test_generated_files_from_python_tool_response() -> None:
    tool_response = ToolResponse(
        rich_response=PythonToolRichResponse(
            generated_files=[
                PythonExecutionFile(
                    filename="chart.png",
                    file_link="https://example.com/api/chat/file/abc",
                )
            ]
        ),
        llm_facing_response="",
    )
    files = tool_response_generated_files(tool_response)
    assert files is not None
    assert files[0].filename == "chart.png"
    assert files[0].file_link == "https://example.com/api/chat/file/abc"


def test_generated_files_from_download_tool_response() -> None:
    tool_response = ToolResponse(
        rich_response=DownloadToolRichResponse(
            files=[_downloaded_file("abc", "report.pdf")],
            failures=[],
        ),
        llm_facing_response="",
    )
    files = tool_response_generated_files(tool_response)
    assert files is not None
    assert files[0].filename == "report.pdf"
    assert files[0].file_link == "https://example.com/api/chat/file/abc"


def test_generated_files_from_analyze_image_tool_response() -> None:
    tool_response = ToolResponse(
        rich_response=AnalyzeImageToolRichResponse(
            files=[_downloaded_file("def", "photo.jpg")],
            failures=[],
        ),
        llm_facing_response="",
    )
    files = tool_response_generated_files(tool_response)
    assert files is not None
    assert files[0].filename == "photo.jpg"
    assert files[0].file_link == "https://example.com/api/chat/file/def"


@pytest.mark.parametrize(
    "rich_response",
    [
        None,
        "plain string response",
        PythonToolRichResponse(generated_files=[]),
    ],
)
def test_generated_files_returns_none_when_absent(
    rich_response: str | None | PythonToolRichResponse,
) -> None:
    tool_response = ToolResponse(
        rich_response=rich_response,
        llm_facing_response="",
    )
    assert tool_response_generated_files(tool_response) is None
