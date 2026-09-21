import json
from collections.abc import Callable
from typing import Any

from sqlalchemy.orm import Session

from onyx.configs.app_configs import AZURE_IMAGE_API_KEY
from onyx.db.connector import check_connectors_exist
from onyx.db.document import check_docs_exist
from onyx.db.models import LLMProvider
from onyx.llm.constants import LlmProviderNames
from onyx.llm.model_capabilities import find_model_obj, get_model_map
from onyx.tools.interface import Tool
from onyx.tools.models import (
    AnalyzeImageToolRichResponse,
    DownloadToolRichResponse,
    PythonExecutionFile,
    PythonToolRichResponse,
    ToolResponse,
)


def explicit_tool_calling_supported(model_provider: str, model_name: str) -> bool:
    model_map = get_model_map()
    model_obj = find_model_obj(
        model_map=model_map,
        provider=model_provider,
        model_name=model_name,
    )

    if not model_obj:
        return False
    return bool(model_obj.get("supports_function_calling"))


def compute_tool_tokens(tool: Tool, token_counter: Callable[[str], int]) -> int:
    return token_counter(json.dumps(tool.tool_definition()))


def compute_all_tool_tokens(
    tools: list[Tool], token_counter: Callable[[str], int]
) -> int:
    return sum(compute_tool_tokens(tool, token_counter) for tool in tools)


def compute_tool_definition_tokens(
    tool_definitions: list[dict[str, Any]], token_counter: Callable[[str], int]
) -> int:
    return sum(
        token_counter(json.dumps(tool_definition))
        for tool_definition in tool_definitions
    )


def is_image_generation_available(db_session: Session) -> bool:
    providers = db_session.query(LLMProvider).all()
    for provider in providers:
        if provider.provider == LlmProviderNames.OPENAI:
            return True

    return bool(AZURE_IMAGE_API_KEY)


def is_document_search_available(db_session: Session) -> bool:
    docs_exist = check_docs_exist(db_session)
    connectors_exist = check_connectors_exist(db_session)
    return docs_exist or connectors_exist


def tool_response_generated_files(
    tool_response: ToolResponse,
) -> list[PythonExecutionFile] | None:
    """Files a tool produced for the user, for persistence on ToolCallInfo.

    Code interpreter files come back as PythonExecutionFile already; the
    download_file and analyze_image tools return DownloadedFile and are
    converted here. Returns None when the response carries no files."""
    rich_response = tool_response.rich_response
    if isinstance(rich_response, PythonToolRichResponse):
        return rich_response.generated_files or None
    if isinstance(rich_response, DownloadToolRichResponse):
        return [
            PythonExecutionFile(
                filename=downloaded_file.filename,
                file_link=downloaded_file.file_url,
            )
            for downloaded_file in rich_response.files
        ]
    if isinstance(rich_response, AnalyzeImageToolRichResponse):
        return [
            PythonExecutionFile(
                filename=analyzed_file.filename,
                file_link=analyzed_file.file_url,
            )
            for analyzed_file in rich_response.files
        ]
    return None


def generate_tools_description(tools: list[Tool]) -> str:
    if not tools:
        return ""
    if len(tools) == 1:
        return tools[0].name
    if len(tools) == 2:
        return f"{tools[0].name} and {tools[1].name}"

    names = [tool.name for tool in tools[:-1]]
    return ", ".join(names) + f", and {tools[-1].name}"
