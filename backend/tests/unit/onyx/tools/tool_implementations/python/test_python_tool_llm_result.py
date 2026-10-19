"""Unit tests for the LLM-facing python execution result serialization.

The generated-file entries carry a ready-to-paste `markdown` snippet next to
`file_link`, so the model can embed the file without assembling a link (and
without copying the `file_link` key name as the URL).
"""

from onyx.tools.models import LlmPythonExecutionResult, PythonExecutionFile


def test_markdown_defaults_to_none() -> None:
    gen_file = PythonExecutionFile(
        filename="out.txt", file_link="https://x/api/chat/file/abc"
    )
    assert gen_file.markdown is None


def test_generated_files_serialize_markdown_snippet() -> None:
    result = LlmPythonExecutionResult(
        stdout="ok",
        stderr="",
        exit_code=0,
        timed_out=False,
        generated_files=[
            PythonExecutionFile(
                filename="chart.png",
                file_link="https://x/api/chat/file/img-1",
                markdown="![chart.png](https://x/api/chat/file/img-1)",
            ),
            PythonExecutionFile(
                filename="data.csv",
                file_link="https://x/api/chat/file/csv-1",
                markdown="[data.csv](https://x/api/chat/file/csv-1)",
            ),
        ],
    )

    serialized = result.model_dump()
    files = serialized["generated_files"]
    assert files[0]["markdown"] == "![chart.png](https://x/api/chat/file/img-1)"
    assert files[1]["markdown"] == "[data.csv](https://x/api/chat/file/csv-1)"
