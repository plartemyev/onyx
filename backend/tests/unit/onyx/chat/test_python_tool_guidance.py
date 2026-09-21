"""Guards the run_python tool guidance: the network section must match the
deployment's sandbox configuration (PYTHON_EXECUTOR_DOCKER_NETWORK), the
{network_guidance} placeholder must always be substituted, and the artifact
persistence wording must stay present so the LLM knows generated files carry
across executions."""

from unittest.mock import MagicMock, patch

from onyx.chat.prompt_utils import build_system_prompt
from onyx.tools.tool_implementations.python.python_tool import PythonTool

ENABLED_MARKER = "Internet access is available in the sandbox"
DISABLED_MARKER = "Internet access for this session is disabled"
DOWNLOAD_MARKER = "downloading files such as images"
ARTIFACT_MARKER = "Files written by a previous call ARE available in later calls"
PLACEHOLDER = "{network_guidance}"
# The sandbox venv has no pip and the executing user cannot write to any
# site-packages, so plain `pip install` always fails. The only working install
# path (network enabled) is `pip install --target` into the writable workspace
# plus a sys.path insert.
BROKEN_PIP_ADVICE_MARKER = "can be installed with `pip install`"
PIP_TARGET_MARKER = "--target"
# uv is not available in the sandbox (no accessible binary; workspace tmpfs is
# noexec), so the guidance must say so to save the LLM failed attempts.
NO_UV_MARKER = "`uv` is not available"


def _prompt(network_enabled: bool, *, with_tool: bool = False) -> str:
    tools = [PythonTool(tool_id=1, emitter=MagicMock())] if with_tool else None
    with (
        patch("onyx.chat.prompt_utils.get_company_context", return_value=None),
        patch(
            "onyx.chat.prompt_utils.PYTHON_SANDBOX_NETWORK_ENABLED",
            network_enabled,
        ),
    ):
        return build_system_prompt(
            "Base prompt.",
            tools=tools,
            include_all_guidance=not with_tool,
        )


def test_network_enabled_guidance_when_sandbox_has_network() -> None:
    prompt = _prompt(True)
    assert ENABLED_MARKER in prompt
    assert DISABLED_MARKER not in prompt
    assert DOWNLOAD_MARKER in prompt
    # Plain `pip install` never works in the sandbox; the guidance must not
    # promise it and must give the working --target install pattern instead.
    assert BROKEN_PIP_ADVICE_MARKER not in prompt
    assert PIP_TARGET_MARKER in prompt
    assert NO_UV_MARKER in prompt
    assert PLACEHOLDER not in prompt


def test_network_disabled_guidance_when_sandbox_isolated() -> None:
    prompt = _prompt(False)
    assert DISABLED_MARKER in prompt
    assert ENABLED_MARKER not in prompt
    # Downloading web images requires network access; must not be promised
    # when the sandbox is isolated.
    assert DOWNLOAD_MARKER not in prompt
    # Package installs need network; the install pattern must not be promised.
    assert PIP_TARGET_MARKER not in prompt
    assert PLACEHOLDER not in prompt


def test_guidance_present_when_python_tool_in_tools() -> None:
    prompt = _prompt(True, with_tool=True)
    assert "## run_python" in prompt
    assert ENABLED_MARKER in prompt
    assert PLACEHOLDER not in prompt


def test_artifact_persistence_wording_always_present() -> None:
    for prompt in (_prompt(True), _prompt(False)):
        assert ARTIFACT_MARKER in prompt
