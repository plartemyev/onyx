"""Guards the run_python tool guidance.

Two guidance variants exist, selected by whether the deployment actually runs
a persistent-session code-interpreter (>= 0.5.0):

- session mode: the sandbox keeps files and installed packages for the whole
  chat, so the guidance must say so (and must not give the broken legacy pip
  workaround);
- legacy mode: each call is a fresh sandbox, so the guidance must warn that
  installs do not persist and describe the `pip install --target` workaround.

The network section must always match the deployment's
PYTHON_SANDBOX_NETWORK_ENABLED, and the {network_guidance} placeholder must
always be substituted.
"""

from unittest.mock import MagicMock, patch

from onyx.chat.prompt_utils import build_system_prompt
from onyx.tools.tool_implementations.python.python_tool import PythonTool

ENABLED_MARKER = "Internet access is available in the sandbox"
DISABLED_MARKER = "Internet access for this session is disabled"
DOWNLOAD_MARKER = "use the `download_file` tool instead of saving it in code"
PLACEHOLDER = "{network_guidance}"

# PlantUML wording (gated on the executor image actually shipping it).
PLANTUML_MARKER = "PlantUML are preinstalled"
PLANTUML_CHECK_MARKER = "plantuml -checkonly"

# Session-mode wording.
SESSION_MARKER = "The sandbox is persistent for this chat"
SESSION_INSTALL_MARKER = "installs persist for the rest of the chat"

# Legacy-mode wording. The legacy sandbox venv has no usable pip and the
# executing user cannot write to any site-packages, so plain `pip install`
# always fails there; the only working install path (network enabled) is
# `pip install --target` into the writable workspace plus a sys.path insert.
LEGACY_FRESH_SANDBOX_MARKER = "each call to this tool runs in a fresh sandbox"
LEGACY_ARTIFACT_MARKER = "ARE available in later calls by filename"
LEGACY_NO_TMP_MARKER = "Do not write files to `/tmp`"
LEGACY_BROKEN_PIP_ADVICE_MARKER = "can be installed with `pip install`"
LEGACY_PIP_TARGET_MARKER = "--target"
LEGACY_NO_UV_MARKER = "`uv` is not available"


def _prompt(
    network_enabled: bool,
    *,
    sessions: bool = False,
    with_tool: bool = False,
    plantuml: bool = False,
) -> str:
    tools = [PythonTool(tool_id=1, emitter=MagicMock())] if with_tool else None
    with (
        patch("onyx.chat.prompt_utils.get_company_context", return_value=None),
        patch(
            "onyx.chat.prompt_utils.PYTHON_SANDBOX_NETWORK_ENABLED",
            network_enabled,
        ),
        patch(
            "onyx.chat.prompt_utils.PYTHON_SANDBOX_PLANTUML",
            plantuml,
        ),
        patch(
            "onyx.chat.prompt_utils._python_tool_sessions_available",
            return_value=sessions,
        ),
    ):
        return build_system_prompt(
            "Base prompt.",
            tools=tools,
            include_all_guidance=not with_tool,
        )


def test_session_mode_guidance_when_sessions_available() -> None:
    prompt = _prompt(True, sessions=True, plantuml=True)
    assert SESSION_MARKER in prompt
    assert SESSION_INSTALL_MARKER in prompt
    assert ENABLED_MARKER in prompt
    assert DOWNLOAD_MARKER in prompt
    # The dedicated internet tools come first; sandbox fetching is the
    # fallback for what they cannot do.
    assert "open_url` tool instead of fetching it in code" in prompt
    assert PLACEHOLDER not in prompt
    # The legacy warnings are wrong for session mode and must not appear.
    assert LEGACY_FRESH_SANDBOX_MARKER not in prompt
    assert LEGACY_PIP_TARGET_MARKER not in prompt
    assert LEGACY_NO_UV_MARKER not in prompt


def test_plantuml_guidance_gated_on_executor_capability() -> None:
    with_plantuml = _prompt(True, sessions=True, plantuml=True)
    assert PLANTUML_MARKER in with_plantuml
    assert PLANTUML_CHECK_MARKER in with_plantuml

    without_flag = _prompt(True, sessions=True, plantuml=False)
    assert PLANTUML_MARKER not in without_flag

    # The sessionless sandbox has no PlantUML either, even with the flag set:
    # the legacy template does not carry the section.
    legacy = _prompt(True, sessions=False, plantuml=True)
    assert PLANTUML_MARKER not in legacy


def test_session_mode_guidance_without_network() -> None:
    prompt = _prompt(False, sessions=True)
    assert SESSION_MARKER in prompt
    assert DISABLED_MARKER in prompt
    assert ENABLED_MARKER not in prompt
    assert DOWNLOAD_MARKER not in prompt
    assert PLACEHOLDER not in prompt


def test_legacy_guidance_when_sessions_unavailable() -> None:
    prompt = _prompt(True, sessions=False)
    assert LEGACY_FRESH_SANDBOX_MARKER in prompt
    assert LEGACY_ARTIFACT_MARKER in prompt
    # Never write outside the working directory: /tmp is wiped between calls.
    assert LEGACY_NO_TMP_MARKER in prompt
    assert ENABLED_MARKER in prompt
    # Plain `pip install` never works in the legacy sandbox; the guidance must
    # not promise it and must give the working --target install pattern instead.
    assert LEGACY_BROKEN_PIP_ADVICE_MARKER not in prompt
    assert LEGACY_PIP_TARGET_MARKER in prompt
    assert LEGACY_NO_UV_MARKER in prompt
    assert PLACEHOLDER not in prompt


def test_legacy_guidance_without_network() -> None:
    prompt = _prompt(False, sessions=False)
    assert DISABLED_MARKER in prompt
    assert ENABLED_MARKER not in prompt
    # Downloading web images requires network access; must not be promised
    # when the sandbox is isolated.
    assert DOWNLOAD_MARKER not in prompt
    # Package installs need network; the install pattern must not be promised.
    assert LEGACY_PIP_TARGET_MARKER not in prompt
    assert LEGACY_ARTIFACT_MARKER in prompt
    assert PLACEHOLDER not in prompt


def test_guidance_present_when_python_tool_in_tools() -> None:
    prompt = _prompt(True, with_tool=True)
    assert "## run_python" in prompt
    assert ENABLED_MARKER in prompt
    assert PLACEHOLDER not in prompt
