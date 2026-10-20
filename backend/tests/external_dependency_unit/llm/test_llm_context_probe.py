"""Live probe against the deployment's llama.cpp server.

Set TEST_LLAMACPP_BASE (e.g. http://172.29.153.35:8011) and, if the server
requires auth, TEST_LLAMACPP_API_KEY. Skipped when the base is unset.
"""

import os

import pytest
import requests

from onyx.llm.context_resolution import probe_context_window

BASE = os.environ.get("TEST_LLAMACPP_BASE")
API_KEY = os.environ.get("TEST_LLAMACPP_API_KEY")

pytestmark = pytest.mark.skipif(not BASE, reason="TEST_LLAMACPP_BASE not set")


@pytest.mark.external_dependency_unit
def test_probe_reads_live_context_window() -> None:
    headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
    props = requests.get(f"{BASE.rstrip('/')}/props", headers=headers, timeout=5)
    props.raise_for_status()
    expected = props.json()["default_generation_settings"]["n_ctx"]

    assert probe_context_window(BASE, API_KEY, "/models/any.gguf") == expected
