"""Redis-backed state for persistent code-interpreter sandbox sessions.

Maps chat sessions to long-lived code-interpreter sessions so `run_python`
state (workspace files, installed packages) survives across tool calls and
chat turns. All keys are tenant-prefixed via the standard Redis client and
carry a TTL matching the sandbox session TTL, so stale mappings expire on
their own.
"""

import json
import logging
from typing import Any

from onyx.redis.redis_pool import get_redis_client

logger = logging.getLogger(__name__)

_SESSION_KEY_PREFIX = "onyx:code_interpreter:session:"
_STAGED_KEY_PREFIX = "onyx:code_interpreter:staged:"
_WORKSPACE_KEY_PREFIX = "onyx:code_interpreter:workspace:"


def _session_key(chat_session_id: str) -> str:
    return f"{_SESSION_KEY_PREFIX}{chat_session_id}"


def _staged_key(chat_session_id: str) -> str:
    return f"{_STAGED_KEY_PREFIX}{chat_session_id}"


def _workspace_key(chat_session_id: str) -> str:
    return f"{_WORKSPACE_KEY_PREFIX}{chat_session_id}"


def fetch_ci_session_id(chat_session_id: str) -> str | None:
    """Return the mapped code-interpreter session id, or None if unmapped/expired."""
    try:
        value: Any = get_redis_client().get(_session_key(chat_session_id))
    except Exception:
        logger.exception("Failed to fetch code-interpreter session mapping")
        return None
    if value is None:
        return None
    return value.decode() if isinstance(value, bytes) else str(value)


def store_ci_session_id(
    chat_session_id: str, ci_session_id: str, ttl_seconds: int
) -> None:
    """Map a chat session to a code-interpreter session with a matching TTL."""
    try:
        client = get_redis_client()
        client.set(_session_key(chat_session_id), ci_session_id, ex=ttl_seconds)
    except Exception:
        logger.exception("Failed to store code-interpreter session mapping")


def delete_ci_session_id(chat_session_id: str) -> None:
    try:
        get_redis_client().delete(_session_key(chat_session_id))
    except Exception:
        logger.exception("Failed to delete code-interpreter session mapping")


def fetch_staged_file_keys(chat_session_id: str) -> set[str]:
    """File keys ("<filename>:<sha256>") already staged into the session."""
    try:
        values = get_redis_client().smembers(_staged_key(chat_session_id))
    except Exception:
        logger.exception("Failed to fetch staged file keys")
        return set()
    return {v.decode() if isinstance(v, bytes) else str(v) for v in values}


def store_staged_file_keys(
    chat_session_id: str, keys: set[str], ttl_seconds: int
) -> None:
    """Record staged file keys and refresh the set's TTL."""
    if not keys:
        return
    try:
        client = get_redis_client()
        client.sadd(_staged_key(chat_session_id), *keys)
        client.expire(_staged_key(chat_session_id), ttl_seconds)
    except Exception:
        logger.exception("Failed to store staged file keys")


def refresh_session_ttls(chat_session_id: str, ttl_seconds: int) -> None:
    """Refresh both the mapping and staged-set TTLs after a keepalive."""
    try:
        client = get_redis_client()
        client.expire(_session_key(chat_session_id), ttl_seconds)
        client.expire(_staged_key(chat_session_id), ttl_seconds)
    except Exception:
        logger.exception("Failed to refresh code-interpreter session TTLs")


def forget_staged_file_keys(chat_session_id: str) -> None:
    """Clear staged-file tracking (used when a session is recreated)."""
    try:
        get_redis_client().delete(_staged_key(chat_session_id))
    except Exception:
        logger.exception("Failed to clear staged file keys")


def fetch_workspace_file_ids(chat_session_id: str) -> dict[str, str]:
    """Workspace paths already reported for this chat: path -> server file id.

    Used as the dedup baseline for session executions so the server only
    returns new or modified files. Ids may be stale (server-side file TTL);
    the service skips missing ids instead of failing.
    """
    try:
        raw: Any = get_redis_client().get(_workspace_key(chat_session_id))
    except Exception:
        logger.exception("Failed to fetch workspace file ids")
        return {}
    if not raw:
        return {}
    value = raw.decode() if isinstance(raw, bytes) else str(raw)
    try:
        mapping = json.loads(value)
    except json.JSONDecodeError:
        logger.warning("Corrupt workspace file-id map; starting a fresh baseline")
        return {}
    return {str(k): str(v) for k, v in mapping.items()}


def store_workspace_file_ids(
    chat_session_id: str, mapping: dict[str, str], ttl_seconds: int
) -> None:
    """Merge reported workspace file ids into the baseline and refresh its TTL."""
    if not mapping:
        return
    try:
        merged = fetch_workspace_file_ids(chat_session_id)
        merged.update(mapping)
        get_redis_client().set(
            _workspace_key(chat_session_id), json.dumps(merged), ex=ttl_seconds
        )
    except Exception:
        logger.exception("Failed to store workspace file ids")


def forget_workspace_file_ids(chat_session_id: str) -> None:
    try:
        get_redis_client().delete(_workspace_key(chat_session_id))
    except Exception:
        logger.exception("Failed to clear workspace file ids")
