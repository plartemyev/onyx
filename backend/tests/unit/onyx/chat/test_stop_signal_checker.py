"""Unit tests for stop_signal_checker and chat_processing_checker.

These modules are safety-critical — they control whether a chat stream
continues or stops.  The tests use a simple in-memory CacheBackend stub
so no external services are needed.
"""

from uuid import uuid4

from onyx.cache.interface import CacheBackend, CacheLock
from onyx.chat.chat_processing_checker import (
    get_processing_run_id,
    is_chat_session_processing,
    set_processing_status,
)
from onyx.chat.stop_signal_checker import (
    FENCE_TTL,
    is_connected,
    reset_cancel_status,
    set_fence,
)


class _MemoryCacheBackend(CacheBackend):
    """Minimal in-memory CacheBackend for unit tests."""

    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}

    def get(self, key: str) -> bytes | None:
        return self._store.get(key)

    def getdel(self, key: str) -> bytes | None:
        return self._store.pop(key, None)

    def set(
        self,
        key: str,
        value: str | bytes | int | float,
        ex: int | None = None,  # noqa: ARG002
    ) -> None:
        if isinstance(value, bytes):
            self._store[key] = value
        else:
            self._store[key] = str(value).encode()

    def set_if_absent(
        self,
        key: str,
        value: str | bytes | int | float,
        ex: int | None = None,
    ) -> bool:
        if key in self._store:
            return False
        self.set(key, value, ex=ex)
        return True

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def exists(self, key: str) -> bool:
        return key in self._store

    def expire(self, key: str, seconds: int) -> None:
        pass

    def ttl(self, key: str) -> int:
        return -2 if key not in self._store else -1

    def lock(self, name: str, timeout: float | None = None) -> CacheLock:
        raise NotImplementedError

    def rpush(self, key: str, value: str | bytes) -> None:
        raise NotImplementedError

    def blpop(self, keys: list[str], timeout: int = 0) -> tuple[bytes, bytes] | None:
        raise NotImplementedError


# ── stop_signal_checker ──────────────────────────────────────────────


class TestSetFence:
    def test_set_fence_true_creates_key(self) -> None:
        cache = _MemoryCacheBackend()
        sid = uuid4()
        set_fence(sid, cache, True)
        assert not is_connected(sid, cache)

    def test_set_fence_false_removes_key(self) -> None:
        cache = _MemoryCacheBackend()
        sid = uuid4()
        set_fence(sid, cache, True)
        set_fence(sid, cache, False)
        assert is_connected(sid, cache)

    def test_set_fence_false_noop_when_absent(self) -> None:
        cache = _MemoryCacheBackend()
        sid = uuid4()
        set_fence(sid, cache, False)
        assert is_connected(sid, cache)

    def test_set_fence_uses_ttl(self) -> None:
        """Verify set_fence passes ex=FENCE_TTL to cache.set."""
        calls: list[dict[str, object]] = []
        cache = _MemoryCacheBackend()
        original_set = cache.set

        def tracking_set(
            key: str,
            value: str | bytes | int | float,
            ex: int | None = None,
        ) -> None:
            calls.append({"key": key, "ex": ex})
            original_set(key, value, ex=ex)

        cache.set = tracking_set  # ty: ignore[invalid-assignment]

        set_fence(uuid4(), cache, True)
        assert len(calls) == 1
        assert calls[0]["ex"] == FENCE_TTL


class TestIsConnected:
    def test_connected_when_no_fence(self) -> None:
        cache = _MemoryCacheBackend()
        assert is_connected(uuid4(), cache)

    def test_disconnected_when_fence_set(self) -> None:
        cache = _MemoryCacheBackend()
        sid = uuid4()
        set_fence(sid, cache, True)
        assert not is_connected(sid, cache)

    def test_sessions_are_isolated(self) -> None:
        cache = _MemoryCacheBackend()
        sid1, sid2 = uuid4(), uuid4()
        set_fence(sid1, cache, True)
        assert not is_connected(sid1, cache)
        assert is_connected(sid2, cache)


class TestResetCancelStatus:
    def test_clears_fence(self) -> None:
        cache = _MemoryCacheBackend()
        sid = uuid4()
        set_fence(sid, cache, True)
        reset_cancel_status(sid, cache)
        assert is_connected(sid, cache)

    def test_noop_when_no_fence(self) -> None:
        cache = _MemoryCacheBackend()
        reset_cancel_status(uuid4(), cache)


# ── chat_processing_checker ──────────────────────────────────────────


class TestSetProcessingStatus:
    def test_set_true_marks_processing(self) -> None:
        cache = _MemoryCacheBackend()
        sid = uuid4()
        set_processing_status(sid, cache, True)
        assert is_chat_session_processing(sid, cache)

    def test_set_false_clears_processing(self) -> None:
        cache = _MemoryCacheBackend()
        sid = uuid4()
        set_processing_status(sid, cache, True)
        set_processing_status(sid, cache, False)
        assert not is_chat_session_processing(sid, cache)

    def test_older_run_refresh_never_moves_fence_backward(self) -> None:
        """Two runs on one session refresh the same fence key; resume readers
        must keep seeing the newest run's buffer (chat 4dc4f5a9: runs 251/257
        flip-flopped the fence)."""
        cache = _MemoryCacheBackend()
        sid = uuid4()
        set_processing_status(sid, cache, True, run_id=257)
        set_processing_status(sid, cache, True, run_id=251)
        assert get_processing_run_id(sid, cache) == 257

    def test_older_run_end_does_not_clear_newer_fence(self) -> None:
        cache = _MemoryCacheBackend()
        sid = uuid4()
        set_processing_status(sid, cache, True, run_id=257)
        set_processing_status(sid, cache, False, run_id=251)
        assert is_chat_session_processing(sid, cache)
        assert get_processing_run_id(sid, cache) == 257

    def test_newest_run_end_clears_fence(self) -> None:
        cache = _MemoryCacheBackend()
        sid = uuid4()
        set_processing_status(sid, cache, True, run_id=257)
        set_processing_status(sid, cache, False, run_id=257)
        assert not is_chat_session_processing(sid, cache)

    def test_newer_run_end_falls_back_to_older_active_run(self) -> None:
        """The newer run can finish first (run 257 errored while 251 ran on).
        Its clear leaves a transient window with no fence; the older run's next
        refresh re-arms the fence with its own run id."""
        cache = _MemoryCacheBackend()
        sid = uuid4()
        set_processing_status(sid, cache, True, run_id=251)
        set_processing_status(sid, cache, True, run_id=257)
        set_processing_status(sid, cache, False, run_id=257)
        assert get_processing_run_id(sid, cache) is None
        set_processing_status(sid, cache, True, run_id=251)
        assert get_processing_run_id(sid, cache) == 251


class TestIsChatSessionProcessing:
    def test_not_processing_by_default(self) -> None:
        cache = _MemoryCacheBackend()
        assert not is_chat_session_processing(uuid4(), cache)

    def test_sessions_are_isolated(self) -> None:
        cache = _MemoryCacheBackend()
        sid1, sid2 = uuid4(), uuid4()
        set_processing_status(sid1, cache, True)
        assert is_chat_session_processing(sid1, cache)
        assert not is_chat_session_processing(sid2, cache)
