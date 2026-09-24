"""Pacing reverse proxy in front of SearXNG.

Serializes POST /search *start times* and spaces them a random
DELAY_MIN..DELAY_MAX seconds apart. GET /config (and any other path)
passes through unpaced, so Onyx's connection test stays fast.

Behavior Onyx relies on:
- GET  /config  -> passthrough (brand check reads brand.GIT_URL)
- POST /search  -> paced passthrough of form data, JSON body returned

The send slot is reserved under the lock, but the upstream fetch runs
outside it: a browser-backed SearXNG search can take tens of seconds, and
holding the lock for the whole round trip would head-of-line block every
queued search until Onyx's client timeout expires.
"""

import os
import random
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = os.environ.get("UPSTREAM", "http://searxng:8080").rstrip("/")
DELAY_MIN = float(os.environ.get("DELAY_MIN", "1"))
DELAY_MAX = float(os.environ.get("DELAY_MAX", "6"))
# Upper bound on time spent queued, so a pile-up can't stall a request forever.
MAX_WAIT = float(os.environ.get("MAX_WAIT", "20"))
UPSTREAM_TIMEOUT = float(os.environ.get("UPSTREAM_TIMEOUT", "60"))

_slot_lock = threading.Lock()
_next_slot = 0.0  # monotonic timestamp of the earliest free /search send slot


def _reserve_send_slot(entered: float) -> float:
    """Reserve the next /search send slot; return how long to sleep first.

    Slots chain off the previously reserved one, so concurrent searches are
    spaced DELAY_MIN..DELAY_MAX apart. A request that has already queued for
    MAX_WAIT is capped at entered + MAX_WAIT so bursts drain instead of
    queueing forever.
    """
    global _next_slot
    with _slot_lock:
        gap = random.uniform(DELAY_MIN, DELAY_MAX)
        now = time.monotonic()
        slot = min(max(_next_slot, now) + gap, entered + MAX_WAIT)
        wait = max(0.0, slot - now)
        # Slots never move backwards, even when the cap pulls one into the past.
        _next_slot = max(slot, now)
        return wait


def _forward(method: str, path: str, body: bytes | None, content_type: str | None):
    req = urllib.request.Request(UPSTREAM + path, data=body, method=method)
    if content_type:
        req.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(req, timeout=UPSTREAM_TIMEOUT) as resp:
            return resp.status, resp.headers.get("Content-Type", "text/plain"), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type", "text/plain"), e.read()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _respond(self, status: int, ctype: str, data: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            # The caller (Onyx) timed out or went away mid-response.
            self.close_connection = True

    def _proxy(self, paced: bool) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        ctype = self.headers.get("Content-Type")

        if paced:
            entered = time.monotonic()
            wait = _reserve_send_slot(entered)
            if wait > 0:
                time.sleep(wait)
            try:
                status, rtype, data = _forward(self.command, self.path, body, ctype)
            except Exception as e:
                print(f"[pacer] upstream error on {self.path}: {e}", flush=True)
                self._respond(502, "text/plain", b"upstream error")
                return
            print(
                f"[pacer] {self.command} {self.path} wait={wait:.1f}s"
                f" upstream={status} bytes={len(data)}",
                flush=True,
            )
        else:
            try:
                status, rtype, data = _forward(self.command, self.path, body, ctype)
            except Exception as e:
                print(f"[pacer] upstream error on {self.path}: {e}", flush=True)
                self._respond(502, "text/plain", b"upstream error")
                return
            print(f"[pacer] {self.command} {self.path} passthrough", flush=True)

        self._respond(status, rtype, data)

    def do_GET(self) -> None:
        self._proxy(paced=False)

    def do_POST(self) -> None:
        self._proxy(paced=self.path.startswith("/search"))

    def log_message(self, fmt: str, *args) -> None:  # silence default stderr noise
        pass


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", 8080), Handler)
    server.daemon_threads = True
    print(
        f"[pacer] up: upstream={UPSTREAM} delay={DELAY_MIN}-{DELAY_MAX}s",
        flush=True,
    )
    server.serve_forever()
