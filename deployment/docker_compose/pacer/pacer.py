"""Pacing reverse proxy in front of SearXNG.

Serializes POST /search requests and spaces them a random
DELAY_MIN..DELAY_MAX seconds apart. GET /config (and any other path)
passes through unpaced, so Onyx's connection test stays fast.

Behavior Onyx relies on:
- GET  /config  -> passthrough (brand check reads brand.GIT_URL)
- POST /search  -> paced passthrough of form data, JSON body returned
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

_lock = threading.Lock()
_last_sent = 0.0  # monotonic timestamp of the last forwarded /search


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
        self.wfile.write(data)

    def _proxy(self, paced: bool) -> None:
        global _last_sent
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        ctype = self.headers.get("Content-Type")

        if paced:
            entered = time.monotonic()
            with _lock:
                gap = random.uniform(DELAY_MIN, DELAY_MAX)
                now = time.monotonic()
                wait = min(
                    max(0.0, _last_sent + gap - now),
                    max(0.0, MAX_WAIT - (now - entered)),
                )
                if wait > 0:
                    time.sleep(wait)
                _last_sent = time.monotonic()
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
    print(
        f"[pacer] up: upstream={UPSTREAM} delay={DELAY_MIN}-{DELAY_MAX}s",
        flush=True,
    )
    server.serve_forever()
