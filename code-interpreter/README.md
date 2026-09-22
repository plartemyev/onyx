# Code Interpreter — Onyx sandbox fork

Fork of the `onyxdotapp/code-interpreter` sandbox stack (service `0.4.7` base,
forked at `0.5.0`) that adds **persistent sessions** for multi-step research:
workspace files, installed packages, and other state survive across
`run_python` calls and chat turns.

Layout:

- `service/` — the FastAPI control plane (manages executor containers, never
  runs user Python itself). Docker-out-of-docker or sibling-container topology
  via the host Docker socket.
- `executor/` — the Arch Linux research executor image (see below).

## What the fork changes

### Persistent sessions (service, 0.5.0)

The upstream service already had short-lived session containers with a bash
route. This fork makes them a real research environment:

- `POST /v1/sessions/{id}/python` and `/python/stream` — run Python inside the
  session with streaming output; returns new/modified workspace files.
  The request's `files` list is a **dedup baseline** (files the caller already
  has), not a staging list — missing IDs are skipped, so stale references never
  fail a call.
- `POST /v1/sessions/{id}/keepalive` — extends the session expiry.
- `POST /v1/sessions/{id}/files` — stage an extra file into the workspace.
- `GET /v1/sessions/{id}/files[/{path}]` — list the workspace / read a file.
- Session workspaces are **named volumes** (not tmpfs), owned by the
  execution user, so files persist across executions.
- A per-session **venv** is created at `/workspace/.venv`
  (`--system-site-packages`, pip bootstrapped) and put first on PATH, so plain
  `pip install` inside session code lands on the workspace volume.
- Expiry lives in a file inside the workspace (docker labels are immutable),
  so keepalives survive service restarts. `SESSION_MAX_LIFETIME_SEC` bounds
  the container's idle sleep, guaranteeing teardown even if this service dies.
- `network_enabled` on session create: explicit True/False per session, None
  defers to `SESSION_NETWORK_MODE`. In the Docker backend, "network" means
  joining `PYTHON_EXECUTOR_DOCKER_NETWORK`.
- Deletion and the TTL reaper remove the workspace volume too.

New env vars: `SESSION_MAX_LIFETIME_SEC`, `SESSION_NETWORK_MODE`,
`SESSION_EXECUTOR_IMAGE`, `SESSION_VENV_ENABLED`.

### Arch Linux research executor (`executor/`)

Replacement for `onyxdotapp/python-executor-sci`, built from a pinned dated
Arch snapshot (`archlinux:base-20260920.0.596911`):

- pacman: `python`, `python-pip`, `uv`, `python-poetry`, `ffmpeg`,
  `libsndfile`, `git`
- uv-installed research stack in `/opt/executor-venv` (requirements.in):
  numpy, scipy, pandas, matplotlib, pillow, opencv-python-headless, librosa,
  soundfile, requests, httpx, openpyxl
- same security posture as upstream (non-root uid 65532, cap-drop ALL,
  no-new-privileges, pids/memory/cpu limits)
- resolved versions recorded into the image
  (`/opt/executor-requirements.lock`, `/opt/pacman-packages.txt`)

Build (must run before the service can use it — executor containers run with
`--pull never`):

```bash
./executor/build.sh            # tags code-interpreter-executor:arch-20260920
```

## Deployment wiring

`deployment/docker_compose/docker-compose.override.yml` in the main repo
builds this service and points it at the Arch executor image, with tuned
per-execution caps (`MAX_EXEC_TIMEOUT_MS`, `CPU_TIME_LIMIT_SEC`,
`MEMORY_LIMIT_MB`, `MAX_FILE_SIZE_MB`). The api side enables sessions with
`CODE_INTERPRETER_SESSIONS_ENABLED` and sets the session TTL plus staging
caps (`CODE_INTERPRETER_*` in `backend/onyx/configs/app_configs.py`).

## How Onyx uses sessions

`backend/onyx/tools/tool_implementations/python/python_tool.py` maps each chat
to one session (Redis-backed, TTL-matched): first `run_python` call creates the
session, later calls keepalive and reuse it. Files uploaded to the chat are
staged once per session; a dedup baseline of already-reported workspace files
keeps large artifacts from being re-downloaded on every call. A vanished
session (TTL, reaper, service restart) is recreated transparently. Deployments
on an older service fall back to the legacy ephemeral path automatically
(version-gated via `/health`).

## Tests

```bash
cd service
uv run --group dev pytest tests/ -q                    # route tests (stub executor)
PYTHON_EXECUTOR_DOCKER_IMAGE=code-interpreter-executor:arch-20260920 \
  uv run --group dev pytest tests/test_executor_docker_sessions.py -q  # real containers
```
