# PROJECT KNOWLEDGE BASE

This file provides guidance to AI agents when working with code in this repository.

## KEY NOTES

- Python deps live in a `uv`-managed virtualenv at `.venv` (repo root). If it doesn't exist yet, create it
  with `uv sync --frozen`, then `source .venv/bin/activate`.
- Test secrets (API keys etc.) are resolved by `backend/tests/utils/aws_secrets.py`, in order: process
  env vars → the gitignored `.vscode/.env` (also used by the pytest commands in `backend/AGENTS.md`;
  create it by copying `.vscode/env_template.txt`) → AWS Secrets Manager (requires `aws sso login`).
  Tests declare what they need via `@pytest.mark.secrets(TestSecret.X)`. If a key you need still can't
  be resolved, ask the user rather than skipping tests.
- If using `playwright` to explore the frontend, log in with username `admin_user@example.com` and password
  `TestPassword123!` (the admin user created by the playwright global setup — see
  `web/tests/e2e/constants.ts`). If it doesn't exist yet, register it via the signup page; the first user
  registered automatically becomes admin. The app can be accessed at `http://localhost:3000`.
  NOTE: on this deployment that user does not exist and signup is invite-only — use the local test user below.
- Local test user (non-admin, Basic group) for manual/live API and UI verification: credentials live in the
  gitignored `.vscode/.env` (`TEST_USER_EMAIL`, `TEST_USER_PASSWORD`, `TEST_USER_ID`). Never copy these
  credentials into tracked files. The workspace is invite-only, so `POST /api/auth/register` is rejected;
  re-create the user with the app's own helpers: run a Python snippet inside `onyx-api_server-1` that calls
  `PasswordHelper().hash(password)` + `assign_user_to_default_groups__no_commit(...)` (mirror
  `backend/tests/external_dependency_unit/conftest.py`), then verify via `POST /api/auth/login`.
- You should assume that all Onyx services are running. To verify, you can check the `backend/log` directory to
  make sure we see logs coming out from the relevant service.
- To connect to the Postgres database, use:
  `PGPASSWORD="${POSTGRES_PASSWORD:-password}" psql -h "${POSTGRES_HOST:-localhost}" -U postgres -c "<SQL>"`.
  This works on a host checkout and inside the devcontainer. If no `psql` client is available, fall back to
  `docker exec onyx-relational_db-1 psql -U postgres -c "<SQL>"` (no `-it` — agent shells have no TTY).
- When making calls to the backend, always go through the frontend. E.g. make a call to `http://localhost:3000/api/persona` not `http://localhost:8080/api/persona`
- `POST /api/auth/login` takes form data (`username` + `password`), not JSON — a JSON body returns 422.
  Save the session cookie (curl `-c` / `-b`) and reuse it for follow-up calls.
- Chat attachments defer processing until send (commit 4f5016f3ca): upload stores the blob with
  `UserFileStatus.UPLOADED`; the first sent message referencing the file flips it to `PROCESSING`
  fire-and-forget (`trigger_user_file_processing` in `onyx/chat/process_message.py`). Sending is never
  gated on processing. Safety net: `check_user_file_processing` also claims stale `UPLOADED` files
  (flips them to `PROCESSING`, excludes incognito uploads — those belong to the session sweep), so a
  file never referenced by a sent message is processed by beat instead of rotting.
- Legacy office formats (.doc, .xls, .odt, .ods) have first-class upload support since commit e3a0b24
  (antiword / xlrd / odfpy extraction, vision path for embedded images). Deployed and verified live on
  2026-10-25 (`MS_WORD_MIME_TYPE` present, `antiword`/`xlrd`/`odfpy` installed in `onyx-api_server-1`).
- EE without a license (deployed 2026-10-28): `docker-compose.override.yml` sets
  `ENABLE_PAID_ENTERPRISE_EDITION_FEATURES=true` + `LICENSE_ENFORCEMENT_ENABLED=false` on `api_server`
  and `background`, so self-hosted tier resolves to ENTERPRISE — admin Usage/Analytics pages, token
  rate limits etc. work with no license installed. BOTH vars are required: flipping
  `LICENSE_ENFORCEMENT_ENABLED` to false alone unsets EE entirely (CE main mounts none of the
  `ee/onyx/server/*` routers → the pages 404). `/api/settings` reports `tier: enterprise`.
- Tool response stats (added 2026-10-28): `open_url` measures its LLM-facing response
  (bytes + tokens) and streams them in `OpenUrlDocuments` (`content_bytes`/`token_count`),
  persists them on `tool_call.response_bytes/response_tokens` (migration 690544b34376), and the
  timeline shows "N KiB · Nk tokens" per step; web-search steps show "N queries · N results";
  uploaded chat documents stream `chat_file_context` packets at turn -1 (measured in
  `build_file_context_stats` at send time, persisted on the USER message's `files` JSONB, and
  replayed via the parent message in `session_loading` — the timeline step shows
  "Reading documents · N · size · tokens").
  Token counting goes through `onyx/llm/engine_token_counter.py`: it POSTs the chat LLM's own
  engine at `{api_base minus /v1}/tokenize` (llama.cpp; requires the provider `api_key` as
  Bearer — 401 without it, and the endpoint lives at server root, NOT under /v1), memoizing
  availability per base URL, and falls back to `get_llm_token_counter`. UI scripts used for
  verification live in `web/tmp/verify_ui.ts` / `web/tmp/poll_dr.ts` /
  `web/tmp/verify_file_ui.ts` (untracked).
- To clean up stuck `UPLOADED` user files (verified 2026-10-25): check references first
  (`chat_message.files` JSONB + `project__user_file`), then run the app's own deletion inside the
  container. `docker exec -i onyx-api_server-1 python -` with
  `SqlEngine.init_engine(pool_size=2, max_overflow=2)` (required — bare scripts fail with
  "Engine not initialized") and `delete_user_file_impl(user_file_id=..., tenant_id="public",
  redis_locking=False)`. It removes the file-store blobs, index chunks and the DB row.
- Deployment rebuilds default to the targeted build. Work in `deployment/docker_compose/` with the
  stack's three compose files (`-f docker-compose.yml -f docker-compose.override.yml
  -f docker-compose.craft.yml`). Backend patches: `build api_server` + `up -d` — `api_server` and
  `background` share the `onyxdotapp/onyx-backend:latest` tag, and the override drops background's
  inherited `build:` (`!reset null`, + `pull_policy: never` so nothing silently pulls the upstream
  image under that tag), so the image is built exactly once. `web_server` only for frontend changes;
  the model servers, searxng and code-interpreter only when their own inputs change. Full builds take
  ~15 min; targeted ~4-7 min.
- Full deploy checklist for backend-image changes (established 2026-10-28):
  1. Check no active user sessions before recreating: `chat_message WHERE time_sent > now() -
     interval '45 minutes'` (empty), no `POST /chat/send-message` / `resume-stream` /
     `handle_stream_message_objects` in api_server logs for ~20 min, no `user_file` rows in
     PROCESSING/UPLOADED, no `Task ... received` in background logs for ~10 min.
  2. Recreate ALL services on the shared image: `up -d api_server background sandbox-proxy`.
     `background` resolves the image by ID (`pull_policy: never`), so after a rebuild it keeps
     running the OLD image while `docker ps` still displays the tag — always recreate it.
  3. Restart nginx: `docker compose ... restart nginx`. Recreated containers get new IPs and nginx
     holds the stale upstream resolution.
  4. Verify: `docker inspect <container> --format '{{.Image}}'` equals
     `docker image inspect onyxdotapp/onyx-backend:latest --format '{{.Id}}'` for all three;
     frontend and `/api/health` return 200 through nginx; all 8 celery workers answer
     `celery_app.control.inspect().ping()` inside `onyx-background-1`.
- SearXNG `/crawl` is single-page only (no pagination following). Onyx-side crawls keep the page's
  visible pagination nav as markdown links (`/topic/X/page/2..6`) plus the "Страница 1 из N" total,
  but IPS drops the last-page link from that nav — the model must construct `.../page/N/` itself
  (open_url tool instructions and OPEN_URL_REMINDER now teach exactly that; verified on prodota
  765-page threads, 2026-10-28).
- Context-size env pins removed from the compose override (2026-10-29): everything derives from the
  live model window now. Sub-agent prompts are bounded only by the model's full input budget
  (256,776 on the live Qwen3.6-35B); intermediate reports cap at the 1/5 window ratio (51,355),
  final report at 2/5 (102,710); input safety margin is the 0.05 default. `download_file` accepts
  6 URLs per call; `open_url` keeps 6 image URLs per scraped page.

## Project Overview

**Onyx** (formerly Danswer) is an open-source Gen-AI and Enterprise Search platform that connects to company documents, apps, and people. It features a modular architecture with both Community Edition (MIT licensed) and Enterprise Edition offerings.

### Technology Stack

- **Backend**: Python 3.13, FastAPI, SQLAlchemy, Alembic, Celery
- **Frontend**: Next.js 16, React 19, TypeScript, Tailwind CSS
- **Database**: PostgreSQL with Redis caching
- **Search**: OpenSearch-backed keyword and vector document index
- **Auth**: OAuth2, SAML, multi-provider support
- **AI/ML**: LangChain, LiteLLM, multiple embedding models

### Repository Layout & Sub-project Guides

Each sub-project has its own agents file with the standards for that area — read it before working
there:

- `backend/` — FastAPI app + Celery workers. `onyx/` is the Community Edition core, `ee/` mirrors its
  layout for Enterprise features, `alembic/` holds migrations, `tests/` the test suites. Standards
  (Celery, migrations, testing, error handling, LLM tracing): `backend/AGENTS.md`.
- `web/` — Next.js frontend. Standards (also cover `desktop/`, the Tauri shell): `web/AGENTS.md`.
- `mobile/` — React Native + Expo app. Standards: `mobile/AGENTS.md`. Mobile differs from web on
  several points (no DOM, NativeWind, expo-router), so do **not** assume the web rules apply there.

Explore the tree with `ls` rather than relying on docs for the full package list.

## Code Quality

```bash
# Install and run pre-commit hooks
pre-commit install
pre-commit run --all-files

# Faster: run only on the files you touched
pre-commit run --files <path> [<path> ...]
```

NOTE: Always make sure everything is strictly typed (both in Python and Typescript).

NOTE: Keep code comments brief and focused on information that stays relevant long-term.

## Writing

These rules apply to all prose you write: docs, commit messages, PR descriptions, reports, and replies.

Follow ASD-STE100 Simplified Technical English for technical text:

- Use approved words only. Each word has one meaning.
- Use one word for one idea. Do not use two words for the same thing.
- Write short sentences. Use 20 words or less for instructions.
- Use active voice. Write "Turn the switch", not "The switch must be turned".
- Write short paragraphs. Keep one topic in each paragraph.
- Keep code comments focused on information that is relevant long-term or for future readers.

## Testing

There are 4 main types of tests: unit, external dependency unit, integration, and playwright e2e
(`web/tests/e2e`). Commands and guidance for all four live in `backend/AGENTS.md`; shared fixtures
and deeper detail in `backend/tests/README.md`. Prefer integration tests over the other types.

## Logs

When (1) writing integration tests or (2) doing live tests (e.g. curl / playwright) you can get access
to logs via the `backend/log/<service_name>_debug.log` file. All Onyx services (api_server, web_server, celery_X)
will be tailing their logs to this file.

## Security Considerations

- Never commit API keys or secrets to the repository
- Use the encrypted credential storage for connector credentials
- Follow existing RBAC patterns for new features

## Creating a Plan

When creating a plan in the `plans` directory (gitignored — create it if it doesn't exist), make sure to
include at least these elements:

**Issues to Address**
What the change is meant to do.

**Important Notes**
Things you come across in your research that are important to the implementation.

**Implementation strategy**
How you are going to make the changes happen. High level approach.

**Tests**
What unit (use rarely), external dependency unit, integration, and playwright tests you plan to write to
verify the correct behavior. Don't overtest. Usually, a given change only needs one type of test.

Do NOT include these: _Timeline_, _Rollback plan_

This is a minimal list - feel free to include more. Do NOT write code as part of your plan.
Keep it high level. You can reference certain files or functions though.

Before writing your plan, make sure to do research. Explore the relevant sections in the codebase.

## Best Practices

In addition to the other content in this file, best practices for contributing
to the codebase can be found in the "Engineering Best Practices" section of
`CONTRIBUTING.md`. Understand its contents and follow them.
