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
- Secondary model for auxiliary flows (2026-11-02): the former "Chat Naming Model" pointer flow
  (`LLMModelFlowType.SECONDARY`, renamed from `CHAT_NAMING` by migration a3f2c81d7e04) now drives a
  small/auxiliary model for ALL secondary flows — query rephrase, keyword expansion, document
  selection, section-relevance classification, time filter, search scope, slack query building,
  session naming, and memory update. Designated via Admin → Chat Preferences → "Secondary Model"
  (`POST/DELETE /api/admin/llm/default-secondary`, response field `default_secondary`); unset →
  every flow falls back to the session/main model. `SearchTool.secondary_llm` (None → main llm) is
  fetched in `tool_constructor._construct_tools_impl` and `server/features/search/api.py`;
  `MemoryTool` gets it too; naming still resolves via `LLMOverride`. Selection content budget and
  expansion history are clamped to the secondary model's `max_input_tokens` (constants in
  `search/constants.py`); all secondary-flow invokes carry `SECONDARY_LLM_FLOW_TIMEOUT_S`.
  Recommended: a dense 3-8B model with ≥32k context (the 25-chunk selection budget is ~25.6k
  tokens). Deployed live 2026-11-02: provider `simple-llm-secondary`
  (openai_compatible → http://172.29.153.35:8013, same Bearer token as the primary
  `simple-llm`), model `/models/Qwen3.5-9B-Q4_K_M.gguf` (shown as "Qwen3.5-9B (secondary)"),
  `max_input_tokens=32768` — matches the server's real `n_ctx` (its 400 error reports it
  directly). The selection token counter deliberately stays on the PRIMARY engine (`self.llm`),
  not the secondary. Server prefill is ~34 tok/s (CPU-class), which is fine for the offload's
  purpose: auxiliary calls stop evicting the single-slot primary's KV cache (no re-prefill of
  the main context), and flows exceeding the 60s `SECONDARY_LLM_FLOW_TIMEOUT_S` abort on the
  secondary without ever touching the primary — they fail open into their graceful fallbacks.
  Net effect: prompts under ~2k tokens (naming, small expansions) genuinely run on the 9B;
  classification (~5k) and selection (~25.6k) mostly fall back until the server's prefill
  speeds up or the env var is raised. INCIDENT 2026-11-02: Qwen3.5 is a thinking model and
  llama.cpp ignores reasoning_effort=OFF, so every flow call generated a multi-thousand-token
  think trace at ~6 tok/s CPU decode; the 9B's decode load also throttled the primary
  35B-A3B on the same host ~13x (1.2k -> 92 tok/s prefill — MoE expert tensors are
  CPU-bandwidth-bound). Fixes: (1) custom_config now supports an `extra_body` JSON passthrough
  (`EXTRA_BODY_CONFIG_KEY` in custom_config_mapping.py, validated at upsert) — the provider is
  configured with `{"chat_template_kwargs": {"enable_thinking": false}}`, verified: flow
  answers in seconds, zero trace; (2) every secondary invoke caps output at
  `SECONDARY_LLM_MAX_OUTPUT_TOKENS` (256) so any thinking model truncates into graceful
  fallbacks instead of running 15 min. Rebalance reevaluation (CPU groups split on .35,
  2026-11-02): primary is UNAFFECTED while the 9B prefills (~1265 vs ~1277 tok/s), but a 9B
  DECODE still throttles it ~9x (~137 tok/s) — acceptable only because thinking is off and
  output is capped, so decode windows are seconds. 9B prefill unchanged (~28 tok/s). Vision
  (2026-11-02): the 9B is vision-capable and holds BOTH the SECONDARY and VISION pointers
  (`supports_image_input=True` + default-vision designation), so indexing image summarization
  (`indexing_pipeline.py`) and the image-analysis tool both run on it; verified live with a
  real image invoke. The SearXNG `captcha_vision` solver in `searxng/settings.yml` stayed on
  the primary (8011) — the fork's solver sends `chat_template_kwargs
  {"enable_thinking": false}` itself for non-Ollama backends.
- Tool response stats (added 2026-10-28, per-doc breakdown 2026-11-01): `open_url` measures its
  LLM-facing response (bytes + tokens) and streams them in `OpenUrlDocuments`
  (`content_bytes`/`token_count`, plus `doc_stats` — per-document breakdown keyed by
  `document_id`, aggregated across sections of one page, covering only documents that made it
  past the char budget). Totals persist on `tool_call.response_bytes/response_tokens`, the
  breakdown on `tool_call.response_doc_stats` (migration 213895a5526b); the timeline shows
  "N KiB · Nk tokens" in the step header and per-chip suffixes on expansion; web-search steps
  show "N queries · N results"; uploaded chat documents stream `chat_file_context` packets at
  turn -1 (measured in `build_file_context_stats` at send time, persisted on the USER message's
  `files` JSONB, and replayed via the parent message in `session_loading` — the timeline step
  shows "Reading documents · N · size · tokens").
  Token counting goes through `onyx/llm/engine_token_counter.py`: it POSTs the chat LLM's own
  engine at `{api_base minus /v1}/tokenize` (llama.cpp; requires the provider `api_key` as
  Bearer — 401 without it, and the endpoint lives at server root, NOT under /v1), and falls
  back to `get_llm_token_counter` (local tiktoken) as a last resort. Since 2026-11-01 every
  call site builds the counter via `get_engine_token_counter` (chat/DR loops, session naming,
  search chunk selection, coding agent, persona sizing, open_url/file stats); a failed probe
  is negative-cached per base URL for 60s, then re-probed. UI scripts used for
  verification live in `web/tmp/verify_ui.ts` / `web/tmp/poll_dr.ts` /
  `web/tmp/verify_file_ui.ts` / `web/tmp/verify_docstats_ui.ts` (untracked).
- Indexed-crawl live E2E green (2026-11-02/03, phase-5 runbook
  `tmp/phase5-live-e2e-runbook.md`): open_url holds its response until saved
  pages finish indexing (`_wait_until_searchable`, `OPEN_URL_INDEX_WAIT_SECONDS=90`,
  2s polls, 0=off), then stamps `IndexedPageRef.search_ready` from the final
  status; the LLM-facing pointer note gates on it — "use internal_search for
  details" only when searchable, "still indexing..." on hold timeout or failure.
  `llm_loop._extend_search_scope_with_indexed_files` adds same-turn
  `indexed_files` to the chat-search scope, so in-turn internal_search sees the
  pages before the files JSONB save lands. `save_chat.py` step 9 dedups indexed
  descriptors by user_file_id within the batch (repeated open_url calls used to
  append one descriptor per call). `doc_indexing.py` uses `setup_logger()` —
  plain stdlib loggers are invisible in docker logs. Verified live: 3 pages
  indexed (~30s hold), truthful notes, scoped search returning only the saved
  files, exact mid-article quote retrieval, dedup reuse on re-open. Diagrams:
  `docs/TOKEN_PRESSURE_FLOW.md`; investigation: `docs/INDEXED_DOC_EMPTY_CONTEXTS.md`.
- Effective search knobs on this deployment (2026-11-03): `MAX_CHUNKS_FED_TO_CHAT=10`
  (at 512-tok chunks this — not the secondary model window — bounds the selection
  prompt; 50 made the ~25.6k-token selection prefill take 13-20 min on CPU and
  tombstone at the 1200s `TOOL_EXECUTION_TIMEOUT_SECONDS` with empty results),
  `SECONDARY_LLM_FLOW_TIMEOUT_S=600`, `TOOL_EXECUTION_TIMEOUT_SECONDS=1200`.
  Cycle-0 "tool args" latency is the main model's think-trace decode (~19 tok/s
  avg): trace length varies run to run — 74s to 10.5 min with no code change;
  do not misattribute it to search/index phases.
- OpenSearch during the PRESENT→FUTURE migration: two indices exist
  (`danswer_chunk_<model>` = PRESENT, `...__danswer_alt_index` = FUTURE); chat
  search reads FUTURE, the port re-embeds PRESENT→FUTURE create-only, and
  deleted files' PRESENT copies linger until the port's orphan sweep. Query the
  right index before concluding chunks are missing — live checks must count
  `__danswer_alt_index` for anything indexed after the migration started.
  User-file chunks currently carry empty `chunk_context`/`doc_summary` (no
  contextual enrichment) — investigation + repro in
  `docs/INDEXED_DOC_EMPTY_CONTEXTS.md`.
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
- Chat-scoped internal search + default-on tool (2026-11-01, local divergence from upstream):
  web no longer seeds `internal_search` as "disabled" per chat (`withSearchDefault` removed from
  `web/src/lib/tools/hooks.ts`; mobile never had the seed). The chat send path scopes the tool to
  the current session's attachments: `SearchToolConfig.chat_document_ids` (user_file ids from
  `chat_message.files` JSONB) merge into `PersonaSearchInfo.attached_document_ids` → the existing
  OR-based knowledge filter in OpenSearch (`attached_document_ids`). Empty scope (no attachments,
  no project/persona knowledge) short-circuits with an honest "nothing to search" response in
  `SearchTool.run` (`chat_scoped` + `_has_knowledge_scope`) instead of falling through to an
  unfiltered search that would surface files from unrelated chats. Project/persona scope filters
  (`project_id_filter`/`persona_id_filter`, set only on context-file overflow) are unchanged.
  Non-chat `SearchTool` callers (Search UI, Slack) pass no `chat_document_ids` → unrestricted,
  as before. Verified live: session with no attachments + forced search returns 0 docs
  (`tool_call` row, empty `tool_call__search_doc`); attached file returned as the only result.
  Verification script: `tmp/verify_scoped_search.py` (untracked; note `/api/chat/send-chat-message`
  is the endpoint, and fd `type` must be a `ChatFileType` value like `plain_text`).
  Live E2E gotcha: the local model server can take 5+ minutes per forced-search turn; poll logs
  instead of assuming a hang.
- Tool row gating aligned with backend availability (2026-11-01, follows the scoping change):
  `ToolLineItem` grayed internal search when `ccPairs.length === 0`, and
  `/api/manage/connector-status` excludes the auto-created `DefaultCCPair` — so user-file-only
  deployments saw the row permanently grayed with an "add connectors" gear. Both gates
  (`needsConnectors` in ToolLineItem, the Basic-user visibility filter in ToolsPopover) now use
  the backend's own availability listing (`/api/tool`, filtered by `SearchTool.is_available`,
  which counts user files). Verified headless: row renders enabled with zero connectors.
- Contextual-RAG reindex incident (2026-11-02): switching the Context Retrieval LLM to the
  THINKING Qwen3.6-35B (model_configuration 27, provider `simple-llm`, no extra_body) made user-file
  port units fail 2h+ each: `add_document_summaries` (indexing_pipeline.py) sends the FULL document
  (n_ctx 262144 → docs up to 256k tokens, 21.5k full-doc-inclusion threshold) and is the ONLY
  unguarded contextual call — one `CONTEXTUAL_RAG_LLM_TIMEOUT` (180s, env-tunable on background)
  breach fails the batch → 5× in-task retries → port attempt FAILED → scheduler retry. Chunk
  contexts (`assign_context`) swallow all errors into `chunk_context=""` silently. Prefill measured
  235 tok/s under load vs ~1265 idle → a 60k-token summary prompt = 255s loaded / 48s idle: the
  180s budget only holds on an idle server. Qwen thinking goes to a separate `reasoning_content`
  field and DID still fit ~524-token output caps (0 empty contexts), but wastes decode time.
  Fix applied: killed the running port attempt (cancel_requested flag, honored at batch boundaries),
  deleted all port_attempt rows for search_settings 7 (scheduler only recreates from FAILED/none —
  SUCCESS/CANCELED units never re-run), repointed FUTURE row to model_configuration 29 (the
  "(secondary)" display name, provider 3 `simple-llm-secondary` whose custom_config carries
  `extra_body {"chat_template_kwargs": {"enable_thinking": false}}`; verified live: content
  non-empty, reasoning None). NOTE: litellm FLATTENS extra_body into the request body — testing
  llama.cpp with a literal `extra_body` wrapper field silently does nothing. REMAINING RISK: user
  2a644c1a owns 4× 453-chunk 186k-token docx; their summary prefill is 147s best-idle vs 180s —
  needs a raised CONTEXTUAL_RAG_LLM_TIMEOUT and/or a clamped summary prompt in the fork.
- Model-server KV reuse (learned 2026-11-02): the llama.cpp 0.7.0+ fork at 8011 keeps a 16 GB
  RAM-side pool of past KV prefills; a new call matching a similar prefill copies RAM→VRAM
  instead of recomputing (eviction FIFO). Serial indexing (MAX_CONTEXTUAL_RAG_WORKERS=1, set in
  the compose override alongside CONTEXTUAL_RAG_LLM_TIMEOUT=900) maximizes hits: one full-doc
  prefill per doc (the summary call), then chunk-context calls ride the cached doc prefix.
  Note ≤21.5k-token docs embed the FULL doc in every chunk prompt (biggest reuse win); >21.5k
  docs use the ~500-token summary as doc_info, so their chunk calls are cheap regardless.
  Whether a 186k-token doc's KV block fits the 16 GB pool at all is model-arch dependent —
  watch per-doc chunk-call cadence (flat ~4-5s = misses, ~1-2s after the first call = hits).
  Also deployed 2026-11-02: indexing_pipeline guards add_document_summaries + the fallback
  summary in add_chunk_summaries (degrade to empty enrichment instead of failing the batch),
  fallback retries 2× with 5s sleep (FALLBACK_DOC_SUMMARY_RETRIES / _RETRY_SLEEP_S).
  CONTEXTUAL_RAG_FULL_DOC_INCLUSION env-ized in context_budgets.py ('num/den' format, default
  256/3125) and set to 1/4 (65,536 @ 262k window) in the override: the 23k-60k docs get
  full-doc chunk contexts (Q4 KV pool makes repeats ~free), 186k docs stay on summary-based
  contexts (needle-in-haystack genericity). api_server/sandbox-proxy still run the PREVIOUS
  image build (inert change) — recreate all three on the next backend rollout per checklist.
- Port stall-loop fix (2026-11-03 night): the port enriches ONE doc per page with no heartbeat
  during enrichment (port_copy.py:109-111, "kept under the stall threshold") — at ~19 tok/s
  decode a 453-chunk doc = ~75 min silent vs the hardcoded 30-min stall watchdog → every
  attempt of the 186k-docx unit was stall-killed, cursor never advanced (5 same-cursor
  failures → auto-pause/backoff). Fix: PORT_STALL_THRESHOLD_SECONDS env-ized on
  _PORT_STALL_THRESHOLD_SECONDS (tasks.py, default 1800), set 7200 on background. Gotcha:
  raising the threshold means the sweep won't reap an ABANDONED IN_PROGRESS row for 2h —
  terminalize it manually (UPDATE ... status='FAILED') and delete the unit's failed rows to
  reset the same-cursor streak before expecting the scheduler to re-enqueue. With KV hits
  (~97% cached tokens observed via the fork's /metrics) decode is the sole bottleneck:
  ~8-13s per chunk context, ~48 min for a 290-chunk unit.
- Deep Research token-pressure wiring (2026-11-04, deployed and live-verified):
  DR's `construct_message_history` call sites now pass `token_counter` +
  per-run digest caches + `llm` (dr_loop.py, research_agent.py), so
  orchestrator/sub-agent compaction stubs carry digests instead of bare
  notices — at `DR_FORCE_REPORT_S=172800` (48h) a run overflows the window
  many times and digests are what keeps findings alive. Sub-agent
  `open_url` responses extend the run-wide search scope immediately
  (`extend_search_scope_with_indexed_files`, renamed public from
  `_extend_search_scope_with_indexed_files`). Thinking-model empty-report
  fix: the intermediate report retry appends `RESEARCH_REPORT_RETRY_NUDGE`
  (run 1 failed with reasoning-only output at finish_reason=stop; run 2
  with the fix wrote both reports + a cited final answer). Regression
  test: `backend/tests/unit/onyx/deep_research/test_dr_compaction_wiring.py`;
  runbook docs in `docs/TOKEN_PRESSURE_FLOW.md`. Follow-up hardening
  (2026-11-04, later deploy): tier-2 digest upgrades route to the
  SECONDARY model (`_resolve_upgrade_llm`; no-thinking provider tuning
  applies) with the chat model as fallback, the digest call's output cap
  includes the thinking reserve on thinking models (run 2's upgrades
  burned the answer-only cap inside the think trace → empty summaries),
  per-attempt wall clock 1800s (`GEN_AI_TOOL_RESPONSE_SUMMARIZATION_TIMEOUT_SECONDS`,
  double headroom over queue+prefill+decode here), and
  `SECONDARY_LLM_MAX_OUTPUT_TOKENS=1024` (256 was the thinking-truncation
  guard; thinking is off on the secondary). Second deploy iteration
  (same day): `DR_THINKING_MODEL_OVERRIDES=/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`
  — the registry cannot classify the custom gguf, so the override drives
  the thinking reserve for digest calls and DR tool-call steps (first
  attempt instead of discover-by-retry). Topology correction: the
  SECONDARY flow now designates the 35B via provider
  `simple-llm-secondary` (carries the `enable_thinking: false`
  extra_body), and BOTH providers point at
  `http://172.29.153.35:8011` — the 9B engine at 8013 referenced in the
  2026-11-02/03 notes below is retired, so upgrades still queue on the
  shared 8011 server (the win is no-thinking output + provider policy,
  not a separate slot). Remaining follow-up: citation
  markers can be lost when a report compacts (ledger idea in the doc).
- Telegram channel crawls and URL identity (2026-11-01): the SearXNG fork transparently serves
  bare `t.me/<channel>` crawls from the `t.me/s/<channel>` web preview, so `RenderedPage.final_url`
  carries the `/s/` URL. `open_url` matches fetched content back to the requested URL — a crawled
  page's `WebContent.link` must stay the requested URL for such transparent rewrites
  (`unwrap_telegram_preview_url` in `onyx/utils/url.py`, applied in the crawler's two browser
  paths). When adding more transparent rewrites on the SearXNG side, mirror the identity rule here
  or `open_url` silently drops the fetched content ("Failed to fetch content from the requested
  resources.").

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
