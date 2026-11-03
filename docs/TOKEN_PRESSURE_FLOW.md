# Token-pressure flow: indexed crawls, lazy digests, oldest-first compaction

How a regular chat turn behaves under context pressure after the
token-pressure optimization work (2026-10/11). Live budgets derive from the
selected model's context window (probe -> LiteLLM map -> 32k fallback); on
this deployment: window 262,144, input budget 243,937 (0.05 margin).

Design rules, in the order they apply:

1. **Fetch is decoupled from context.** `open_url` may fetch up to 180k
   chars per URL / 540k per batch. Pages at or above 6k chars are saved as
   indexed user files; the model inlines only a 4k-char excerpt plus a
   pointer (`indexed_as`) per saved page.
2. **Newest content stays raw.** When the in-turn tail overflows, the newest
   exchanges replay verbatim; older exchanges stub (digest + note), then
   drop, oldest first. Same for history before the last user message on
   later turns.
3. **Digests are lazy.** A stub carries a deterministic digest immediately;
   an LLM summary upgrade runs in the background and later re-compactions
   pick it up. Nothing on the request path waits for an LLM.
4. **Stubbed pages point at searchable copies.** Saved pages carry the
   session's internal search scope (via the assistant message's file
   descriptors), so the model retrieves details instead of re-fetching.
5. **The response waits for the searchable copy.** `open_url` holds its
   answer (bounded by `OPEN_URL_INDEX_WAIT_SECONDS`, 2s polls) until the
   freshly saved pages reach a terminal indexing state; the pointer note
   only promises `internal_search` once the file is really searchable
   (`IndexedPageRef.search_ready`, stamped from the final status).

## Main sequence: fetch, index, answer, search later

```plantuml
@startuml
autonumber
skinparam responseMessageBelowArrow true

actor User
participant "ChatBackend" as CB
participant "LLMLoop" as LOOP
participant "HistoryBuilder" as HIST
participant "DigestStore" as DIG
participant "OpenURL tool" as OPEN
participant "DocIndexing" as IDX
database "FileStore + Postgres" as DB
queue "user-file Celery" as CEL
database "OpenSearch" as OS
participant "Model server\n(llama.cpp)" as LLM

== Turn 1: fetch + index + answer ==

User -> CB: analyze these 3 URLs
CB -> LOOP: run turn (available = 243,937 tok)
LOOP -> LLM: cycle 0 request
note right of LOOP
  "Cycle 0 budgets: available=... tools=...
   cycle_history=... request=... output_cap=..."
end note
LLM --> LOOP: open_url([u1, u2, u3])
LOOP -> OPEN: run(urls)
OPEN -> OPEN: crawl via SearXNG\n(caps: 180k/URL, 540k/batch)
OPEN -> IDX: index_crawled_pages(sections)
IDX -> IDX: skip incognito, skip < 6k chars,\ndedup by normalized link_url
IDX -> DB: save markdown blob + UserFile(PROCESSING, link_url)
DB -> CEL: PROCESS_SINGLE_USER_FILE
IDX -> IDX: hold: poll user_file statuses until\nterminal (budget OPEN_URL_INDEX_WAIT_SECONDS)
IDX -> IDX: stamp refs search_ready =\n(final status == COMPLETED)
IDX --> OPEN: [IndexedPageRef(url, user_file_id, name,\nsearch_ready)]
OPEN -> OPEN: llm JSON: indexed pages inline\n4k-char excerpt + pointer note:
note right of OPEN
  search_ready=true:
  "[... use internal_search for details ...]"
  search_ready=false (hold timed out or
  indexing failed):
  "[... still indexing; internal_search
  will find it shortly ...]"
end note
OPEN --> LOOP: ToolResponse(indexed_files, bytes/tokens, doc_stats)
LOOP -> LOOP: _extend_search_scope_with_indexed_files:\nchat-scope += this turn's user_file_ids\n(in-turn internal_search sees them now)
LOOP -> LOOP: indexed_file_notes[tool_call_id] =\n"search instead of refetch"
LOOP -> HIST: construct(cycle budget, digest cache, notes)
HIST -> HIST: newest exchanges stay raw\n(request fits -> no compaction)
LOOP -> LLM: cycle 1 request (excerpts raw)
LLM --> LOOP: answer + citations
LOOP -> CB: ToolCallInfo(indexed_files)
CB -> DB: assistant_message.files +=\n{id, type: plain_text, user_file_id}\n(dedup by user_file_id within the batch)
CEL -> DB: load blob, chunk, embed
CEL -> OS: index chunks (document_id = user_file_id)

== Turn 2: retrieve from the saved copies ==

User -> CB: detail question, no new URL
CB -> DB: _collect_available_file_ids(chat_history)
note right of CB
  reads files JSONB of the session's
  messages; user_file_id entries scope
  internal search to this chat
end note
note over CB, LOOP
  the in-turn scope extension already added
  these ids during turn 1, so searches work
  before the JSONB save lands too
end note
CB -> OS: internal_search (session-scoped)
OS --> CB: hits from the saved full text\n(behind the 4k excerpt)
CB -> LLM: answer quotes mid-article detail

== Pressure path: stub, then upgrade ==

User -> CB: more turns; history grows
CB -> LOOP: run turn
LOOP -> HIST: construct(budget shrinks as cycles are consumed)
HIST -> HIST: raw suffix no longer fits\n-> oldest exchanges stub oldest-first
HIST -> DIG: stub(tool_call_id) -> cache miss
DIG -> DIG: deterministic digest now\n(store + per-turn cache)
DIG --> HIST: stub = digest + "search instead\nof refetch" note
HIST -> DIG: schedule LLM summary upgrade (background)
DIG -> LLM: summarize original payload (never blocks a request)
DIG -> DB: store updated with summary\n(bounded 1024-entry, process-wide)
note over HIST, DIG
  every cycle re-compacts from the full history:
  the next request automatically replays the
  upgraded summary instead of the extractive digest
end note
HIST --> LOOP: history fits; newest exchange still raw
LOOP -> LLM: continue turn
@enduml
```

## Compaction decision (both in-turn tail and cross-turn history)

```plantuml
@startuml
start
:total tokens <= budget?;
if (yes) then
  :replay verbatim; stop;
else (no)
  :group messages into exchanges\n(USER/ASSISTANT starts an exchange;\nresponses attach to it);
  :keep the newest run of exchanges\nthat fits raw;
  :older exchanges: stub tool responses\n(digest + indexed-file note);
  if (stubbed prefix fits?) then
    :keep all stubbed exchanges;
  else (no)
    :drop oldest stubbed exchanges\nuntil the remainder fits;
  endif
  :log "kept N raw, stubbed M, dropped K";
endif
stop
@enduml
```

## Where the code lives

| Piece | Location |
| --- | --- |
| Cycle budget + ledger, compaction, history fit | `backend/onyx/chat/llm_loop.py` |
| Mid-turn search-scope extension (`extend_search_scope_with_indexed_files`, shared with DR sub-agents) | `backend/onyx/chat/llm_loop.py` |
| Digest cache/store, lazy tier-2 upgrade | `backend/onyx/chat/tool_response_digest.py` |
| Crawled-page indexing + hold-until-searchable | `backend/onyx/tools/tool_implementations/open_url/doc_indexing.py` (`_wait_until_searchable`) |
| Excerpt, pointer note gating, `indexed_as` | `backend/onyx/tools/tool_implementations/open_url/open_url_tool.py` (`_inline_content_for_section`) |
| Assistant-message file descriptors (batch dedup) | `backend/onyx/chat/save_chat.py` (step 9) |
| Ratios and caps | `backend/onyx/llm/context_budgets.py`, `backend/onyx/configs/app_configs.py` |
| DR orchestrator / final report (digests wired through) | `backend/onyx/deep_research/dr_loop.py` |
| DR research sub-agent (digests, indexed-file notes, run-wide scope extension) | `backend/onyx/tools/fake_tools/research_agent.py` |

## Deep Research under the same rules

The DR orchestrator and its research sub-agents run their own cycles but
build every request through the same `construct_message_history`, so the
compaction machinery applies there too. Wired and live-verified
2026-11-04 (run with a temporary 16k sub-agent cap: 4 compaction rounds,
tier-1 digests on stubs, tier-2 upgrades attempting with deterministic
fallback):

- The orchestrator keeps one digest cache per run; the final report and
  every orchestrator cycle replay digested sub-agent reports instead of
  bare notices when the history overflows. A 48h run (the deployment's
  `DR_FORCE_REPORT_S`) overflows many times — digests are what survive.
- Sub-agents keep per-call digest + note caches; their own overflow
  stubs carry digests and the indexed-file note.
- Pages a sub-agent saves via `open_url` extend the run-wide search
  scope immediately, so later sub-agents retrieve them instead of
  re-fetching the same URLs.

Also live-verified 2026-11-04: thinking models can end the intermediate
report step with an empty answer at `finish_reason=stop` (the report
lands in the reasoning trace, sometimes with further searches planned
that the tool-less step cannot run). The retry now appends an explicit
report nudge (`RESEARCH_REPORT_RETRY_NUDGE`) alongside the doubled
generation budget; run 2 wrote both intermediate reports and a cited
final answer without triggering it. Regression test:
`backend/tests/unit/onyx/deep_research/test_dr_compaction_wiring.py`.

Known limitations:
- Tier-2 digest upgrades run on the primary engine and race the DR loop
  for it; under load they time out into the deterministic fallback (by
  design). Routing them to the secondary model is the follow-up.
- A compacted sub-agent report can lose its citation markers, so the
  final report may under-cite documents that stayed in
  `final_documents`. The citation ledger idea (marker -> doc + one-line
  claim, compacted last) is the follow-up if live runs show
  under-citing.

## Live values (this deployment)

| Knob | Value |
| --- | --- |
| Input budget / safety margin | 243,937 tok / 0.05 |
| `MAX_LLM_CYCLES` | 12 (22 for scheduled runs) |
| Fetch caps | 180,000 chars/URL, 540,000/batch |
| Index threshold / inline excerpt | 6,000 / 4,000 chars |
| Hold-until-searchable budget | `OPEN_URL_INDEX_WAIT_SECONDS` 90 (2s polls; 0 disables) |
| `MAX_CHUNKS_FED_TO_CHAT` | 10 (bounds the selection prompt and the search feed; at 512-tok chunks 50 blew the 9B-era selection prefill past every budget) |
| `SECONDARY_LLM_FLOW_TIMEOUT_S` | 600 (secondary flows run on the GPU pair; thinking off via provider `extra_body`) |
| `TOOL_EXECUTION_TIMEOUT_SECONDS` | 1200 — the search tool's whole batch must fit inside this |
| Digest stub cap / summary output | 2,400 chars / 11/1250 of window (~2,259 tok) |
| Summary input cap | 400,000 chars (head+tail kept) |
| Cross-turn digest store | 1,024 entries, in-process |
| In-flight digest upgrades | max 4 |
