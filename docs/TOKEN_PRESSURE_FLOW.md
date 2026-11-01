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
IDX --> OPEN: [IndexedPageRef(url, user_file_id, name)]
OPEN -> OPEN: llm JSON: indexed pages inline\n4k-char excerpt + "indexed_as" name
OPEN --> LOOP: ToolResponse(indexed_files, bytes/tokens, doc_stats)
LOOP -> LOOP: indexed_file_notes[tool_call_id] =\n"search instead of refetch"
LOOP -> HIST: construct(cycle budget, digest cache, notes)
HIST -> HIST: newest exchanges stay raw\n(request fits -> no compaction)
LOOP -> LLM: cycle 1 request (excerpts raw)
LLM --> LOOP: answer + citations
LOOP -> CB: ToolCallInfo(indexed_files)
CB -> DB: assistant_message.files +=\n{id, type: plain_text, user_file_id}
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
| Digest cache/store, lazy tier-2 upgrade | `backend/onyx/chat/tool_response_digest.py` |
| Crawled-page indexing | `backend/onyx/tools/tool_implementations/open_url/doc_indexing.py` |
| Excerpt + `indexed_as` in the LLM JSON | `backend/onyx/tools/tool_implementations/open_url/open_url_tool.py` |
| Assistant-message file descriptors | `backend/onyx/chat/save_chat.py` (step 9) |
| Ratios and caps | `backend/onyx/llm/context_budgets.py`, `backend/onyx/configs/app_configs.py` |

## Live values (this deployment)

| Knob | Value |
| --- | --- |
| Input budget / safety margin | 243,937 tok / 0.05 |
| `MAX_LLM_CYCLES` | 12 (22 for scheduled runs) |
| Fetch caps | 180,000 chars/URL, 540,000/batch |
| Index threshold / inline excerpt | 6,000 / 4,000 chars |
| Digest stub cap / summary output | 2,400 chars / 11/1250 of window (~2,259 tok) |
| Summary input cap | 400,000 chars (head+tail kept) |
| Cross-turn digest store | 1,024 entries, in-process |
| In-flight digest upgrades | max 4 |
