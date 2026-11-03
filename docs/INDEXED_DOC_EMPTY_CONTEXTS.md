# Investigation: empty `chunk_context` / `doc_summary` on indexed documents

Status: open — needs root-cause + fix.
Observed: 2026-11-02/03 during the phase-5 live E2E (indexed crawled pages).

## Symptom

Chunks of user-file documents carry **empty** enrichment fields in both
migration indices — contextual retrieval therefore runs on raw chunk
content only, and the contextual-RAG quality lever is dead:

| Doc | Index | `chunk_context` | `doc_summary` |
| --- | --- | --- | --- |
| Run-8 Ayutthaya page (502be09c, 135 chunks, ported overnight) | PRESENT | `''` | `''` |
| Run-9 pages (58ab12b5 / f0727c47 / a37ee91d, forward-indexed) | FUTURE | `''` | `''` |

Verified live 2026-11-03 via direct OpenSearch queries.

## Impact

- Search still *works*: both E2E retrieval checks (turn-2 style questions)
  retrieved the exact mid-article passages from un-enriched chunks.
  Specific-fact queries match on raw content fine.
- What is lost: the contextual-retrieval quality benefit for ambiguous
  queries (chunk-level document/section context is what rescues vague
  hits), and anything downstream that reads `doc_summary`.

## Facts collected

1. `USE_CHUNK_SUMMARY` and `USE_DOCUMENT_SUMMARY` both default `true`
   (`app_configs.py`) and are **not** overridden in the deployment env.
2. Run-9 user files (180k-char markdown, ~135 chunks): created 06:52:54,
   status `COMPLETED` by 06:57:41 — **4.8 minutes total**. The user-file
   worker log (`celery_worker_user_file_processing.log`) shows only
   `embed_chunks` events in that window (20.4s for 135 texts) and **no LLM
   summarization calls**. At ~8-13s per chunk-context call, 135 calls
   cannot have happened.
3. The migration port (attempt 52, 02:24→06:17 Nov 3, 10 docs ported —
   search_settings 8, `cc_pair_id` NULL) ran its slow enrichment phase
   overnight — yet the sampled ported chunks still show empty fields.
4. `assign_context` swallows **all** errors into `chunk_context=""`
   (silent failure path, no log/metric), and `add_document_summaries`
   failures degrade to empty enrichment (post-2026-11-02 guards). An
   empty field is therefore indistinguishable from "not attempted".
5. The secondary/flow LLM carries the provider's thinking-off
   `extra_body` (`chat_template_kwargs {"enable_thinking": false}`) only
   through the `custom_config` mapping — any enrichment path that builds
   its LLM object differently does not get it. Without that flag, the
   thinking model burns the output cap on a think trace and returns
   empty content (verified by A/B against both engines).

## Hypotheses (ordered)

1. **Enrichment never runs for forward-indexed user files.** The 4.8-min
   COMPLETED time and the absence of LLM calls in the worker log suggest
   `add_chunk_summaries` / `add_document_summaries` are not invoked on
   the user-file processing path at all (port-only wiring), or are gated
   behind a flag that reads false in the worker.
2. **Thinking-model empty content inside enrichment.** If enrichment
   *does* run somewhere, its LLM calls may bypass the provider
   `extra_body` (thinking on) → empty content → swallowed to `''`. Check
   whether the contextual-RAG LLM object is built via
   `map_custom_config_to_model_kwargs` (extra_body carried) or bare.
3. **Env/worker propagation gap.** `USE_CHUNK_SUMMARY` /
   `USE_DOCUMENT_SUMMARY` are read at import time; confirm the
   `user_file_processing` worker and the port worker see them as true.
4. **Port enrichment coverage.** Attempt 52 ported 10 docs with the slow
   phase, yet sampled ported chunks are empty — verify which docs the 10
   were and whether enrichment output was written to the FUTURE index or
   lost (create-only writes + sweep semantics).

## How to reproduce / verify

1. Create a small user file (well under the 21.5k full-doc-inclusion
   threshold), wait for `COMPLETED`, then read a FUTURE-index chunk:
   `chunk_context` / `doc_summary` are `''`.
2. Run the enrichment unit for that doc manually with
   `LOG_ONYX_MODEL_INTERACTIONS=true` and watch whether an LLM call fires
   (and whether the request body contains `chat_template_kwargs`).
3. Inspect the contextual-RAG LLM object on the enrichment path:
   `llm._model_kwargs["extra_body"]` should contain the flag.
4. After any fix, the same chunk fields must come back non-empty, and
   `assign_context` should log a warning (not stay silent) when a call
   yields empty context.

## Fix direction

- Make enrichment failures **loud**: log + counter when a context comes
  back empty (keep the graceful degrade, add the signal).
- Ensure the enrichment LLM inherits the provider `extra_body` (thinking
  off) regardless of construction path.
- Decide deliberately whether forward-indexed user files should get
  enrichment inline (accepting the added indexing latency — bounded by
  `CONTEXTUAL_RAG_LLM_TIMEOUT`, serial workers + KV reuse already tuned)
  or via a follow-up pass; then close the wiring gap.
- Re-enrich affected docs afterwards (port re-run or per-doc re-upsert);
  the already-indexed crawled pages stay un-enriched until then.

## Related

- `docs/TOKEN_PRESSURE_FLOW.md` — hold-until-searchable, live values.
- `AGENTS.md` KEY NOTES — secondary-model wiring, thinking-off
  enforcement, and the model-server KV-reuse notes.
