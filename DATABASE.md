# Database map

What's in the database, table by table, and which part of the app reads or writes each one. Matches
`schema.sql` exactly - if this ever drifts, `schema.sql` is the source of truth.

## The shape of it

There are three groups of tables:

1. **Past repairs** (`pcr_cases` + `pcr_case_summaries`) - real repair tickets.
2. **Service manual** (`manual_chunks` + `manual_chunk_summaries`) - official diagnostic procedures.
3. **Diagnosis flow** (`diagnosis_trees`, `diagnoses`, `diagnosis_outcomes`) - what the app built and what happened with it.

Groups 1 and 2 share the same pattern on purpose: a **content table** (the full real text, never
searched directly) and a **search table** (a short summary + its embedding, which retrieval actually
queries, then joins back to the content table for the full text). This split exists because the full
case/manual text is too long and noisy to embed well directly - a short, focused summary embeds and
matches far more accurately. See `app/retrieval.py` for how the join happens.

A fourth group (`model_prices`, `llm_calls`, `run_costs`) tracks what every LLM call cost - unrelated to
diagnosis logic, used only by `app/usage.py` and the `/diagnoses/{id}/cost` endpoint.

```
pcr_cases ----------------< pcr_case_summaries (embedding)
                                                              \
                                                               >-- app/retrieval.py joins these for evidence
                                                              /
manual_chunks -------------< manual_chunk_summaries (embedding)

diagnosis_trees ---< diagnoses ---< diagnosis_outcomes
     (cached tree)   (one run)      (mechanic's feedback)

llm_calls ---< run_costs   (run_id links back to a diagnoses.id)
model_prices  (static price table, used to cost llm_calls)
```

---

## 1. Past repairs

### `pcr_cases` - the content store
One row per real repair ticket (PCR). Never queried directly by embedding - `app/retrieval.py` always
goes through `pcr_case_summaries` first, then joins back here.

| Column | What it holds |
|---|---|
| `ticket_no` | The real PCR ticket number (unique). |
| `model_line`, `model_variant`, `fuel` | Vehicle identification. |
| `dtc_code` | The fault code this ticket was filed against. |
| `customer_verbatim` | The original complaint, as reported. |
| `comments_raw` | The full back-and-forth thread (kept for display/citation only, not sent to the LLM). |
| `conclusion_text` | The last substantive comment in the thread - this is the actual diagnosis + resolution, and is what gets sent to the LLM as "resolution" text. |
| `failed_parts` | JSON array of `{"part_code", "part_name"}` - the parts actually replaced. |
| `conclusion_matches_parts` | Data-quality flag: does `conclusion_text` agree with `failed_parts`? Set during ingestion, not read by the running app. |
| `resolution_status` | `'resolved'` or otherwise; retrieval only pulls `resolved` cases. |

Written by: `scripts/ingest_pcr.py`. Read by: `app/retrieval.py` (joined from `pcr_case_summaries`).

### `pcr_case_summaries` - the search index
One row per case (today), holding a short summary written for embedding search plus the embedding
itself. This is the table `ORDER BY embedding <=>` actually runs against.

| Column | What it holds |
|---|---|
| `pcr_case_id` | FK to `pcr_cases`. |
| `summary_for_search` | A short summary + likely mechanic questions, written to embed well. |
| `embedding` | `vector(1536)`, from `text-embedding-3-small`. |

Written by: `scripts/ingest_pcr.py`. Read by: `app/retrieval.py` (`semantic_evidence`, both the code-given and no-code branches - `idx_pcr_summary_embedding` is the HNSW index it hits).

---

## 2. Service manual

### `manual_chunks` - the content store
One row per manual section (one table-of-contents entry = one shared diagnostic procedure). A single
section can cover several DTC codes at once (`dtc_codes` is an array); the same code appearing under a
different submodule gets its own separate row, since that content genuinely differs.

| Column | What it holds |
|---|---|
| `dtc_codes` | `TEXT[]` - every fault code this section's procedure covers. |
| `model_line`, `submodule` | Vehicle + the system/module this belongs to (e.g. "VCU DTC List"). |
| `start_page`, `end_page` | Source PDF page range, for traceability. |
| `full_markdown` | The complete verbatim section: codes, description, trigger conditions, the diagnostic procedure, common problems, etc. This is what gets sent to the LLM. |

Written by: `scripts/ingest_manual.py` / `scripts/load_manual_markdown_only.py`. Read by: `app/retrieval.py` and `app/manual.py` (joined from `manual_chunk_summaries`, or looked up directly by `dtc_codes` for exact-match and catalog lookups).

### `manual_chunk_summaries` - the search index
Same split as the case tables: a short summary + embedding per manual chunk, kept in its own table so a
chunk could later have more than one summary variant without touching the content store.

| Column | What it holds |
|---|---|
| `manual_chunk_id` | FK to `manual_chunks`. |
| `summary_for_search` | Summary + likely user questions for this section. |
| `embedding` | `vector(1536)`. |

Written by: the same ingestion scripts. Read by: `app/retrieval.py` (`idx_manual_embedding` HNSW index).

---

## 3. Diagnosis flow

### `diagnosis_trees` - the cached tree
One row per `(dtc_code, model_line, symptom_hash)` combination. The expensive part (the LLM call that
builds the whole decision tree) only happens once per combination - every later request with the same
code/model/symptom hits this table and reuses the tree instantly, no LLM call.

| Column | What it holds |
|---|---|
| `dtc_code`, `model_line` | `''` for `dtc_code` when no code was given (symptom-only query). |
| `symptom_hash` | `''` for the catalog (code-only) flow; otherwise `sha256` of the lowercased, trimmed free-text query. This is the actual cache key alongside code+model. |
| `tree` | The full `DiagnosisTree` JSON - `root` node id + every node (branches and leaves). This is exactly what gets sent to the browser; the whole tree is walked client-side with no further server round-trip. |
| `evidence` | A condensed summary of what was retrieved (case count, top parts, manual status) - powers the "What this is based on" panel. The full case/manual text used to build the tree is *not* stored here; see `diagnosis_trace.log` (not in the DB) for that, captured only for the most recent build. |

**Known gap:** this cache key has no prompt/schema version in it. Changing `TREE_SYSTEM` in
`app/diagnosis.py` does not invalidate old cached trees - they keep being served as-is until manually
cleared:
```sql
UPDATE diagnoses SET tree_id = NULL;
DELETE FROM diagnosis_trees;
```

Written and read by: `app/diagnosis.py` (`_get_or_build_tree`).

### `diagnoses` - one guided-diagnosis run
A thin anchor row per diagnosis session. It does **not** track which questions were answered or which
node the mechanic is on - that state lives entirely in the browser (`app/static/app.js`), which is why
`GET /diagnoses/{id}` (resume) always restarts at the tree's root unless the browser's own local storage
still has the exact in-progress state.

| Column | What it holds |
|---|---|
| `id` | UUID, generated per diagnosis, shown to the mechanic at the end. |
| `dtc_code`, `model_line` | Copied from the request. |
| `tree_id` | FK to `diagnosis_trees` - which cached tree this run is walking. |
| `status` | `'asking'`, `'concluded'`, or `'no_evidence'` - the status *at creation time* only; not updated as the mechanic answers questions. |

Written and read by: `app/diagnosis.py` (`start_diagnosis`, `get_diagnosis`).

### `diagnosis_outcomes` - mechanic's feedback
One row per diagnosis (1:1, upserted), recorded after the repair via the "After the repair: did it work?"
form. This is the ground-truth data that would power a real accuracy measurement, if one were built.

| Column | What it holds |
|---|---|
| `diagnosis_id` | PK, FK to `diagnoses`. |
| `result` | `'worked'`, `'partly'`, or `'not_worked'`. |
| `actual_cause`, `parts_replaced`, `notes` | What the mechanic says actually happened, if different from the AI's conclusion. |

Written by: `app/diagnosis.py` (`save_outcome`). Read by: `outcome_summary()` (the `/outcomes/summary`
endpoint - currently just a result-count breakdown, nothing per-code or per-tree yet).

---

## 4. Cost tracking (unrelated to diagnosis quality)

### `model_prices`
Static price-per-million-tokens table, seeded directly in `schema.sql`. One row per model actually used
(`gpt-6-luna`, `gpt-6-sol`, `text-embedding-3-small`), sourced from the OpenRouter catalog. Update this
table's rows (not the code) when prices change.

### `llm_calls`
One row per individual LLM call (every `embed` and every `structured` call), tagged with `run_id` (a
`diagnoses.id`) and `purpose` (`embed`, `tree_build`). Written by `app/usage.py`'s `UsageRecorder`,
called from `app/llm.py` after every call. Read by `GET /diagnoses/{id}/cost`.

### `run_costs`
One row per `run_id`, pre-aggregated totals (calls, tokens, cost) so the cost endpoint doesn't have to
sum `llm_calls` on every request. Kept in sync by the same `UsageRecorder` that writes `llm_calls`.

---

## Quick lookups

- **"Where's the actual case text that went into a diagnosis?"** Not stored per-diagnosis - only the
  condensed `diagnosis_trees.evidence` summary persists. The full text is only ever visible in
  `diagnosis_trace.log` (project root, overwritten on each build - see `app/trace.py`).
- **"Why does the same query sometimes skip the LLM entirely?"** `diagnosis_trees` cache hit - same
  `(dtc_code, model_line, symptom_hash)` seen before.
- **"Where do I clear stale trees after a prompt change?"** See the `diagnosis_trees` section above.
- **"Which table has the DTC-to-manual-page mapping?"** `manual_chunks.dtc_codes` (a Postgres array,
  GIN-indexed - `idx_manual_dtc_model`).
