# KG Localization + Repair Pipeline — Complete Description

Best-performing harness: **KG exploration (file_level, embedding-augmented) + single-call list-wise LLM rerank + SEARCH/REPLACE repair**.

Headline test results (SWE-bench Lite, n=299): **R@1 74.2 / R@3 84.6 / R@5 88.0**, pool ceiling 90.3
(vs Agentless 65.6/84.3/87.3, BM25 k=5 42.1/56.2/66.2).
End-to-end dev (local gemma, sr repair on top-5 context): **5/23 resolved (21.7%)**.

Code references are relative to this repository. Entry point: `python -m kg.run <config> --embeddings`
(`src/kg/run.py:285`), which runs Stages 0–4 per instance; Stage 5 is `scripts/llm-rerank-files.py`;
Stage 6 is `scripts/compare-file-localization.py`; Stage 7 is `scripts/build-rerank-dataset.py`;
Stage 8 is `src/entrypoints/run_repair_inference.py`; Stage 9 is the standard SWE-bench harness.

---

## Diagram

```mermaid
flowchart TB
    subgraph S0["Stage 0 — repo acquisition (per repo@commit, cached)"]
        A[SWE-bench instance<br/>repo + base_commit + problem_statement] --> B[git clone from mirror<br/>git clean -fd, checkout --force]
    end

    subgraph S1["Stage 1 — knowledge-graph build (static AST)"]
        B --> C[code_kg_builder.Orchestrator<br/>filesystem scan → AST → edges]
        C --> D[("KG data.json<br/>11 node types, 9 edge types<br/>CALLS / CONTAINS / INHERITS / ...")]
    end

    subgraph S2["Stage 2 — embedding index over nodes"]
        D --> E[node text composition<br/>type + name + path + docstring + source[:2048]]
        E --> F["nomic-embed-code-Q8_0<br/>(search_code_document: prefix)"]
        F --> G[("per-graph Chroma (cosine)<br/>+ cross-graph blob-SHA cache<br/>~74k vectors")]
    end

    subgraph S3["Stage 3 — agentic exploration loop"]
        H[problem statement] -->|"embed → top-15 seed nodes"| I["chat LLM loop (≤20 rounds, ≤50 calls)<br/>search_elements · search_by_embedding ·<br/>get_neighbors · traverse · get_element · find_connected"]
        D --> I
        I -->|"every node seen in tool results"| J[touched_nodes<br/>ordered candidate pool]
    end

    subgraph S4["Stage 4 — context assembly (file_level)"]
        J --> K["nodes → .py files via CONTAINS<br/>files ranked by touch frequency"]
        B --> L[repo files at base_commit]
        K --> L
        L --> M["style-3 prompt<br/>READMEs + full contents of ALL touched files<br/>in [start of path] ... [end of path] blocks"]
    end

    subgraph S5["Stage 5 — list-wise LLM rerank (1 call)"]
        M --> N["per-file signals<br/>touch_freq · sim_max · sim_mean(top-3) · parts(≤3 nodes)"]
        G -->|"issue ↔ node cosine sims"| N
        N --> O["chat LLM: strict JSON array of top-5 paths<br/>(chunks if >100k chars · 3 retries)"]
        O --> P[ranked_files<br/>= LLM top-5 + frequency tail]
    end

    P --> Q(("Stage 6 — localization scoring<br/>Recall@1/3/5 · Cov@5 · pool ceiling<br/>vs gold-patch files"))
    P --> R["Stage 7 — context rebuild<br/>top-5 ranked files, verbatim blocks<br/>(no LLM)"]
    R --> S["Stage 8 — SEARCH/REPLACE repair<br/>1 chat call (t=0, 8192 max tokens)<br/>unique-match edits on repo@base_commit"]
    S --> T["difflib unified diff<br/>model_patch"]
    T --> U(("Stage 9 — SWE-bench harness<br/>FAIL_TO_PASS + PASS_TO_PASS<br/>resolved rate"))
```

ASCII overview (arrow = data flow, cache = reused across instances):

```
instance ──> [0 clone/checkout] ──> [1 KG build] ──> data.json
                   │                    │
                   │                    ├──> [2 embed index] ──> Chroma (+blob cache)
                   │                    │
 problem_statement ┼──────────┐         │
                   │          ▼         ▼
                   │    [3 LLM explore loop] ──> touched_nodes (pool, ordered)
                   │          │
                   ▼          ▼
             repo files   [4 pool→files, freq rank, style-3 prompt] ──> text_inputs
                              │
                              ▼
                  [5 list rerank: signals + 1 LLM call] ──> ranked_files
                       │                         │
                       ▼                         ▼
           [6 score vs gold patch files]   [7 rebuild top-5 context] ──> rerank5 dataset
                                                             │
                                                             ▼
                                              [8 sr repair: 1 LLM call] ──> model_patch
                                                             │
                                                             ▼
                                                  [9 SWE-bench evaluation]
```

---

## Serving infrastructure

One llama-server on port 8081 hosting both models (OpenAI-compatible `/v1`):

| Role | Model | Context | Notes |
|---|---|---|---|
| Chat (exploration + rerank) | `qwen3-coder-next-Q8_0` | 131072 | temp 0.0 by default |
| Embeddings | `nomic-embed-code-Q8_0` | 32768 | 3584-dim vectors |

Both Stages 3 and 5 talk to the same endpoint; embedding queries go to the embeddings model
(config `embedding_api_base`, production runs pin both to :8081).

---

## Stage 0 — Repo acquisition (cached)

`src/kg/build_graphs.py:16` (`ensure_repo_cloned`), `:31` (`_checkout`).

**Input**: instance fields `repo`, `base_commit`.
**Processing**: clone from `https://github.com/swe-bench-repos/{repo}.git` into
`outputs/kg_repos/{repo__org}` once per repo; for each new commit
`git clean -fd` + `git checkout --force base_commit`.
**Output**: a clean source tree at the exact commit under test.
**Caching**: the *graph* cache key is `{repo}__{base_commit}` under `outputs/kg_graphs/`
(`build_graph` at `build_graphs.py:59` returns immediately when a populated `data.json` exists) —
instances sharing a commit share one graph (~28 issues per repository in SWE-bench Lite, which is
the construction-amortization argument).

## Stage 1 — Knowledge-graph build (static analysis)

`code_kg_builder.Orchestrator` invoked from `build_graph()`.

**Input**: the checked-out Python repository.
**Processing**: executor pipeline in fixed order — filesystem scan first, then AST structure
extraction, then edge resolution; every edge batch is validated against meta-schema
(source_type, target_type) signatures before commit (`orchestrator.py:194`), invalid edges are
counted and dropped, never guessed.
**Graph contents** (schema `swe_bench_kg/meta.json`):

- 11 node types: `Class, Method, Function, Variable, Import, Decorator, File, Directory,
  Exception, TypeAnnotation, Dependency`
- 9 edge types: `CONTAINS` (Directory→File, Class→Method, …), `CALLS`, `INHERITS`, `IMPORTS`,
  `DECORATES`, `RAISES`, `CATCHES`, `HAS_TYPE`, `DEPENDS_ON`
- node properties: `meta_type, name, docstring, source_code, line_start, line_end, complexity,
  relative_path`

**Output**: flat JSON graph `data.json = {nodes, edges, allValues}` (properties live in
`allValues`, keyed by node id) plus schema files `origin.json` / `meta.json`.
Purely static: no LLM, no execution — deterministic and reproducible per commit.

## Stage 2 — Embedding index over graph nodes

`src/kg/embeddings.py:432` (`build_embeddings`).

**Input**: `data.json` + (for the cache) `git ls-tree` blob SHAs of the checkout.
**Text composition** (`compose_node_text`, `embeddings.py:106`): for code nodes
`"{Function|Method|Class} {name} in {relative_path}"` + docstring + `source_code[:2048]`;
Files → `"File {path}"`; Variables/Imports/Decorators get short templates; node types with no
standalone semantics are skipped (`Exception, TypeAnnotation, Dependency, Directory`).
Task prefixes are applied consistently (nomic-embed-code contract): documents get
`search_code_document: `, queries get `search_code_query: `. Documents are capped at 8192 chars;
requests batched at ≤32 items and ≤30k estimated tokens.
**Vector store**: per-graph Chroma PersistentClient at `{graph_dir}/chroma_db`, collection
`kg_nodes`, cosine distance; metadata carries `meta_type, name, label, relative_path`; a
completion stamp (`embedding_model`, `node_count`) makes interrupted builds self-healing.
**Cross-graph cache** (`embedding_cache_dir`, ~74k vectors): key =
`sha256(fingerprint \0 model \0 blob \0 {git-blob-SHA} \0 {node_id})` with a document-hash
fallback — a file whose content is unchanged across SWE-bench commits is never re-embedded,
so only changed files cost embedding work per new commit.
**Output**: a semantic index queryable by node similarity.

## Stage 3 — Agentic exploration loop

`src/kg/explore.py:281`.

**Input**: `problem_statement` + the graph (+ embedding index).
**Setup**: system prompt = extracted graph schema + fault-localization workflow + JSON-array
function-call format (`prompts.py:173`). Tools exposed (all read-only graph operations):
`search_elements` (keyword/label/property filter), `search_by_embedding` (semantic, embed-only),
`get_neighbors`, `traverse` (multi-hop, e.g. follow CALLS), `get_element` (fetch `source_code`),
`find_connected` (nearest node of a type — used to find a node's containing File via CONTAINS),
`get_function_info` (schema help).
**Head start** (embed variant): the issue is embedded once and the top-15 nodes by cosine
similarity are injected into the first user message (`get_top_k_for_issue`,
`embeddings.py:640`), so the loop starts from semantic anchors instead of a cold keyword search.
**Loop**: the LLM replies with a JSON array of tool calls → calls are parsed and executed →
results are rendered as a compact text block (`--- TOOL RESULTS ---`, ids + labels, first 20
items, `source_code` truncated to 500 chars) → appended as the next user message. Every node id
appearing in any successful result is accumulated into the ordered `touched` dict — this is the
candidate-pool generator.
**Budgets and robustness**: ≤20 rounds, ≤50 tool calls; base temperature 0.0, +0.15 per
consecutive rejected round capped at 0.9; three rotated rejection phrasings; one full
conversation reset after 3 consecutive rejections (grants 6 extra rounds); an idle nudge when
nothing was touched; a forced "critically review your findings" round before the model may
finish.
**Termination**: the model answers with text only (no tool calls) after the review round, or a
budget is exhausted.
**Output**: `touched_nodes` — node ids in exploration order, plus any properties seen along the
way. This defines the localization pool (and hence the pool ceiling: 90.3% on test).

## Stage 4 — Context assembly (file_level)

`src/kg/context.py:262` (`assemble_file_level`), `_rank_files` at `:22`, `read_all_files` at
`:100`.

**Input**: `touched_nodes`, graph, repo checkout at `base_commit`, `problem_statement`.
**Node → file mapping**: for each touched node, `find_connected(CONTAINS → File)` walks up to
the nearest `File` ancestor (handles Method→Class→File hops); only `.py` files count. Each
touched node is one vote for its file → **files ranked by touch frequency, descending**.
**Content**: ALL ranked files are read in full (no token-budget cut — the budget cutoff was
deliberately removed so low-frequency files, often the single-touch gold file, reach the
reranker) plus root READMEs.
**Prompt**: standard SWE-bench **style-3** layout — premise, `<issue>` text, `<code>` block with
`[start of <path>] … [end of <path>]` file sections in frequency order, patch example, final
instruction.
**Output**: progress-JSONL row `{instance_id, repo, base_commit, problem_statement, patch,
text_inputs, touched_nodes, …}` (append-only, resumable; rows with empty `text_inputs` are
re-processed on resume, `run.py:64`).

## Stage 5 — List-wise LLM rerank (the ranking stage)

`scripts/llm-rerank-files.py`, `--mode list --signals freq,sim_max,sim_mean,parts --top-k 5`.

**Input**: the progress row; graph dir; Chroma index.
**Candidates**: files parsed from the `[start of <path>]` markers of `text_inputs`
(`extract_files_with_content`, READMEs skipped) — i.e. exactly the Stage-4 pool, in exploration
order.
**Per-file signal assembly** (`assemble_file_signals`, line 334):

| Signal | Definition |
|---|---|
| `touch_freq` | number of touched graph nodes whose containing file is this file (Stage-3/4 votes) |
| `sim_max` | max cosine similarity (1 − distance) between the issue embedding and any touched node of the file |
| `sim_mean` | mean of the top-3 node similarities of the file |
| `parts` | ≤3 touched nodes in exploration order: `meta_type name (Lx–Ly) "first docstring line"` + first 120 chars of source (newlines → `\|`) |

Node similarities come from one Chroma query per instance (`get_top_k_for_issue`, k=500) mapping
node ids to similarities; files are listed to the LLM in touch-frequency order.

**Prompt** (`_LIST_PROMPT_TEMPLATE`, line 431): issue text + one block per candidate
(`### path` / `touch_freq: N | sim_max: … | sim_top3_mean: …` / part lines) + instruction:
*return ONLY the top-5 files, most relevant first, as a strict JSON array of paths*.
**Call** (`ListRanker`, line 514): single chat completion, max_tokens 2048, three attempts —
plain t=0.0 → disable-thinking t=0.0 → disable-thinking t=0.7 (thinking fallback:
`chat_template_kwargs.enable_thinking=false`; retry ladder shared with the pairwise mode).
Answer parsing accepts raw, fenced, or prose-embedded JSON arrays.
**Oversized pools**: if the rendered prompt exceeds 100k chars, candidates are split into greedy
frequency-ordered chunks that fit; each chunk is ranked by its own call; chunk winner lists are
merged round-robin and the frequency tail pads the rest (`_chunk_prompt_signals` line 556,
`rank_files_list` line 587). Large django instances → 2 calls.
**Validation**: paths outside the chunk's candidate set are dropped; the final
`ranked_files = LLM top-5 + all remaining candidates in frequency order` (the full pool stays
represented; scoring reads only the top-k). On total failure the frequency order is kept and the
row is marked `failed`.
**Output**: rerank progress JSONL (one self-describing row per instance: model, spec
mode/signals/top_k/subparts/src_chars/max_prompt_chars, ranked_files, n_files, n_calls).

## Stage 6 — Scoring (localization-only evaluation)

`scripts/compare-file-localization.py` (also `--eval-only` in the rerank script).

**Ground truth**: unique files matched by `(?m)^\+\+\+ b/(.+)$` in the gold `patch`.
**Metrics**: Recall@k (any GT file within the method's top-k), Cov@5 (share of GT files covered
by the top-5), pool ceiling = unbounded recall over ALL surfaced files (upper bound for any
rerank of this pool). Methods compared on the identical instance set: KG (freq order, no
rerank), KG+list rerank, Agentless `found_files`, BM25 `hits` (k=5, docid prefix stripped).

## Stage 7 — Context rebuild from ranked files (no LLM)

`scripts/build-rerank-dataset.py` (`parse_blocks` :47, `rebuild_context` :73).

**Input**: the KG embed progress JSONL (source of truth for prompts and file contents) + the
list-rerank progress JSONL of the same split.
**Processing**: pure `text_inputs` surgery — the `[start of path] … [end of path]` blocks are
re-assembled with the **top-k (5) ranked files in rerank order**; the header, `<issue>` block and
tail instructions stay verbatim. No LLM and no graph access.
**Output**: HF dataset dir `…__embed__rerank5__<model>` + progress JSONL, schema-identical to the
chain output — this is the repair-stage dataset.

## Stage 8 — Repair inference (SEARCH/REPLACE)

`src/entrypoints/run_repair_inference.py` (`--mode sr`, production); prompt/parsing/patch logic in
`src/kg/repair.py`.

**Input**: the rerank5 dataset row (style-3 prompt) + repo checkout at `base_commit`.
**Context guards** (`repair.py:54`): per file 60k chars (oversized file keeps head+tail with a
truncation note), total 360k chars (~90–100k tokens — fits n_ctx=131072 with the issue and the
completion), optional `--max_context_files` drops sections beyond the first N (a guard for rerank
fallbacks that kept the whole pool; first sections are the top-ranked files).
**Prompt** (`build_sr_prompt`, :278): premise + `<issue>` + `<code>` (line-number prefixes and
graph metadata stripped, `strip_line_numbers` :30) + an S&R example + instruction: for each file
to change, a `### path` header followed by one or more
`<<<<<<< SEARCH … ======= … >>>>>>> REPLACE` blocks.
**Call**: single chat completion, temperature 0.0, max_tokens 8192; system/user split on the
first newline. Thinking fallback (`generate_with_thinking_fallback`, entrypoint :103): if a
reasoning model (gemma) spends the budget thinking and returns empty content, exactly one retry
with `chat_template_kwargs.enable_thinking=false`.
**Edit application** (`parse_search_replace_blocks` :229, `apply_search_replace` :246,
`synthesize_sr_patch` :498): each SEARCH block must match **exactly one** location in the file —
not-found or ambiguous edits are skipped with a warning, never guessed. Edits are applied to the
working tree, then a unified diff is generated with `difflib` (`generate_unified_diff` :176,
`a/`/`b/` headers) — **the model never writes diff syntax**.
**Alternative modes**: `node` — graph-guided whole-node replacement (`### name in path` + complete
new code; node index from the graph via `build_node_index` :353, applied only when the stored
`source_code` still matches the file at `line_start..line_end`, `verify_and_replace_node` :421 —
guards against source drift) with S&R blocks as fallback for small edits; `tool` — multi-turn
agent repair (experimental, not included in this release). Production e2e used `sr`.
**Output**: `{model}__{dataset}__{mode}__{split}.jsonl` rows
`{instance_id, text, full_output, model_patch, thinking_retry}`; incremental writes, resume by
`instance_id`, instances processed shortest-context-first.

## Stage 9 — SWE-bench evaluation

Standard SWE-bench Lite harness: `model_patch` is applied in the per-instance container and judged
by the FAIL_TO_PASS + PASS_TO_PASS test suites; the metric is the resolved rate.
(Dev e2e, KG localization + sr repair, local gemma: 5/23 = 21.7%. `scripts/compare-runs.py`
aggregates several runs and their resolved-set overlap.)

---

## Cost profile

- Stage 1: no LLM (amortized per repo@commit). Stage 2: embeddings only, amortized per file
  content (blob cache). Stage 3: the only multi-call LLM stage (≤50 calls, typically ~15).
- Stage 5: exactly **1 LLM call per instance** (rarely 2 with chunking) — this is the
  cost-asymmetry argument vs pairwise comparator sorting (~250× fewer calls).
- Stage 7: no LLM (pure string surgery). Stage 8: exactly **1 LLM call** (max_tokens 8192; +1
  retry only when a thinking model returns empty content).
- Production pace on the old V100 server: ~6–7.5 min/instance for Stage 5 including signal
  assembly.

## Key configuration (production values)

| Parameter | Value | Where |
|---|---|---|
| chat model | `qwen3-coder-next-Q8_0` @ :8081 | config `model`, `api_base` |
| embedding model | `nomic-embed-code-Q8_0`, dim 3584 | `embedding_model`, `embedding_dim` |
| context mode | `file_level` | `context_mode` |
| explore budget | 20 rounds / 50 tool calls, t=0.0 | `max_rounds`, `max_tool_calls`, `temperature` |
| escalation | +0.15/rejection, cap 0.9, reset after 3 (+6 rounds) | `escalation_temp_*`, `rejection_restart_after` |
| head-start seeds | top-15 nodes | `embedding_top_k` |
| rerank mode | list, signals `freq,sim_max,sim_mean,parts`, top_k 5 | CLI `--mode/--signals/--top-k` |
| rerank detail | subparts 3, src_chars 120, max_prompt_chars 100000 | CLI defaults |
| rerank retries | 3 (t=0 → no-think t=0 → no-think t=0.7), max_tokens 2048 | `ListRanker` |
| repair context | top-5 reranked files, verbatim blocks | `build-rerank-dataset --top-k 5` |
| repair mode | `sr` (SEARCH/REPLACE), t=0.0, max_tokens 8192 | `run_repair_inference --mode sr` |
| repair context guards | 60k chars/file, 360k chars total | `_MAX_FILE_CHARS`, `_MAX_TOTAL_CHARS` |
