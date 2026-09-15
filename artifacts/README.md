# Artifacts

Per-instance outputs and evaluation logs backing the paper's tables.
**This directory is a placement plan**: the structure ships with the
repository, data files are added separately (each section below gives
the exact source file and preparation command).

Paths under `outputs/` are relative to the experiment root;
`results/...` paths are relative to the Agentless runtime tree.

## weight_hashes/

| File | Source | Preparation |
|---|---|---|
| `qwen3-coder-next-Q8_0__3shards.sha256` | `outputs/weight_hashes/qwen3-coder-next-Q8_0__3shards.sha256` | copy as-is |
| `nomic-embed-code.Q8_0.gguf.sha256` | embedding weight | `sha256sum <nomic-embed-code.Q8_0.gguf> > nomic-embed-code.Q8_0.gguf.sha256` |

## localization/

Knowledge-graph localization progress (graph exploration output), one
JSONL row per instance, `text_inputs` stripped, pool preserved. A
single file covers all 323 instances (dev 23 + test 300); the
evaluator takes split ids from the Agentless baseline files:

| File | Source |
|---|---|
| `kg__k-auto__embed.progress.jsonl` (323 rows) | `outputs/data__swe-bench_lite__style-3__fs-kg__k-auto__embed.progress.jsonl` |
| `kg__k-auto__embed__file_snippet.progress.jsonl` (23 rows, optional) | `outputs/data__swe-bench_lite__style-3__fs-kg__k-auto__file_snippet.progress.jsonl` |
| `kg__k-auto__embed__node_source.progress.jsonl` (23 rows, optional) | `outputs/data__swe-bench_lite__style-3__fs-kg__k-auto__node_source.progress.jsonl` |

The source files' `k-auto` dataset merge has one `test` split of 323
rows; dev = the first 23 instance ids. Preparation (per row): extract
the candidate pool from the prompt text BEFORE dropping the field:

```python
import json, re
pool_re = re.compile(r"\[start of (.+?)\]")
# row["pool"] = pool_re.findall(row.pop("text_inputs"))
# keep: instance_id, repo, base_commit, problem_statement, patch,
#       touched_nodes, pool
```

## localization_rerank/

List-wise rerank progress (ranked files per instance), ships as-is:

| File | Source |
|---|---|
| `llm_rerank__file_level__embed__list__freq-simmax-simmean-parts__dev__qwen3-coder-next-Q8_0.progress.jsonl` (23 rows) | `outputs/` same name |
| `llm_rerank__file_level__embed__list__freq-simmax-simmean-parts__test__qwen3-coder-next-Q8_0.progress.jsonl` (300 rows) | `outputs/` same name |

## repair/

Single-call SEARCH/REPLACE repair predictions (`--mode sr`):

| File | Source |
|---|---|
| `qwen3-coder-next-Q8_0__*__sr__dev.jsonl` | `outputs/` (dev rerank5 tree) |
| `qwen3-coder-next-Q8_0__*__sr__test.jsonl` | `outputs/` (test rerank5 tree) |

## baselines/

| File | Source | Note |
|---|---|---|
| `bm25_file_name_and_contents.retrieval.jsonl` | `SWEbench baseline tree: outputs/swe-bench_lite/file_name_and_contents.retrieval.jsonl` | strip `{instance_id}/` docid prefix when scoring |
| `agentless_found_files_dev.jsonl` | `results/swe-bench-lite-minimal/file_level/loc_outputs.jsonl` | fields: `instance_id`, `found_files` |
| `agentless_found_files_test.jsonl` | `results/swe-bench-lite-test/file_level/loc_outputs.jsonl` | same |

## eval/

Reports backing the paper tables, produced by
`scripts/compare-file-localization.py` over the files above.
Conventions: ground truth = files added by the gold patch; READMEs
removed from every ranked list; macro-averaged per-instance fractions;
`sympy__sympy-17139` recorded as failed (pool exceeds the reranking
budget), so test-split localization metrics use n=299.

Key numbers to reproduce (test split, qwen arm):
Recall@1/3/5 = 74.2 / 84.6 / 88.0 (n=299), pool ceiling 90.3,
oracle fusion over KG+BM25 top-5 = 93.6; BM25 42.1 / 56.2 / 66.2;
Agentless 65.6 / 84.3 / 87.3.

The `--eval-only` mode of `scripts/llm-rerank-files.py` recomputes
rerank metrics from a progress file; it needs
`--signals freq,sim_max,sim_mean,parts`.

## gemma26b/

Robustness arm (Gemma-4-26B-A4B backbone; not reported in the paper).
To be added when the test-split rerank completes: rerank progress
`outputs/gemma26b/llm_rerank__file_level__embed__list__freq-simmax-simmean-parts__test__gemma4-26b-A4B-Q8_0.progress.jsonl`
plus the `--eval-only` report. The config already ships as
`config/kg_config.gemma26b.json`.
