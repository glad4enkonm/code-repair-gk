# code-repair-gk

Bug localization and program repair with local open models via a
repository knowledge graph. Released with the paper "Bug Localization
and Program Repair with Local Open Models via a Repository Knowledge
Graph" (Sibiron 2026).

Instead of retrieving files by similarity, the pipeline replaces the
retrieval stage of localization with an explicit representation of
repository structure: a deterministic knowledge graph (built with
Python's `ast` module from the repository checkout, cached per repo and
base commit) that the model explores with five graph-query tools, a
candidate pool assembled from the explored subgraph, and one list-wise
rerank call. Repair is a single SEARCH/REPLACE chat call. Everything
runs against a local llama.cpp server; no external API is used.

## Results (SWE-bench Lite, test split)

| Method | R@1 | R@3 | R@5 |
|---|---|---|---|
| BM25 (file name + contents) | 42.1 | 56.2 | 66.2 |
| Agentless (hierarchical localization) | 65.6 | 84.3 | 87.3 |
| **KG exploration + list-wise rerank** | **74.2** | **84.6** | **88.0** |
| KG pool ceiling (any reranker) | | | 90.3 |
| Oracle fusion KG + BM25 top-5 | | | 93.6 |

Localization metrics: n=299 (`sympy__sympy-17139` pool exceeds the
reranking budget and is recorded as failed), ground truth = files
modified by the gold patch, READMEs removed from ranked lists,
macro-averaged per-instance fractions. End-to-end resolved rates with
one local model: 33.0% KG-rerank context vs 22.3% BM25 (n=300).

## Pipeline

| Stage | What | Code |
|---|---|---|
| 0 | repo clone + checkout at base commit (cached) | `src/kg/build_graphs.py` |
| 1 | parse Python files, build knowledge graph (cached per repo@commit) | `src/kg/build_graphs.py` |
| 2 | node embeddings into a cross-repo blob-keyed cache | `src/kg/embeddings.py` |
| 3 | LLM explores the graph with query tools | `src/kg/explore.py`, `src/kg/prompts.py` |
| 4 | candidate pool assembled from explored subgraph | `src/kg/context.py` |
| 5 | list-wise rerank of pool files | `scripts/llm-rerank-files.py` |
| 6 | style-3 prompt packing (READMEs + file contents) | `src/kg/context.py` |
| 8 | single SEARCH/REPLACE repair call | `src/kg/repair.py`, `src/entrypoints/run_repair_inference.py` |
| eval | localization metrics, GT from gold patch | `scripts/compare-file-localization.py` |

A stage-by-stage description with code references is in
`docs/Pipeline-Description.md`; metric conventions are in
`docs/LocalizationMetrics.md`.

## Install

```bash
pip install -e .            # core pipeline
pip install -e ".[dev]"     # + tests, lint, type-check
```

The pipeline talks to an OpenAI-compatible local server (llama.cpp
`llama-server`, e.g. served from a GPU container; endpoint and model
names are set in `config/kg_config.json`). Baseline entrypoints that
drive the Agentless fork additionally need
`pip install -e ".[agentless]"`.

## Models

Both weights are fully local and pinned by SHA-256 in
`artifacts/weight_hashes/`:

- chat backbone: unsloth Q8_0 GGUF of Qwen3-Coder-Next
  (https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF/tree/main/Q8_0)
- embeddings: nomic-embed-code Q8_0
  (https://huggingface.co/nomic-ai/nomic-embed-code-GGUF/blob/main/nomic-embed-code.Q8_0.gguf)

## Quickstart

```bash
# 1. build graphs for a dataset split (cached per repo@commit)
scripts/kg-build.sh <dataset> <split>

# 2. graph exploration + pool assembly + prompt packing
scripts/kg-run.sh <dataset> <split>

# 3. list-wise rerank of the candidate pool
python scripts/llm-rerank-files.py --split <split> --variant embed \
    --mode list --signals freq,sim_max,sim_mean,parts \
    --top-k 5 --subparts 3 --src-chars 120 \
    --model qwen3-coder-next-Q8_0 --config config/kg_config.json \
    --progress <progress.jsonl>

# 4. single SEARCH/REPLACE repair
#    (--mode sr; the experimental --mode tool agent loop is not part of
#    this release)
python src/entrypoints/run_repair_inference.py \
    --dataset_name_or_path <rerank5_dataset> --split <split> \
    --mode sr --model_name_or_path qwen3-coder-next-Q8_0 \
    --output_dir outputs/ --max_tokens 8192 --temperature 0.0

# 5. localization metrics against gold patches
python scripts/compare-file-localization.py --help
```

## Repository layout

```
src/kg/          pipeline stages (graph build, explore, context, repair)
src/code_kg_builder/  deterministic knowledge-graph construction package
                 (ast-based executors, graph writer, orchestrator)
src/verification/  Z3-based graph verification (origin/meta/data levels),
                 used by graph commits
src/entrypoints/ dataset builders and inference entrypoints
src/inference/   llama.cpp chat client
src/tool/        vendored subset of the graph-query tool package (see
                 src/tool/README.md)
scripts/         runnable stage scripts and evaluation
config/          serving + pipeline configs (qwen primary, gemma26b
                 robustness arm)
swe_bench_kg/    graph schema seeds (origin/meta levels) plus the empty
                 data-level store, copied into every built graph dir
tests/           pytest suite (python_files = *_test.py)
docs/            pipeline description, metric conventions
artifacts/       per-instance outputs + eval logs (see artifacts/README.md)
```

## Related artifacts

- interactive KG viewer: https://fair-code.ru/kg/ (source:
  https://github.com/glad4enkonm/logico2)
- full knowledge-graph dump for the pydicom example:
  https://fair-code.ru/thedata/kg_pydicom_dev.json
- Agentless fork used as the baseline:
  https://github.com/glad4enkonm/Agentless

## License

MIT (see LICENSE).
