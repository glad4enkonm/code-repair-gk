# Localization Metrics — How to Compute GT-Hit Correctly

How to score localization (file-level) for KG / Agentless / BM25 without the
known pitfalls. Short, canonical, copy-paste-able.

## 1. Ground-truth files

From any progress/dataset row's `patch` field:

```python
import re
gt = set(re.findall(r'(?m)^\+\+\+ b/(.+)$', row["patch"]))
```

## 2. Candidate pool (KG)

The pool that LLM rerank sees = files with contents in `text_inputs`:

```python
pool = set(re.findall(r"\[start of (.+?)\]", row["text_inputs"]))
```

`touched_nodes` mapped to files is the SAME pool (origin). The mapping MUST go
through the graph — node ids are `File-`, `Class-`, `Method-`, `Function-`, ...
and only the graph knows each node's containing file:

```python
from pathlib import Path
from kg.embeddings import resolve_relative_paths

graph_dir = str(Path("outputs/kg_graphs") /
                f"{row['repo'].replace('/', '__')}__{row['base_commit']}")
node_paths = resolve_relative_paths(graph_dir)   # {node_id: file_path}

touched_files = {node_paths[n] for n in row["touched_nodes"]
                 if node_paths.get(n, "").endswith(".py")}
```

### Do NOT

- **Do not** strip only the `File-` prefix and leave other node ids as-is.
  Non-File nodes then become garbage "files"; files touched only via e.g. a
  `Method-` node are silently lost. (This bug once reported 65.9% instead of
  86.4%.)
- **Do not** compare node ids to file paths directly.
- **Do not** reuse `graph_dir_for_row` results across repos — the mapping is
  per graph (per `repo__base_commit`).

## 3. Baselines

- **Agentless**: `found_files` list from
  `Agentless/results/{split}/file_level/loc_outputs.jsonl` — use as-is.
- **BM25**: `hits[].docid` from `swe-bench_lite/file_name_and_contents.retrieval.jsonl`,
  ranked by score desc. Take top-k. Docids are bare relative paths (verified
  2026-09-10: 0/6439 carry a prefix), but strip defensively before matching GT.

```python
prefix = f"{instance_id}/"
docs = [d[len(prefix):] if d.startswith(prefix) else d
        for d in [h["docid"] for h in row["hits"]]]
```

- Eval-result jsonl files (`qwen3-coder-next-Q8_0__*.jsonl`) live under
  `outputs/`, NOT in the experiment root.

## 4. Comparing methods on equal footing

Always restrict ALL methods to the **same instance set** (the intersection of
available ids — e.g. BM25 retrieval covers 43/44 for some slices) and report
list size next to every rate. Reference comparison (45 test instances,
2026-08-28):

| Method | Hit | Rate | List size |
|---|---|---|---|
| Agentless (k=4) | 40/45 | 88.9% | 4 |
| KG pool (k-auto) | 39/45 | 86.7% | avg 58 |
| BM25@20 | 36/45 | 80.0% (saturates here) | 20 |
| BM25@5 | 31/45 | 68.9% | 5 |

Canonical snippet (load once, score every method on the same `ids`):

```python
import json, re

rows = [json.loads(l) for l in open(PROGRESS)][23:]          # skip dev rows
gt = {r["instance_id"]: set(re.findall(r'(?m)^\+\+\+ b/(.+)$', r["patch"]))
      for r in rows}
kg = {r["instance_id"]: set(re.findall(r"\[start of (.+?)\]", r["text_inputs"]))
      for r in rows}
ids = list(gt)

ag = {}                                                        # Agentless
for l in open(f"{AGENTLESS}/swe-bench-lite-test/file_level/loc_outputs.jsonl"):
    j = json.loads(l)
    if j["instance_id"] in gt:
        ag[j["instance_id"]] = set(j["found_files"])

bm = {}                                                        # BM25
for l in open(f"{SWEBENCH}/outputs/swe-bench_lite/file_name_and_contents.retrieval.jsonl"):
    j = json.loads(l)
    if j["instance_id"] in gt:
        p = j["instance_id"] + "/"
        bm[j["instance_id"]] = [d[len(p):] if d.startswith(p) else d
                                for d in (h["docid"] for h in j["hits"])]

rate = lambda hits, n: f"{hits}/{n} = {hits/n*100:.1f}%"
print(rate(sum(1 for i in ids if kg[i] & gt[i]), len(ids)))          # KG pool
print(rate(sum(1 for i in ids if ag.get(i, set()) & gt[i]), len(ids)))
k = 5
print(rate(sum(1 for i in ids if set(bm.get(i, [])[:k]) & gt[i]), len(ids)))
```

Also compute **miss overlap** (`KG&AG`, `KG&BM`, all) — complementary misses
mean a union/hybrid pool raises the ceiling (KG ∪ Agentless = 44/45 = 97.8%
on the reference slice). Use `bm.get(i, [])` — retrieval files can miss
instances.

## 5. Definitions

- **Pool ceiling** (a.k.a. Recall@∞): GT ∩ pool ≠ ∅. Upper bound for any
  rerank — rerank reorders the pool, it can never recover a file outside it.
- **Recall@k** (post-rerank): GT ∩ top-k of reranked list ≠ ∅.
- Always report pool size (avg files) next to the rate — a high rate over a
  500-file pool is not comparable to k=4.

## 6. Sanity checks before reporting

1. Same instance set across methods being compared (intersect ids).
2. Pool from `text_inputs` markers == `touched_nodes` mapped via
   `resolve_relative_paths` (allow ~1 file drift; large drift = mapping bug).
3. Rows counted once (progress files may contain re-runs; dedupe by
   `instance_id`, keep last).
