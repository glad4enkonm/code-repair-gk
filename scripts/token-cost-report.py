"""Token-cost report: KG pool vs BM25 k=5 vs Agentless file localization.

Two cost axes on the same instance set (the KG test rows present in the
embed progress file):

A. pipeline input tokens — what each method's own pipeline consumes,
   PER STAGE and for every LLM call (``--agentless-dir``,
   ``--bm25-jsonl``, ``--kg-sr``, ``--rerank-progress``). Agentless
   stages use server-reported ``usage`` (exact); BM25/KG solve stages
   tokenize ``text``/``full_output``; the list rerank is reconstructed
   deterministically from the sibling rerank module. The embedding
   stage is accounted SEPARATELY (``--embed-cache``): exact vector
   count from the shared chroma cache, node-side chars via optional
   graph replay, query-side estimate — nomic tokens, not Qwen.

B. solver context tokens — tokens of problem statement + file contents
   each approach hands downstream. KG/BM25: ``text_inputs``/``text``
   tokenized directly. Agentless: estimated by tokenizing the
   ``found_files`` contents from the per-instance BM25 corpus
   (``file_name_and_contents_indexes``) plus the problem statement.

Usage:

    python scripts/token-cost-report.py \
        --kg-progress artifacts/localization/kg__k-auto__embed.progress.jsonl \
        --skip-dev 23 \
        --bm25-jsonl <bm25_predictions_test.jsonl> \
        --agentless artifacts/baselines/agentless_found_files_test.jsonl \
        --agentless-dir <agentless_results_test_dir> \
        --corpus-dir <bm25_corpus_indexes_dir> \
        --tokenizer <tokenizer.json> \
        --kg-sr artifacts/repair/<kg_sr_test>.jsonl \
        --embed-cache <chroma_db_dir> \
        --output outputs/token_cost_report_test.json
"""

import argparse
import importlib.util
import json
import logging
import math
import re
import sqlite3
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

# Agentless per-stage figures cited verbatim from arXiv 2407.01489v2
# (Tables 1-2): full-pipeline avg tokens are input+output combined for
# GPT-4o (gpt-4o-2024-05-13); the file-level localization stage cost is
# the "Combined (default)" row ($0.02 prompting + $0.04 embedding-based
# with irrelevant-folder filtering, out of $0.70 per task in total).
AGENTLESS_PAPER_NUMBERS = {
    "avg_tokens_full_pipeline": 78166,
    "avg_cost_full_pipeline_usd": 0.70,
    "file_level_stage_cost_usd": 0.06,
    "file_level_prompting_cost_usd": 0.02,
    "file_level_embedding_cost_usd": 0.04,
    "llm": "gpt-4o-2024-05-13",
    "source": "https://arxiv.org/abs/2407.01489 (v2), Tables 1-2",
}

_POOL_FILE_RE = re.compile(r"^\[start of (.+?)\]$", re.MULTILINE)
_GT_FILE_RE = re.compile(r"(?m)^\+\+\+ b/(.+)$")


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


class TokenCounter:
    """Exact token counts via a tokenizers.Tokenizer-like object."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._tokenizer.encode(text).ids)

    def describe(self) -> str:
        vocab = getattr(self._tokenizer, "get_vocab_size", lambda: None)()
        return f"exact tokenizer (vocab={vocab})"


class CharsHeuristicCounter:
    """chars/4 approximation used when no tokenizer file is available."""

    def count(self, text: str) -> int:
        return len(text) // 4

    def describe(self) -> str:
        return "chars/4 heuristic"


def load_qwen_tokenizer(tokenizer_path: str):
    """Load a tokenizer.json via the tokenizers library (lazy import)."""
    from tokenizers import Tokenizer  # noqa: PLC0415 — optional dependency

    return Tokenizer.from_file(tokenizer_path)


def make_counter(tokenizer_path: str | None = None, tokenizer=None):
    """Build a counter; exact when a tokenizer is available, else chars/4."""
    if tokenizer is not None:
        return TokenCounter(tokenizer)
    if tokenizer_path:
        return TokenCounter(load_qwen_tokenizer(tokenizer_path))
    return CharsHeuristicCounter()


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def pool_files(text_inputs: str) -> list[str]:
    """Candidate pool file paths, in exploration order."""
    return _POOL_FILE_RE.findall(text_inputs or "")


def gt_files(patch: str) -> list[str]:
    """Unique files modified by the gold patch, in diff order."""
    seen = []
    for path in _GT_FILE_RE.findall(patch or ""):
        if path not in seen:
            seen.append(path)
    return seen


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def read_jsonl(path: str):
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_kg_rows(progress_path: str, skip: int = 0) -> list[dict]:
    """KG progress rows (skipping the first ``skip`` dev rows)."""
    rows = []
    for position, row in enumerate(read_jsonl(progress_path)):
        if position < skip:
            continue
        rows.append(
            {
                "instance_id": row["instance_id"],
                "repo": row.get("repo", ""),
                "problem_statement": row.get("problem_statement", ""),
                "patch": row.get("patch", ""),
                "text_inputs": row.get("text_inputs", ""),
            }
        )
    return rows


def load_bm25_texts(dataset_dir: str | None = None, jsonl_path: str | None = None):
    """Map instance_id -> rendered prompt text for the BM25 k=5 baseline."""
    if jsonl_path:
        return {
            row["instance_id"]: row.get("text", "")
            for row in read_jsonl(jsonl_path)
        }
    if dataset_dir:
        from datasets import load_from_disk  # noqa: PLC0415 — optional dependency

        dataset = load_from_disk(dataset_dir)
        texts = {}
        for split in dataset:
            for row in dataset[split]:
                texts[row["instance_id"]] = row.get("text", "")
        return texts
    raise ValueError("one of --bm25-dir / --bm25-jsonl is required")


def load_agentless_found(loc_outputs_path: str) -> dict[str, list[str]]:
    """Map instance_id -> Agentless found_files (ordered most→least)."""
    return {
        row["instance_id"]: row.get("found_files", [])
        for row in read_jsonl(loc_outputs_path)
    }


_ISSUE_RE = re.compile(r"<issue>\n(.*?)\n</issue>", re.DOTALL)
_CODE_RE = re.compile(r"<code>\n(.*?)\n</code>", re.DOTALL)


def _load_sr_prompt_builders():
    """Import build_sr_prompt / truncate_file_sections from src/kg/repair.py."""
    src_dir = Path(__file__).resolve().parent.parent / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from kg.repair import build_sr_prompt, truncate_file_sections

    return build_sr_prompt, truncate_file_sections


def load_sr_predictions(
    sr_path: str,
    sr_reconstruct: bool = False,
    truncated_ids: set[str] | None = None,
) -> dict[str, dict]:
    """Map instance_id -> solver prompt / completion text.

    ``text`` in the jsonl is the *dataset* prompt (style-3, line-numbered).
    For the BM25 solve stage that text IS what was sent to the model
    (``sr_reconstruct=False``, default). For the KG sr stage the sent
    prompt differs: ``run_repair_inference`` extracts ``<issue>/<code>``,
    applies ``truncate_file_sections(max_files=5)`` for the ids in
    ``truncated_ids`` (the truncfix12 resume run), strips line numbers and
    re-wraps via ``build_sr_prompt`` — pass ``sr_reconstruct=True`` to store
    that reconstruction as ``prompt``. The raw dataset text is always kept
    as ``prompt_dataset_text``.
    """
    build_sr_prompt = truncate_file_sections = None
    if sr_reconstruct:
        build_sr_prompt, truncate_file_sections = _load_sr_prompt_builders()
    truncated = truncated_ids or set()
    result: dict[str, dict] = {}
    for row in read_jsonl(sr_path):
        text = row.get("text", "")
        prompt = text
        if sr_reconstruct:
            issue_match = _ISSUE_RE.search(text)
            code_match = _CODE_RE.search(text)
            if issue_match and code_match:
                code = code_match.group(1)
                if row["instance_id"] in truncated:
                    code = truncate_file_sections(code, max_files=5)
                prompt = build_sr_prompt(issue_match.group(1), code)
            else:
                logging.getLogger(__name__).warning(
                    "sr prompt reconstruction skipped for %s "
                    "(no <issue>/<code> markers); using dataset text",
                    row.get("instance_id"),
                )
        result[row["instance_id"]] = {
            "prompt": prompt,
            "prompt_dataset_text": text,
            "output": row.get("full_output", ""),
        }
    return result


def load_corpus(index_dir: str, instance_id: str) -> dict[str, str]:
    """Map file path -> contents from the per-instance BM25 documents.jsonl."""
    documents_path = Path(index_dir) / instance_id / "documents.jsonl"
    if not documents_path.exists():
        return {}
    return {
        row["id"]: row.get("contents", "")
        for row in read_jsonl(str(documents_path))
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def token_stats(values: list[int]) -> dict:
    """mean / median / p90 (nearest-rank) of a token-count list."""
    if not values:
        return {"mean": None, "median": None, "p90": None}
    ordered = sorted(values)
    n = len(ordered)
    p90_index = min(n - 1, math.ceil(0.9 * n) - 1)
    return {
        "mean": sum(ordered) / n,
        "median": ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2,
        "p90": ordered[p90_index],
    }


def evaluate_method(instances: list[dict]) -> dict:
    """Aggregate per-instance token/file/hit records into one summary.

    Each instance record: {instance_id, tokens, files, hit?}. ``hit`` is
    optional; without it acc / tokens_per_hit are reported as None.
    """
    tokens = [i["tokens"] for i in instances]
    file_counts = [i["files"] for i in instances]
    hits = [i for i in instances if i.get("hit") is True]
    summary = {
        "n": len(instances),
        "avg_files": (sum(file_counts) / len(file_counts)) if file_counts else None,
        **{f"{k}_tokens": v for k, v in token_stats(tokens).items()},
    }
    if all("hit" in i for i in instances) and instances:
        summary["acc"] = len(hits) / len(instances)
        summary["tokens_per_hit"] = (sum(tokens) / len(hits)) if hits else None
    else:
        summary["acc"] = None
        summary["tokens_per_hit"] = None
    return summary


# ---------------------------------------------------------------------------
# Agentless solver-context estimation
# ---------------------------------------------------------------------------


def agentless_context_tokens(
    found_files: list[str],
    problem_statement: str,
    corpus: dict[str, str],
    counter,
) -> tuple[int, list[str]]:
    """Token estimate for problem statement + found file contents.

    Files missing from the corpus contribute zero tokens and are
    returned in the missing list (they skew the estimate low).
    """
    missing = [path for path in found_files if path not in corpus]
    total = counter.count(problem_statement)
    for path in found_files:
        if path in corpus:
            total += counter.count(corpus[path])
    return total, missing


# ---------------------------------------------------------------------------
# Per-stage token accounting (every LLM call per pipeline)
# ---------------------------------------------------------------------------

# Agentless results layout: <dir>/<rel path>, trajectory field per stage.
AGENTLESS_STAGE_FILES = {
    "file_loc": ("file_level/loc_outputs.jsonl", "file_traj"),
    "related_loc": ("related_elements/loc_outputs.jsonl", "related_loc_traj"),
    "edit_loc": ("edit_location_samples/loc_outputs.jsonl", "edit_loc_traj"),
}

EMBED_DOC_PREFIX_CHARS = 22  # "search_code_document: "
EMBED_QUERY_PREFIX_CHARS = 19  # "search_code_query: "
EMBED_DOC_CAP_CHARS = 8192  # config.embedding_max_doc_chars


def _sum_usage_records(records: list) -> tuple[int, int, int]:
    """Sum (prompt, completion, calls) over per-call usage records."""
    prompt = completion = calls = 0
    for call in records:
        usage = call.get("usage") or {}
        prompt += usage.get("prompt_tokens") or 0
        completion += usage.get("completion_tokens") or 0
        calls += 1
    return prompt, completion, calls


def load_agentless_stage_usage(path: str, traj_key: str) -> dict:
    """instance_id -> server-reported usage for one localization stage.

    The traj field is a single dict for file_loc/edit_loc but a LIST of
    sequential calls for related_loc (Agentless prompts twice there);
    both shapes are summed per instance.
    """
    usages = {}
    for row in read_jsonl(path):
        traj = row.get(traj_key)
        if isinstance(traj, list):
            prompt, completion, calls = _sum_usage_records(traj)
        elif isinstance(traj, dict):
            prompt, completion, calls = _sum_usage_records([traj])
        elif traj is None:
            prompt = completion = calls = 0
        else:
            print(
                f"warning: {path}: {row.get('instance_id')}: unexpected "
                f"{traj_key} type {type(traj).__name__}, counted as zero"
            )
            prompt = completion = calls = 0
        usages[row["instance_id"]] = {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "calls": calls,
        }
    return usages


def load_agentless_repair_usage(path: str) -> dict:
    """instance_id -> usage summed over the per-sample traj list."""
    usages = {}
    for row in read_jsonl(path):
        prompt, completion, calls = _sum_usage_records(row.get("traj") or [])
        usages[row["instance_id"]] = {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "calls": calls,
        }
    return usages


def usage_stage_summary(usages: dict) -> dict:
    """Aggregate per-instance usage records into one stage summary."""
    if not usages:
        return {"n": 0}
    prompts = [u["prompt_tokens"] for u in usages.values()]
    completions = [u["completion_tokens"] for u in usages.values()]
    totals = [p + c for p, c in zip(prompts, completions)]
    calls = [u.get("calls", 1) for u in usages.values()]
    prompt_stats = token_stats(prompts)
    completion_stats = token_stats(completions)
    total_stats = token_stats(totals)
    return {
        "n": len(usages),
        "avg_calls": sum(calls) / len(calls),
        "prompt_mean_tokens": prompt_stats["mean"],
        "prompt_median_tokens": prompt_stats["median"],
        "prompt_p90_tokens": prompt_stats["p90"],
        "completion_mean_tokens": completion_stats["mean"],
        "completion_median_tokens": completion_stats["median"],
        "completion_p90_tokens": completion_stats["p90"],
        "total_mean_tokens": total_stats["mean"],
        "total_median_tokens": total_stats["median"],
        "total_p90_tokens": total_stats["p90"],
    }


def collect_agentless_stages(agentless_dir: str) -> dict:
    """Per-instance usages for every Agentless stage found on disk."""
    stages = {}
    base = Path(agentless_dir)
    for stage, (rel_path, traj_key) in AGENTLESS_STAGE_FILES.items():
        stage_path = base / rel_path
        if stage_path.exists():
            stages[stage] = load_agentless_stage_usage(str(stage_path), traj_key)
        else:
            print(f"warning: agentless stage file missing: {stage_path}")
    repair_path = base / "repair" / "output.jsonl"
    if repair_path.exists():
        stages["repair"] = load_agentless_repair_usage(str(repair_path))
    else:
        print(f"warning: agentless repair file missing: {repair_path}")
    return stages


def agentless_per_instance_totals(stages: dict) -> dict:
    """instance_id -> prompt+completion tokens summed over all stages."""
    totals: dict = {}
    for usages in stages.values():
        for iid, usage in usages.items():
            totals[iid] = totals.get(iid, 0) + (
                usage["prompt_tokens"] + usage["completion_tokens"]
            )
    return totals


def solve_stage_from_predictions(predictions: dict, counter) -> dict:
    """Input/output token stats for a single-call solve stage.

    ``predictions`` maps instance_id -> {prompt, output} (the generic
    text/full_output predictions schema shared by BM25 and KG sr runs).
    """
    inputs = []
    outputs = []
    totals = []
    for value in predictions.values():
        input_tokens = counter.count(value.get("prompt") or "")
        output_tokens = counter.count(value.get("output") or "")
        if value.get("prompt"):
            inputs.append(input_tokens)
        if value.get("output"):
            outputs.append(output_tokens)
        if value.get("prompt") or value.get("output"):
            totals.append(input_tokens + output_tokens)
    input_stats = token_stats(inputs)
    output_stats = token_stats(outputs)
    total_stats = token_stats(totals)
    return {
        "n": len(predictions),
        "input_mean_tokens": input_stats["mean"],
        "input_median_tokens": input_stats["median"],
        "input_p90_tokens": input_stats["p90"],
        "output_n": len(outputs),
        "output_mean_tokens": output_stats["mean"],
        "output_median_tokens": output_stats["median"],
        "output_p90_tokens": output_stats["p90"],
        "total_per_instance_mean_tokens": total_stats["mean"],
        "total_per_instance_median_tokens": total_stats["median"],
        "total_per_instance_p90_tokens": total_stats["p90"],
    }


def embed_query_chars(
    problem_statement: str, doc_cap: int = EMBED_DOC_CAP_CHARS, prefix_len: int = EMBED_QUERY_PREFIX_CHARS
) -> int:
    """Chars embedded for one query-side call (prefix + truncated issue)."""
    return prefix_len + min(len(problem_statement or ""), doc_cap)


def embed_cache_stats(cache_dir: str) -> dict:
    """Exact vector count + per-repo breakdown from the shared embed cache."""
    sqlite_path = Path(cache_dir) / "chroma.sqlite3"
    if not sqlite_path.exists():
        print(f"warning: embed cache db not found: {sqlite_path}")
        return {"vectors": 0, "per_repo": {}}
    connection = sqlite3.connect(str(sqlite_path))
    try:
        vectors = connection.execute("SELECT count(*) FROM embeddings").fetchone()[0]
        per_repo = dict(
            connection.execute(
                "SELECT string_value, count(*) FROM embedding_metadata "
                "WHERE key = 'repo_key' GROUP BY 1 ORDER BY 2 DESC"
            ).fetchall()
        )
    finally:
        connection.close()
    return {"vectors": vectors, "per_repo": per_repo}


def graphs_embed_chars(graphs_dir: str) -> dict:
    """Replay composed node-document lengths over cached KG graphs.

    Mirrors the real embedding inputs (DOCUMENT_PREFIX + composed node
    text, source capped at 2048 chars, doc at 8192, skipped node types
    excluded). Counts every node of every graph WITHOUT cross-commit
    dedupe — the shared cache deduped repeats, so this is an upper
    bound on the unique vectors actually computed.
    """
    src_dir = _ROOT / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from kg.config import KGConfig  # noqa: PLC0415 — lazy, heavy-free import
    from kg.embeddings import (  # noqa: PLC0415
        DEFAULT_SKIP_NODE_TYPES,
        DOCUMENT_PREFIX,
        compose_node_text,
        resolve_relative_paths,
    )

    config = KGConfig()
    skip = set(config.embedding_skip_node_types or DEFAULT_SKIP_NODE_TYPES)
    graphs = nodes = total_chars = 0
    for graph_dir in sorted(Path(graphs_dir).iterdir()):
        data_path = graph_dir / "data.json"
        if not data_path.exists():
            continue
        data = json.loads(data_path.read_text())
        node_paths = resolve_relative_paths(str(graph_dir))
        all_values = data.get("allValues", {})
        for node in data.get("nodes", []):
            props = all_values.get(node["id"], {})
            if props.get("meta_type") in skip:
                continue
            document = DOCUMENT_PREFIX + compose_node_text(
                node, props, node_paths.get(node["id"], ""), config
            )
            total_chars += len(document[: config.embedding_max_doc_chars])
            nodes += 1
        graphs += 1
    return {"graphs": graphs, "nodes": nodes, "total_chars": total_chars}


# ---------------------------------------------------------------------------
# List-rerank token reconstruction (pure core; LLM-free by design)
# ---------------------------------------------------------------------------


def estimate_rerank_tokens(
    issue: str,
    prompt_signals: list[dict],
    signals: list[str],
    top_k: int,
    max_prompt_chars: int,
    n_calls: int,
    counter,
    build_prompt,
    chunk_prompts,
) -> dict:
    """Reconstruct list-mode rerank token cost for one instance.

    ``build_prompt(issue, chunk, signals, top_k) -> str`` and
    ``chunk_prompts(issue, prompt_signals, signals, top_k, cap)`` mirror
    the sibling rerank module; they are injected so this function stays
    free of imports and LLM/graph dependencies. ``n_calls`` is the
    stored per-instance LLM call count (includes retries).
    """
    chunks = chunk_prompts(issue, prompt_signals, signals, top_k, max_prompt_chars)
    prompt_tokens = sum(
        counter.count(build_prompt(issue, chunk, signals, top_k))
        for chunk in chunks
    )
    if not chunks:
        return {
            "input_tokens": 0,
            "output_tokens_estimate": 0,
            "n_chunks": 0,
            "retry_multiplier": None,
        }
    retry_multiplier = (n_calls / len(chunks)) if n_calls else 1.0
    # Output estimate: the JSON-array answer of top_k paths plus a small
    # allowance for prose/formatting the template invites (raw answers
    # were not stored; this is a lower-bound-ish estimate).
    answer_paths = [fs.get("path", "") for fs in prompt_signals[:top_k]]
    output_estimate = counter.count(json.dumps(answer_paths)) + 16
    return {
        "input_tokens": int(round(prompt_tokens * retry_multiplier)),
        "output_tokens_estimate": output_estimate,
        "n_chunks": len(chunks),
        "retry_multiplier": round(retry_multiplier, 3),
    }


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def build_report(
    kg_instances: list[dict],
    bm25_texts: dict[str, str],
    agentless_found: dict[str, list[str]],
    corpus_dir: str,
    counter,
    sr_prompts: dict[str, dict] | None = None,
    stages: dict | None = None,
    embed: dict | None = None,
) -> dict:
    """Assemble the per-method summaries over the KG instance set.

    ``sr_prompts`` (optional, from ``--kg-sr``) adds a ``kg_sr`` method:
    the exact solver prompts of the current rerank-top-5 pipeline.
    Instances absent from the sr file fall back to the full pool text
    (marked ``from_pool_fallback``) so partial sr output never crashes
    the report.
    """
    instance_ids = [r["instance_id"] for r in kg_instances]

    kg_records = []
    for row in kg_instances:
        gt = set(gt_files(row["patch"]))
        pool = pool_files(row["text_inputs"])
        kg_records.append(
            {
                "instance_id": row["instance_id"],
                "repo": row["repo"],
                "tokens": counter.count(row["text_inputs"]),
                "files": len(pool),
                "hit": bool(gt & set(pool)),
            }
        )

    kg_sr_records = []
    if sr_prompts is not None:
        for row in kg_instances:
            gt = set(gt_files(row["patch"]))
            sr = sr_prompts.get(row["instance_id"])
            if sr is not None and sr.get("prompt"):
                prompt = sr["prompt"]
                record = {
                    "instance_id": row["instance_id"],
                    "repo": row["repo"],
                    "tokens": counter.count(prompt),
                    "files": len(pool_files(prompt)),
                    "hit": bool(gt & set(pool_files(prompt))),
                    "from_pool_fallback": False,
                    "output_tokens": (
                        counter.count(sr.get("output", ""))
                        if sr.get("output")
                        else None
                    ),
                }
            else:
                prompt = row["text_inputs"]
                record = {
                    "instance_id": row["instance_id"],
                    "repo": row["repo"],
                    "tokens": counter.count(prompt),
                    "files": len(pool_files(prompt)),
                    "hit": bool(gt & set(pool_files(prompt))),
                    "from_pool_fallback": True,
                    "output_tokens": None,
                }
            kg_sr_records.append(record)

    bm25_records = []
    agentless_records = []
    agentless_missing = []
    for row in kg_instances:
        iid = row["instance_id"]
        gt = set(gt_files(row["patch"]))
        text = bm25_texts.get(iid, "")
        bm25_records.append(
            {
                "instance_id": iid,
                "repo": row["repo"],
                "tokens": counter.count(text) if text else None,
                "files": len(pool_files(text)),
                "hit": bool(gt & set(pool_files(text))) if text else None,
            }
        )
        found = agentless_found.get(iid, [])
        corpus = load_corpus(corpus_dir, iid)
        tokens, missing = agentless_context_tokens(
            found, row["problem_statement"], corpus, counter
        )
        agentless_records.append(
            {
                "instance_id": iid,
                "repo": row["repo"],
                "tokens": tokens,
                "files": len(found),
                "hit": bool(gt & set(found)),
            }
        )
        agentless_missing.extend(f"{iid}:{path}" for path in missing)

    bm25_present = [r for r in bm25_records if r["tokens"] is not None]
    methods = {
        "kg_pool": evaluate_method(kg_records),
        "bm25_k5": evaluate_method(bm25_present),
        "agentless": evaluate_method(agentless_records),
    }
    instances = {
        "kg_pool": kg_records,
        "bm25_k5": bm25_records,
        "agentless": agentless_records,
    }
    if kg_sr_records:
        kg_sr_summary = evaluate_method(kg_sr_records)
        kg_sr_summary["n_fallback"] = sum(
            1 for r in kg_sr_records if r["from_pool_fallback"]
        )
        output_tokens = [
            r["output_tokens"]
            for r in kg_sr_records
            if r["output_tokens"] is not None
        ]
        kg_sr_summary["output_n"] = len(output_tokens)
        kg_sr_summary.update(
            {
                f"output_{k}_tokens": v
                for k, v in token_stats(output_tokens).items()
            }
        )
        methods["kg_sr"] = kg_sr_summary
        instances["kg_sr"] = kg_sr_records
    return {
        "counter": counter.describe(),
        "instance_count": len(instance_ids),
        "bm25_instances_missing": len(instance_ids) - len(bm25_present),
        "agentless_files_missing_from_corpus": agentless_missing,
        "agentless_paper_numbers": AGENTLESS_PAPER_NUMBERS,
        "methods": methods,
        "instances": instances,
        **({"stages": stages} if stages else {}),
        **({"embed": embed} if embed else {}),
    }


def _fmt(value, kind="int"):
    if value is None:
        return "—"
    if kind == "int":
        return f"{int(round(value)):,}"
    if kind == "float1":
        return f"{value:.1f}"
    return f"{value:.1%}"


def render_markdown(report: dict) -> str:
    """Render the headline comparison table (B-axis + acc + tok/hit)."""
    m = report["methods"]
    paper = AGENTLESS_PAPER_NUMBERS
    lines = [
        "# Token-cost report (file-level localization)",
        "",
        f"Counter: {report['counter']} — {report['instance_count']} instances",
        "",
        "| Method | n | avg files | acc | mean tok | median | p90 | tok/hit |",
        "|---|---|---|---|---|---|---|---|",
        f"| KG pool (k-auto) | {m['kg_pool']['n']} | {_fmt(m['kg_pool']['avg_files'], 'float1')} | {_fmt(m['kg_pool']['acc'], 'pct')} | {_fmt(m['kg_pool']['mean_tokens'])} | {_fmt(m['kg_pool']['median_tokens'])} | {_fmt(m['kg_pool']['p90_tokens'])} | {_fmt(m['kg_pool']['tokens_per_hit'])} |",
    ]
    if "kg_sr" in m:
        lines.append(
            f"| KG sr (rerank top-5) | {m['kg_sr']['n']} | {_fmt(m['kg_sr']['avg_files'], 'float1')} | {_fmt(m['kg_sr']['acc'], 'pct')} | {_fmt(m['kg_sr']['mean_tokens'])} | {_fmt(m['kg_sr']['median_tokens'])} | {_fmt(m['kg_sr']['p90_tokens'])} | {_fmt(m['kg_sr']['tokens_per_hit'])} |"
        )
    lines += [
        f"| BM25 k=5 | {m['bm25_k5']['n']} | {_fmt(m['bm25_k5']['avg_files'], 'float1')} | {_fmt(m['bm25_k5']['acc'], 'pct')} | {_fmt(m['bm25_k5']['mean_tokens'])} | {_fmt(m['bm25_k5']['median_tokens'])} | {_fmt(m['bm25_k5']['p90_tokens'])} | {_fmt(m['bm25_k5']['tokens_per_hit'])} |",
        f"| Agentless | {m['agentless']['n']} | {_fmt(m['agentless']['avg_files'], 'float1')} | {_fmt(m['agentless']['acc'], 'pct')} | {_fmt(m['agentless']['mean_tokens'])} | {_fmt(m['agentless']['median_tokens'])} | {_fmt(m['agentless']['p90_tokens'])} | {_fmt(m['agentless']['tokens_per_hit'])} |",
        "",
    ]
    if "kg_sr" in m:
        sr = m["kg_sr"]
        lines += [
            f"KG sr solver output tokens (exact, n={sr['output_n']}):",
            "",
            f"- mean {_fmt(sr['output_mean_tokens'])}, median "
            f"{_fmt(sr['output_median_tokens'])}, p90 {_fmt(sr['output_p90_tokens'])}",
            f"- pool-text fallback: {sr['n_fallback']} instance(s)",
            "",
        ]
    lines += [
        "A-axis (pipeline input cost): see the per-stage table below.",
        f"Paper reference (GPT-4o): Agentless full-pipeline {paper['avg_tokens_full_pipeline']:,}"
        f" avg tokens/task — {paper['source']}",
        "",
    ]
    stages = report.get("stages")
    if stages:
        lines += [
            "## Per-stage token accounting (every LLM call per pipeline)",
            "",
            "| Pipeline | Stage | calls/inst | prompt tok (mean) | completion tok (mean) | total tok/inst (mean) |",
            "|---|---|---|---|---|---|",
            "| BM25 | retrieval | — | non-LLM | — | — |",
        ]
        bm25_solve = stages.get("bm25_solve")
        if bm25_solve and bm25_solve.get("n"):
            lines.append(
                f"| BM25 | solve (top-5 files) | 1 | {_fmt(bm25_solve['input_mean_tokens'])} | {_fmt(bm25_solve['output_mean_tokens'])} | {_fmt(bm25_solve['total_per_instance_mean_tokens'])} |"
            )
        agentless_stages = stages.get("agentless", {})
        stage_labels = {
            "file_loc": "file localization",
            "related_loc": "related elements",
            "edit_loc": "edit locations",
            "repair": "repair (samples)",
        }
        for stage_key, label in stage_labels.items():
            stage = agentless_stages.get(stage_key)
            if stage and stage.get("n"):
                lines.append(
                    f"| Agentless | {label} | {stage['avg_calls']:.1f} | {_fmt(stage['prompt_mean_tokens'])} | {_fmt(stage['completion_mean_tokens'])} | {_fmt(stage['total_mean_tokens'])} |"
                )
        agentless_total = stages.get("agentless_total")
        if agentless_total and agentless_total.get("n"):
            lines.append(
                f"| Agentless | **total** | | | | **{_fmt(agentless_total['total_mean_tokens'])}** |"
            )
        kg_rerank = stages.get("kg_rerank")
        if kg_rerank and kg_rerank.get("n"):
            lines.append(
                f"| KG+embed | list rerank | {kg_rerank['avg_chunks']:.2f} | {_fmt(kg_rerank['mean_input_tokens'])} | {_fmt(kg_rerank['mean_output_tokens_estimate'])} (est) | {_fmt(kg_rerank['mean_input_tokens'] + kg_rerank['mean_output_tokens_estimate'])} |"
            )
        kg_solve = stages.get("kg_solve")
        if kg_solve and kg_solve.get("n"):
            lines.append(
                f"| KG+embed | solve (rerank top-5) | 1 | {_fmt(kg_solve['input_mean_tokens'])} | {_fmt(kg_solve['output_mean_tokens'])} | {_fmt(kg_solve['total_per_instance_mean_tokens'])} |"
            )
        lines.append("")
    embed = report.get("embed")
    if embed:
        lines += [
            "### Embedding cost (separate model: "
            f"{embed.get('model', 'nomic-embed-code-Q8_0')} — not Qwen tokens)",
            "",
            f"- vectors computed (exact, cache count): {_fmt(embed.get('vectors'))}",
        ]
        replay = embed.get("graphs_replay")
        if replay:
            lines.append(
                f"- node-side chars (per-graph replay, before cross-commit"
                f" dedupe — upper bound): {_fmt(replay['total_chars'])}"
                f" over {replay['nodes']:,} nodes / {replay['graphs']:,} graphs"
            )
        lines.append(
            f"- node-side chars hard cap: {_fmt(embed.get('node_chars_upper_bound'))}"
            " (vectors × 8,214: 22-char prefix + 8,192-char doc cap;"
            " typical Function doc ≈ 22 + header + min(len(source), 2,048))"
        )
        if embed.get("query_calls_estimate"):
            lines.append(
                f"- query-side (estimate): {embed['query_calls_estimate']:,} embeds"
                f" × ≤8,211 chars ≈ {_fmt(embed.get('query_chars_estimate'))} chars"
                " (1 head-start embed per instance)"
            )
        per_repo = embed.get("per_repo") or {}
        if per_repo:
            top_repos = ", ".join(
                f"{repo} {count:,}" for repo, count in list(per_repo.items())[:5]
            )
            lines.append(f"- per-repo vectors (top 5): {top_repos}")
        lines.append("")
    rerank = report.get("rerank_list") or report.get("rerank_list_dev")
    if rerank:
        lines += [
            "List-rerank (reconstruction, "
            f"{rerank['n']} instances):",
            "",
            f"- input tokens: mean {rerank['mean_input_tokens']:,.0f}, "
            f"median {rerank['median_input_tokens']:,.0f}",
            f"- output tokens (estimate): mean {rerank['mean_output_tokens_estimate']:,.0f}",
            f"- avg chunks per instance: {rerank['avg_chunks']:.2f}, "
            f"avg retry multiplier: {rerank['avg_retry_multiplier']:.2f}",
            "",
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dev-only rerank reconstruction (imports the sibling rerank module)
# ---------------------------------------------------------------------------


def reconstruct_rerank(
    kg_progress_path: str,
    rerank_progress_path: str,
    graphs_dir: str,
    config_path: str | None,
    counter,
) -> dict:
    """Replay list-mode prompt construction for the rerank progress rows.

    Mirrors llm-rerank-files.py's list closure exactly (signals, sims
    via chroma, chunking, spec read from the progress rows themselves).
    Requires the embed endpoint + graph cache; performs no LLM calls.
    """
    rerank_spec = importlib.util.spec_from_file_location(
        "lrf", Path(__file__).resolve().parent / "llm-rerank-files.py"
    )
    lrf = importlib.util.module_from_spec(rerank_spec)
    rerank_spec.loader.exec_module(lrf)

    defaults = lrf._load_defaults(config_path)

    def sim_provider(problem_statement, graph_dir):
        return lrf._node_sims_from_chroma(problem_statement, graph_dir, defaults)

    kg_rows = {
        row["instance_id"]: row
        for row in read_jsonl(kg_progress_path)
    }
    results = []
    for record in read_jsonl(rerank_progress_path):
        spec = {
            "signals": record.get("signals", ["freq", "sim_max", "sim_mean", "parts"]),
            "top_k": record.get("top_k", 5),
            "subparts": record.get("subparts", 3),
            "src_chars": record.get("src_chars", 120),
            "max_prompt_chars": record.get("max_prompt_chars", 100_000),
        }
        row = kg_rows.get(record["instance_id"])
        if row is None:
            continue
        files = lrf.extract_files_with_content(row.get("text_inputs", ""))
        graph_dir = lrf.graph_dir_for_row(row, graphs_dir)
        file_signals = lrf.assemble_file_signals(
            row,
            graph_dir,
            subparts=spec["subparts"],
            src_chars=spec["src_chars"],
            sim_provider=sim_provider,
        )
        signals_by_path = {fs["path"]: fs for fs in file_signals}
        freq_order = [
            path
            for path, _ in sorted(
                files,
                key=lambda f: -signals_by_path.get(f[0], {}).get("touch_freq", 0),
            )
        ]
        prompt_signals = [
            signals_by_path.get(path, {"path": path, "touch_freq": 0})
            for path in freq_order
        ]
        estimate = estimate_rerank_tokens(
            issue=row.get("problem_statement", ""),
            prompt_signals=prompt_signals,
            signals=spec["signals"],
            top_k=spec["top_k"],
            max_prompt_chars=spec["max_prompt_chars"],
            n_calls=record.get("n_comparisons", 0),
            counter=counter,
            build_prompt=lrf.build_list_prompt,
            chunk_prompts=lrf._chunk_prompt_signals,
        )
        results.append(
            {"instance_id": record["instance_id"], "n_files": record.get("n_files"), **estimate}
        )
    if not results:
        return {}
    n = len(results)
    return {
        "n": n,
        "mean_input_tokens": sum(r["input_tokens"] for r in results) / n,
        "median_input_tokens": sorted(r["input_tokens"] for r in results)[n // 2],
        "mean_output_tokens_estimate": sum(
            r["output_tokens_estimate"] for r in results
        )
        / n,
        "avg_chunks": sum(r["n_chunks"] for r in results) / n,
        "avg_retry_multiplier": sum(
            r["retry_multiplier"] or 1.0 for r in results
        )
        / n,
        "instances": results,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kg-progress", required=True)
    parser.add_argument("--skip-dev", type=int, default=23)
    parser.add_argument("--bm25-dir", default=None, help="HF dataset dir")
    parser.add_argument("--bm25-jsonl", default=None, help="JSONL fallback")
    parser.add_argument("--agentless", required=True)
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--tokenizer", default=None, help="tokenizer.json path")
    parser.add_argument(
        "--kg-sr",
        default=None,
        help="sr predictions jsonl (run_repair_inference output); adds the "
        "kg_sr method with solver prompt + completion token counts "
        "(prompts reconstructed as sent: issue/code extraction, "
        "line-number stripping, sr re-wrap)",
    )
    parser.add_argument(
        "--kg-truncated-ids",
        default=None,
        help="file with instance ids (one per line) whose sr prompts were "
        "truncated by truncate_file_sections in the truncfix12 resume run; "
        "applies only to --kg-sr reconstruction",
    )
    parser.add_argument("--output", default=None, help="summary JSON path")
    parser.add_argument(
        "--rerank-dev",
        "--rerank-progress",
        dest="rerank_progress",
        default=None,
        help="rerank progress file; add reconstructed list-rerank costs",
    )
    parser.add_argument(
        "--agentless-dir",
        default=None,
        help="Agentless results root (file_level/, repair/, …); adds the "
        "per-stage Agentless token table from server-reported usage",
    )
    parser.add_argument(
        "--embed-cache",
        default=None,
        help="shared embed cache dir (chroma.sqlite3); adds the separate "
        "embedding-cost block (exact vector count)",
    )
    parser.add_argument(
        "--walk-embed-graphs",
        action="store_true",
        help="with --embed-cache: replay composed node-doc chars over "
        "--graphs-dir for an upper-bound node-side char count",
    )
    parser.add_argument("--graphs-dir", default="outputs/kg_graphs")
    parser.add_argument("--config", default=None, help="kg_config.json path")
    args = parser.parse_args()

    counter = make_counter(args.tokenizer)
    print(f"counter: {counter.describe()}")
    kg_instances = load_kg_rows(args.kg_progress, skip=args.skip_dev)
    print(f"kg instances: {len(kg_instances)} (skipped {args.skip_dev} dev rows)")
    bm25_texts = load_bm25_texts(args.bm25_dir, args.bm25_jsonl)
    agentless_found = load_agentless_found(args.agentless)
    print(f"bm25 texts: {len(bm25_texts)}, agentless rows: {len(agentless_found)}")
    truncated_ids = set()
    if args.kg_truncated_ids:
        with open(args.kg_truncated_ids, encoding="utf-8") as handle:
            truncated_ids = {line.strip() for line in handle if line.strip()}
        print(f"kg truncated ids: {len(truncated_ids)} (--kg-truncated-ids)")
    sr_prompts = (
        load_sr_predictions(
            args.kg_sr, sr_reconstruct=True, truncated_ids=truncated_ids
        )
        if args.kg_sr
        else None
    )
    if sr_prompts is not None:
        print(f"sr predictions: {len(sr_prompts)} rows (--kg-sr)")

    stages = {}
    if args.bm25_jsonl:
        bm25_predictions = load_sr_predictions(args.bm25_jsonl)
        stages["bm25_solve"] = solve_stage_from_predictions(bm25_predictions, counter)
        print(
            f"bm25 solve stage: {stages['bm25_solve']['n']} instances, "
            f"mean input {stages['bm25_solve']['input_mean_tokens']:,.0f} tok"
        )
    if args.agentless_dir:
        agentless_stage_usages = collect_agentless_stages(args.agentless_dir)
        stages["agentless"] = {
            key: usage_stage_summary(usages)
            for key, usages in agentless_stage_usages.items()
        }
        totals = agentless_per_instance_totals(agentless_stage_usages)
        total_stats = token_stats(list(totals.values()))
        stages["agentless_total"] = {"n": len(totals), **{
            f"total_{key}_tokens": value for key, value in total_stats.items()
        }}
        print(
            f"agentless stages: {list(stages['agentless'])}, "
            f"total mean {stages['agentless_total']['total_mean_tokens']:,.0f} tok/inst"
        )
    if sr_prompts is not None:
        stages["kg_solve"] = solve_stage_from_predictions(sr_prompts, counter)
        print(
            f"kg solve stage: {stages['kg_solve']['n']} instances, "
            f"mean input {stages['kg_solve']['input_mean_tokens']:,.0f} tok"
        )

    embed = None
    if args.embed_cache:
        embed = {"model": "nomic-embed-code-Q8_0"}
        embed.update(embed_cache_stats(args.embed_cache))
        embed["node_chars_upper_bound"] = embed["vectors"] * (
            EMBED_DOC_PREFIX_CHARS + EMBED_DOC_CAP_CHARS
        )
        if args.walk_embed_graphs:
            embed["graphs_replay"] = graphs_embed_chars(args.graphs_dir)
            replay = embed["graphs_replay"]
            print(
                f"embed replay: {replay['nodes']:,} nodes / {replay['graphs']:,} graphs, "
                f"{replay['total_chars']:,} chars (upper bound)"
            )
        embed["query_calls_estimate"] = len(kg_instances)
        embed["query_chars_estimate"] = sum(
            embed_query_chars(row["problem_statement"]) for row in kg_instances
        )
        print(f"embed cache: {embed['vectors']:,} vectors (separate model)")

    report = build_report(
        kg_instances,
        bm25_texts,
        agentless_found,
        args.corpus_dir,
        counter,
        sr_prompts=sr_prompts,
        stages=stages or None,
        embed=embed,
    )
    if args.rerank_progress:
        print("reconstructing list-rerank tokens (needs embed endpoint)…")
        report["rerank_list"] = reconstruct_rerank(
            args.kg_progress,
            args.rerank_progress,
            args.graphs_dir,
            args.config,
            counter,
        )
        if report["rerank_list"]:
            stages["kg_rerank"] = report["rerank_list"]
            report["stages"] = stages

    markdown = render_markdown(report)
    print("\n" + markdown)
    if args.output:
        Path(args.output).write_text(
            json.dumps(report, indent=2) + "\n" + markdown
        )
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
