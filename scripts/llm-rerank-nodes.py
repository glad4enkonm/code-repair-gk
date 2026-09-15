#!/usr/bin/env python3
"""Re-rank graph nodes per SWE-bench instance with one listwise LLM call.

Candidates: ALL graph nodes (Function/Method/Class with source and line
bounds) of the file-rerank top-k files — the pool with 91.3% GT-node
recall on dev, vs 68.8% for touched-only (see node-cover-report.py).
Signals per node: touch frequency, chroma similarity to the issue,
kind/name/line-span/file path, and a source head snippet.

Answer: a strict JSON array of node ids (top_k, default 10). Candidate
lists longer than --max-prompt-chars are ranked chunk by chunk and
merged round-robin; retry ladder identical to llm-rerank-files.py
(no-think fallback, then resample).

Progress is checkpointed per instance to a JSONL file (resume-safe;
failed rows count as done). Evaluation reports node-level Recall@k /
Cover@k against GT nodes (nodes overlapping gold-patch hunks).

Usage:
    python scripts/llm-rerank-nodes.py --split dev --dry-run
    python scripts/llm-rerank-nodes.py --split dev --limit 3
    python scripts/llm-rerank-nodes.py --split dev --eval-only
"""

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_sibling(file_name: str, module_name: str):
    spec = importlib.util.spec_from_file_location(
        module_name, Path(__file__).resolve().parent / file_name
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lrf = _load_sibling("llm-rerank-files.py", "lrf_nodes_shared")
ncr = _load_sibling("node-cover-report.py", "ncr_nodes_shared")

_NODE_SIGNAL_ORDER = ("freq", "sim")


# ---------------------------------------------------------------------------
# Signal spec
# ---------------------------------------------------------------------------


def parse_node_signals(spec: str) -> list:
    """Parse a comma-separated subset of freq,sim (canonical order)."""
    tokens = [t.strip() for t in (spec or "").split(",") if t.strip()]
    if not tokens:
        return list(_NODE_SIGNAL_ORDER)
    expanded = set()
    for token in tokens:
        if token not in _NODE_SIGNAL_ORDER:
            raise ValueError(
                f"unknown signal '{token}' (valid: {', '.join(_NODE_SIGNAL_ORDER)})"
            )
        expanded.add(token)
    return [name for name in _NODE_SIGNAL_ORDER if name in expanded]


def node_signals_tag(signals: list) -> str:
    """Filesystem-safe tag for a node-signal subset."""
    return "-".join(signals) or "none"


# ---------------------------------------------------------------------------
# Signal assembly
# ---------------------------------------------------------------------------


def assemble_node_signals(
    row: dict,
    graph_dir: str,
    top_files: list,
    src_chars: int = 120,
    sim_provider=None,
    nodes: list | None = None,
) -> list:
    """Per-node signal dicts for the candidate pool, freq-ordered.

    ``nodes`` may be injected (tests); by default all graph nodes are
    loaded and filtered to ``.py`` files within ``top_files``.
    ``sim_provider(problem_statement, graph_dir) -> {node_id: sim}``
    supplies node similarities; ``None`` omits the sim fields.
    """
    if nodes is None:
        nodes = ncr.load_graph_nodes(graph_dir)
    top = set(top_files)
    touched = row.get("touched_nodes", []) or []
    freq: dict = {}
    for node_id in touched:
        freq[node_id] = freq.get(node_id, 0) + 1

    node_sims = {}
    if sim_provider is not None:
        try:
            node_sims = (
                sim_provider(row.get("problem_statement", ""), graph_dir) or {}
            )
        except Exception as exc:
            logger.warning(
                "sim provider failed for %s — sim signals omitted: %s",
                row.get("instance_id", "<unknown>"),
                exc,
            )
            node_sims = {}

    signals = []
    for node in nodes:
        if not node.file.endswith(".py") or node.file not in top:
            continue
        entry = {
            "node_id": node.node_id,
            "path": node.file,
            "kind": node.kind,
            "name": node.node_id.rsplit("-", 1)[-1],
            "line_start": node.line_start,
            "line_end": node.line_end,
            "touch_freq": freq.get(node.node_id, 0),
            "head": (node.source_code or "").strip()[:src_chars],
        }
        if node.node_id in node_sims:
            entry["sim"] = node_sims[node.node_id]
        signals.append(entry)
    signals.sort(key=lambda s: (-s["touch_freq"], s["node_id"]))
    return signals


# ---------------------------------------------------------------------------
# Listwise prompt
# ---------------------------------------------------------------------------

_NODE_LIST_TEMPLATE = """\
You are localizing a bug fix in a repository. Given the issue and the
candidate code elements (functions, methods, classes) of the most
relevant files, rank the elements by how likely the fix must change
them.

Rules:
- Use the per-element signals (touch frequency, similarity, span,
  source head) as evidence.
- Return ONLY the top {top_k} element ids, most relevant first.
- Answer with a strict JSON array of element ids from the candidate
  list. No explanations, no other text.

ISSUE:
{issue}

CANDIDATE ELEMENTS ({n_nodes} total, ordered by touch frequency):

{node_blocks}

Answer: a JSON array of the top {top_k} element ids, e.g. \
["Function-src/lib/foo.py-parse_args"]"""


def _render_node_block(node_signal: dict, signals: list) -> str:
    """Render one candidate's signal block for the listwise prompt."""
    lines = [f"### {node_signal['node_id']}"]
    header = [
        f"kind: {node_signal['kind']}",
        f"name: {node_signal['name']}",
        f"L{node_signal['line_start']}-L{node_signal['line_end']}",
        f"path: {node_signal['path']}",
    ]
    lines.append(" | ".join(header))
    metric_line = []
    if "freq" in signals:
        metric_line.append(f"touch_freq: {node_signal.get('touch_freq', 0)}")
    if "sim" in signals and "sim" in node_signal:
        metric_line.append(f"sim: {node_signal['sim']:.4f}")
    if metric_line:
        lines.append(" | ".join(metric_line))
    if node_signal.get("head"):
        lines.append(f"src: {node_signal['head'].replace(chr(10), ' | ')}")
    return "\n".join(lines)


def build_node_list_prompt(
    issue: str, node_signals: list, signals: list, top_k: int
) -> str:
    """Render the listwise node-ranking prompt."""
    blocks = [_render_node_block(ns, signals) for ns in node_signals]
    return _NODE_LIST_TEMPLATE.format(
        top_k=top_k,
        issue=issue,
        n_nodes=len(node_signals),
        node_blocks="\n\n".join(blocks),
    )


def _chunk_node_signals(
    issue: str,
    node_signals: list,
    signals: list,
    top_k: int,
    max_prompt_chars: int,
) -> list:
    """Split freq-ordered signals into chunks whose prompts fit the cap."""
    overhead = len(build_node_list_prompt(issue, [], signals, top_k))
    chunks = []
    current = []
    used = overhead
    for node_signal in node_signals:
        block_len = len(_render_node_block(node_signal, signals)) + 2
        if current and used + block_len > max_prompt_chars:
            chunks.append(current)
            current = []
            used = overhead
        current.append(node_signal)
        used += block_len
    if current:
        chunks.append(current)
    return chunks


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def rank_nodes_list(
    issue: str,
    node_signals: list,
    top_k: int,
    engine,
    signals: list,
    max_prompt_chars: int = 0,
) -> dict:
    """One-shot listwise node ranking; on failure keep frequency order.

    Mirrors llm-rerank-files.rank_files_list: chunked prompts merged
    round-robin, frequency tail pads the ranking so every candidate
    stays represented, failure only when every chunk fails.
    """
    freq_order = [ns["node_id"] for ns in node_signals]
    if len(node_signals) <= 1:
        return {"ranked": freq_order, "failed": False, "n_calls": 0}

    chunks = (
        _chunk_node_signals(issue, node_signals, signals, top_k, max_prompt_chars)
        if max_prompt_chars > 0
        else [node_signals]
    )
    winner_lists = []
    for chunk in chunks:
        prompt = build_node_list_prompt(issue, chunk, signals, top_k)
        answer = engine.ask_ranking(prompt)
        if answer is None:
            continue
        chunk_ids = {ns["node_id"] for ns in chunk}
        winners = []
        for node_id in answer:
            if node_id in chunk_ids and node_id not in winners:
                winners.append(node_id)
        winner_lists.append(winners)

    if not winner_lists:
        return {
            "ranked": freq_order,
            "failed": True,
            "n_calls": engine.n_llm_calls,
        }

    ranked = []
    for position in range(max(len(w) for w in winner_lists)):
        for winners in winner_lists:
            if position < len(winners) and winners[position] not in ranked:
                ranked.append(winners[position])
    ranked += [node_id for node_id in freq_order if node_id not in set(ranked)]
    return {"ranked": ranked, "failed": False, "n_calls": engine.n_llm_calls}


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def load_nodes_rerank(path: str) -> dict:
    """Load progress JSONL -> {instance_id: ranked_nodes}."""
    out = {}
    progress = Path(path)
    if not progress.exists():
        return out
    with open(progress) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[row["instance_id"]] = row.get("ranked_nodes", [])
    return out


def default_node_progress_path(
    split: str, model: str, signals: list, file_k: int, top_k: int
) -> str:
    """Default progress file per split, model, and spec (no collisions)."""
    name = lrf._sanitize_model_name(model)
    return str(
        Path(__file__).resolve().parent.parent
        / "outputs"
        / f"node_rerank__list__{node_signals_tag(signals)}"
        f"__f{file_k}__k{top_k}__{split}__{name}.progress.jsonl"
    )


# ---------------------------------------------------------------------------
# GT nodes + evaluation
# ---------------------------------------------------------------------------


def gt_nodes_for_row(row: dict, graph_dir: str, node_paths: dict | None = None):
    """Node ids overlapping any gold-patch hunk (loose GT)."""
    nodes = ncr.load_graph_nodes(graph_dir, node_paths=node_paths)
    hunks = ncr.parse_patch_hunks(row.get("patch", ""))
    return {
        n.node_id for n in nodes if any(ncr.node_overlaps(n, h) for h in hunks)
    }


def evaluate_nodes(rerank: dict, gt_by_instance: dict, ks=(1, 3, 5, 10)) -> dict:
    """Instance-level Recall@k (any GT node in top-k) and Cover@k (all)."""
    with_gt = [
        (iid, gt) for iid, gt in gt_by_instance.items() if gt
    ]
    summary = {
        "n_instances": len(gt_by_instance),
        "n_with_gt": len(with_gt),
        "n_empty_gt": len(gt_by_instance) - len(with_gt),
    }
    for k in ks:
        hits = 0
        covers = 0
        for iid, gt in with_gt:
            top = set(rerank.get(iid, [])[:k])
            if top & gt:
                hits += 1
            if gt <= top:
                covers += 1
        summary[f"recall@{k}"] = hits / len(with_gt) if with_gt else None
        summary[f"cover@{k}"] = covers / len(with_gt) if with_gt else None
    return summary


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------


def process_rows(
    rows: list,
    ranked_files_by_id: dict,
    engine_factory,
    progress_path: str,
    graphs_dir: str,
    file_k: int,
    top_k: int,
    signals: list,
    src_chars: int,
    max_prompt_chars: int,
    sim_provider,
    model: str,
    limit: int = 0,
    dry_run: bool = False,
    node_overrides: dict | None = None,
) -> dict:
    """Rank nodes per instance, checkpointing each result (resume-safe)."""
    done = load_nodes_rerank(progress_path)
    to_process = [r for r in rows if r.get("instance_id") not in done]
    stats = {
        "instances": len(to_process),
        "skipped": len(rows) - len(to_process),
        "processed": 0,
        "failed": 0,
        "comparisons": 0,
    }
    if dry_run:
        return stats
    if limit > 0:
        to_process = to_process[:limit]

    progress = Path(progress_path)
    progress.parent.mkdir(parents=True, exist_ok=True)
    with open(progress, "a") as out:
        for row in to_process:
            iid = row["instance_id"]
            if node_overrides and iid in node_overrides:
                node_signals = node_overrides[iid]
            else:
                graph_dir = ncr.graph_dir_for_row(row, graphs_dir)
                node_signals = assemble_node_signals(
                    row,
                    graph_dir,
                    list(ranked_files_by_id.get(iid, []))[:file_k],
                    src_chars=src_chars,
                    sim_provider=sim_provider,
                )
            result = rank_nodes_list(
                row.get("problem_statement", ""),
                node_signals,
                top_k,
                engine_factory(),
                signals,
                max_prompt_chars,
            )
            record = {
                "instance_id": iid,
                "model": model,
                "ranked_nodes": result["ranked"],
                "failed": result["failed"],
                "n_nodes": len(node_signals),
                "n_comparisons": result["n_calls"],
                "mode": "list",
                "signals": signals,
                "top_k": top_k,
                "file_k": file_k,
                "src_chars": src_chars,
                "max_prompt_chars": max_prompt_chars,
            }
            out.write(json.dumps(record) + "\n")
            out.flush()
            stats["processed"] += 1
            stats["comparisons"] += result["n_calls"]
            if result["failed"]:
                stats["failed"] += 1
            status = "FAILED (kept freq order)" if result["failed"] else "ok"
            print(
                f"[{iid}] {status}: {result['n_calls']} calls, "
                f"{len(result['ranked'])} nodes",
                flush=True,
            )
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-rank graph nodes of the file-rerank top files "
        "with one listwise LLM call per instance"
    )
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument(
        "--variant",
        choices=["plain", "embed"],
        default="embed",
        help="KG run variant (default: embed — the current experiment)",
    )
    parser.add_argument(
        "--file-k",
        type=int,
        default=3,
        help="Candidate pool = ALL nodes of the file-rerank top-K files "
        "(default: 3; 91.3%% GT-node recall on dev)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Answer size: top-N node ids (default: 10)",
    )
    parser.add_argument(
        "--signals",
        default="freq,sim",
        help="Node signals, comma-separated subset of freq,sim "
        "(default: freq,sim)",
    )
    parser.add_argument("--src-chars", type=int, default=120)
    parser.add_argument("--max-prompt-chars", type=int, default=100_000)
    parser.add_argument("--graphs-dir", default=None)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--instance", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--progress", default=None)
    parser.add_argument(
        "--files-rerank",
        default=None,
        help="File-rerank progress JSONL (default: the canonical "
        "freq-simmax-simmean-parts list run for this split+model)",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    defaults = lrf._load_defaults(args.config)
    model = args.model or defaults.get("model", "qwen3-coder-next-Q8_0")
    api_base = (
        args.api_base
        or lrf.os.environ.get("OPENAI_BASE_URL")
        or defaults.get("api_base", "http://localhost:8081/v1")
    )
    api_key = (
        args.api_key
        or lrf.os.environ.get("OPENAI_API_KEY")
        or defaults.get("api_key", "any")
    )
    graphs_dir = args.graphs_dir or defaults.get(
        "graph_cache_dir", "outputs/kg_graphs"
    )
    try:
        signals = parse_node_signals(args.signals)
    except ValueError as exc:
        parser.error(str(exc))
        return 2

    progress_path = args.progress or default_node_progress_path(
        args.split, model, signals, args.file_k, args.top_k
    )

    cfl = lrf.cfl
    kg_file_level_path = cfl.kg_progress_path("file_level", args.variant)
    split_ids = cfl.get_split_ids(args.split)
    gt_by_id = cfl.load_gt(kg_file_level_path)
    rows = [
        r
        for r in cfl._read_jsonl(kg_file_level_path)
        if r["instance_id"] in split_ids and r["instance_id"] in gt_by_id
    ]
    rows = lrf.select_rankable_rows(rows)
    rows = lrf.filter_rows_by_instance(rows, args.instance)
    print(
        f"Split '{args.split}': {len(split_ids)} instances, "
        f"{len(rows)} with KG progress + GT"
    )
    print(f"Model: {model} @ {api_base}")

    files_rerank_path = args.files_rerank or lrf.default_progress_path(
        args.split,
        model,
        args.variant,
        mode="list",
        signals=["freq", "sim_max", "sim_mean", "parts"],
    )
    ranked_files_by_id = lrf.load_rerank(files_rerank_path)
    missing_files = [
        r["instance_id"] for r in rows if r["instance_id"] not in ranked_files_by_id
    ]
    print(
        f"File rerank: {files_rerank_path} "
        f"({len(ranked_files_by_id)} rows; {len(missing_files)} rows missing)"
    )

    sim_provider = None
    if "sim" in signals:
        sim_provider = (
            lambda problem_statement, graph_dir: lrf._node_sims_from_chroma(
                problem_statement, graph_dir, defaults
            )
        )

    if not args.eval_only:
        def factory():
            return lrf.ListRanker(
                client=lrf.build_client(api_base, api_key, args.timeout),
                model=model,
                retries=args.retries,
            )

        stats = process_rows(
            rows,
            ranked_files_by_id,
            factory,
            progress_path,
            graphs_dir,
            file_k=args.file_k,
            top_k=args.top_k,
            signals=signals,
            src_chars=args.src_chars,
            max_prompt_chars=args.max_prompt_chars,
            sim_provider=sim_provider,
            model=model,
            limit=args.limit,
            dry_run=args.dry_run,
        )
        print(
            f"\nRun stats: {stats['processed']}/{stats['instances']} processed "
            f"({stats['skipped']} skipped, {stats['failed']} failed), "
            f"{stats['comparisons']} calls"
        )
        if args.dry_run:
            pending = [
                r for r in rows if r.get("instance_id") not in load_nodes_rerank(
                    progress_path
                )
            ]
            if pending:
                preview_row = pending[0]
                preview_signals = assemble_node_signals(
                    preview_row,
                    ncr.graph_dir_for_row(preview_row, graphs_dir),
                    list(ranked_files_by_id.get(preview_row["instance_id"], []))[
                        : args.file_k
                    ],
                    src_chars=args.src_chars,
                    sim_provider=sim_provider,
                )
                print(
                    f"\n--- list-mode prompt preview "
                    f"({preview_row['instance_id']}, "
                    f"{len(preview_signals)} candidates) ---"
                )
                print(
                    build_node_list_prompt(
                        preview_row.get("problem_statement", ""),
                        preview_signals,
                        signals,
                        args.top_k,
                    )
                )
                print("--- end preview ---")
            return 0

    rerank = load_nodes_rerank(progress_path)
    covered = sum(1 for r in rows if r["instance_id"] in rerank)
    print(f"\nEvaluating: node rerank covers {covered}/{len(rows)} instances")
    gt_by_instance = {
        row["instance_id"]: gt_nodes_for_row(
            row, ncr.graph_dir_for_row(row, graphs_dir)
        )
        for row in rows
    }
    summary = evaluate_nodes(rerank, gt_by_instance, ks=(1, 3, 5, 10))
    print(
        f"Node GT: {summary['n_with_gt']} instances with GT nodes, "
        f"{summary['n_empty_gt']} without (structurally impossible)"
    )
    header = f"{'k':>4} {'Recall@k':>9} {'Cover@k':>9}"
    print(f"\nNode-localization ({summary['n_instances']} instances)\n")
    print(header)
    print("-" * len(header))
    for k in (1, 3, 5, 10):
        print(
            f"{k:>4} {summary[f'recall@{k}']:>9.1%} "
            f"{summary[f'cover@{k}']:>9.1%}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
