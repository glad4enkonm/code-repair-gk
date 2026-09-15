#!/usr/bin/env python3
"""Node-cover oracle report: can node-level context cover the gold diff?

Two numbers per instance, both computed offline (no LLM, no server):

- Ceiling (structural): each gold-patch hunk is classified against the
  graph nodes of its file — class 1 fully contained in one node span
  (a whole-node rewrite can fix it), class 2 partially overlapping,
  class 3 outside every node (module-level code), "new" for new files.
- Floor (min-token cover): greedy set cover over class-1 hunks — the
  smallest node set (by ``counter`` tokens of ``source_code``) whose
  spans cover every coverable hunk. Class 2/3/"new" hunks are counted
  as ``fallback_hunks`` (they need a SEARCH block or fail).

Everything is computed twice: over ALL graph nodes (oracle) and over
the practical pool (touched nodes within the file-rerank top-k files).

Usage (remote, from the experiment root):

    python scripts/node-cover-report.py \
        --kg-progress <kg_dev23.jsonl> \
        --rerank-progress outputs/llm_rerank__file_level__embed__list__freq-simmax-simmean-parts__dev__qwen3-coder-next-Q8_0.progress.jsonl \
        --graphs-dir outputs/kg_graphs \
        --tokenizer <tokenizer.json> \
        --output outputs/node_cover_report_dev.json
"""

import argparse
import importlib.util
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, "src")

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_FILE_OLD_RE = re.compile(r"^--- ([^\t]+)")
_FILE_NEW_RE = re.compile(r"^\+\+\+ ([^\t]+)")


@dataclass(frozen=True)
class Hunk:
    """Old-file side of a unified-diff hunk (1-based, inclusive).

    ``count == 0`` marks a pure insertion after line ``start``: no old
    lines are touched, and containment means the insertion point lies
    strictly inside the node span.
    """

    file: str
    start: int
    end: int
    count: int
    is_new_file: bool


@dataclass(frozen=True)
class NodeSpan:
    node_id: str
    file: str
    kind: str
    line_start: int
    line_end: int
    source_code: str


@dataclass
class HunkClassification:
    hunk: Hunk
    cls: object  # 1 contained | 2 partial | 3 outside | "new"
    covering: tuple = ()
    overlapping: tuple = ()


# ---------------------------------------------------------------------------
# Patch parsing
# ---------------------------------------------------------------------------


def parse_patch_hunks(patch: str) -> list:
    """Hunks of a unified diff, old-file side only (see ``Hunk``).

    New files (``--- /dev/null``) keep ``is_new_file`` and contribute
    no coverable old lines; deleted files (``+++ /dev/null``) keep the
    old path. Binary sections are skipped.
    """
    hunks = []
    old_path = new_path = None
    binary = False
    for line in patch.splitlines():
        match = _FILE_OLD_RE.match(line)
        if match:
            old_path = match.group(1).strip()
            binary = False
            continue
        match = _FILE_NEW_RE.match(line)
        if match:
            new_path = match.group(1).strip()
            binary = False
            continue
        if line.startswith("GIT binary patch"):
            binary = True
            continue
        if binary:
            continue
        match = _HUNK_RE.match(line)
        if match and old_path and new_path:
            start = int(match.group(1))
            count = int(match.group(2)) if match.group(2) is not None else 1
            if new_path == "/dev/null":
                file_path = old_path.removeprefix("a/")
            else:
                file_path = new_path.removeprefix("b/")
            hunks.append(
                Hunk(
                    file=file_path,
                    start=start,
                    end=start + count - 1 if count else start,
                    count=count,
                    is_new_file=old_path == "/dev/null",
                )
            )
    return hunks


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def node_contains(node: NodeSpan, hunk: Hunk) -> bool:
    """True when a whole-node rewrite covers the hunk completely."""
    if hunk.is_new_file or hunk.file != node.file:
        return False
    if hunk.count == 0:
        return node.line_start <= hunk.start < node.line_end
    return node.line_start <= hunk.start and hunk.end <= node.line_end


def node_overlaps(node: NodeSpan, hunk: Hunk) -> bool:
    """True when the hunk touches the node span at all."""
    if hunk.is_new_file or hunk.file != node.file:
        return False
    if hunk.count == 0:
        return node.line_start <= hunk.start <= node.line_end
    return hunk.start <= node.line_end and hunk.end >= node.line_start


def classify_hunks(hunks: list, nodes: list) -> list:
    """Classify each hunk against the node spans of its file."""
    by_file = defaultdict(list)
    for node in nodes:
        by_file[node.file].append(node)
    result = []
    for hunk in hunks:
        if hunk.is_new_file:
            result.append(HunkClassification(hunk, "new", (), ()))
            continue
        file_nodes = by_file.get(hunk.file, [])
        covering = tuple(
            sorted(n.node_id for n in file_nodes if node_contains(n, hunk))
        )
        if covering:
            result.append(HunkClassification(hunk, 1, covering, ()))
            continue
        overlapping = tuple(
            sorted(n.node_id for n in file_nodes if node_overlaps(n, hunk))
        )
        result.append(
            HunkClassification(hunk, 2 if overlapping else 3, (), overlapping)
        )
    return result


# ---------------------------------------------------------------------------
# Greedy min-token cover
# ---------------------------------------------------------------------------


def min_token_cover(classifications: list, nodes: list, cost_of) -> dict:
    """Smallest-token node set covering every class-1 hunk (greedy).

    Greedy pick: most newly covered hunks per token; ties broken by
    cheaper cost, then node id (deterministic). Hunks of class 2/3/
    "new" are uncoverable by splicing and reported as fallbacks.
    """
    node_by_id = {n.node_id: n for n in nodes}
    covers = defaultdict(set)
    universe = set()
    for index, classification in enumerate(classifications):
        if classification.cls != 1:
            continue
        universe.add(index)
        for node_id in classification.covering:
            covers[node_id].add(index)
    remaining = set(universe)
    chosen = []
    tokens = 0
    while remaining:
        best_key = None
        best_node = None
        best_gain = None
        for node_id, covered in covers.items():
            gain = len(covered & remaining)
            if not gain:
                continue
            cost = cost_of(node_by_id[node_id])
            key = (-(gain / cost), cost, node_id)
            if best_key is None or key < best_key:
                best_key, best_node, best_gain = key, node_id, covered & remaining
        if best_node is None:
            break
        chosen.append(best_node)
        tokens += cost_of(node_by_id[best_node])
        remaining -= best_gain
    return {
        "chosen": chosen,
        "tokens": tokens,
        "covered_hunks": len(universe) - len(remaining),
        "fallback_hunks": sum(1 for c in classifications if c.cls != 1),
    }


def pool_nodes(nodes: list, touched_ids, top_files: list) -> list:
    """Practical candidate pool: touched nodes within the top-k files."""
    top = set(top_files)
    touched = set(touched_ids)
    return [n for n in nodes if n.file in top and n.node_id in touched]


def pool_variants(
    nodes: list, touched_ids, ranked_files: list, file_ks=(1, 3, 5)
) -> dict:
    """Named candidate pools for the ceiling/floor comparison.

    ``file{k}`` — ALL nodes (touched or not) of the rerank top-k files;
    ``touched`` — all touched nodes, no file restriction;
    ``touched_file5`` — the intersection baseline (touched ∩ top-5).
    """
    variants = {}
    for k in sorted(set(file_ks)):
        top = set(ranked_files[:k])
        variants[f"file{k}"] = [n for n in nodes if n.file in top]
    variants["touched"] = [
        n for n in nodes if n.node_id in set(touched_ids)
    ]
    variants["touched_file5"] = pool_nodes(nodes, touched_ids, ranked_files[:5])
    return variants


# ---------------------------------------------------------------------------
# Graph loading
# ---------------------------------------------------------------------------


def graph_dir_for_row(row: dict, graphs_dir: str) -> str:
    """Graph cache dir for a progress row (mirrors build_graphs key)."""
    repo_key = row.get("repo", "").replace("/", "__")
    return str(Path(graphs_dir) / f"{repo_key}__{row.get('base_commit', '')}")


def load_graph_nodes(graph_dir: str, node_paths: dict | None = None) -> list:
    """Function/Method/Class spans with source + line bounds + file.

    ``node_paths`` (node id -> relative path) may be injected for
    tests; by default it is resolved via ``kg.embeddings``.
    """
    data_path = Path(graph_dir) / "data.json"
    if not data_path.exists():
        return []
    with open(data_path) as handle:
        data = json.load(handle)
    if node_paths is None:
        from kg.embeddings import resolve_relative_paths  # noqa: PLC0415

        node_paths = resolve_relative_paths(graph_dir)
    values = data.get("allValues", {})
    spans = []
    for node in data.get("nodes", []):
        node_id = node["id"]
        props = values.get(node_id, {})
        if props.get("meta_type") not in ("Function", "Method", "Class"):
            continue
        source = props.get("source_code")
        line_start = props.get("line_start")
        line_end = props.get("line_end")
        file_path = node_paths.get(node_id, "")
        if not source or line_start is None or line_end is None or not file_path:
            continue
        spans.append(
            NodeSpan(
                node_id=node_id,
                file=file_path,
                kind=props["meta_type"],
                line_start=line_start,
                line_end=line_end,
                source_code=source,
            )
        )
    return spans


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def _sibling(module_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(
        module_name, Path(__file__).parent / file_name
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_report(
    rows: list,
    rerank_ranked: dict,
    graphs_dir: str,
    counter,
    file_ks=(1, 3, 5),
    path_resolver=None,
) -> dict:
    """Per-instance ceiling/floor over oracle nodes and named pools."""
    token_stats = _sibling("tcr_shared", "token-cost-report.py").token_stats
    instances = []
    for row in rows:
        instance_id = row["instance_id"]
        graph_dir = graph_dir_for_row(row, graphs_dir)
        nodes = load_graph_nodes(
            graph_dir,
            node_paths=path_resolver(graph_dir) if path_resolver else None,
        )
        if not nodes and not (Path(graph_dir) / "data.json").exists():
            instances.append(
                {"instance_id": instance_id, "graph_missing": True}
            )
            continue
        hunks = parse_patch_hunks(row.get("patch", ""))
        cost_of = lambda node: counter.count(node.source_code)  # noqa: E731

        oracle = classify_hunks(hunks, nodes)
        ranked = list(rerank_ranked.get(instance_id, []))
        variants = pool_variants(
            nodes, row.get("touched_nodes", []), ranked, file_ks
        )

        gt_ids = {
            n.node_id
            for n in nodes
            if any(node_overlaps(n, h) for h in hunks)
        }
        variant_results = {}
        for name, pool in variants.items():
            pool_ids = {n.node_id for n in pool}
            variant_results[name] = {
                "pool_size": len(pool),
                "gt_nodes_in_pool": len(gt_ids & pool_ids),
                "pool_recall": (
                    len(gt_ids & pool_ids) / len(gt_ids) if gt_ids else None
                ),
                "floor": min_token_cover(
                    classify_hunks(hunks, pool), pool, cost_of
                ),
            }

        old_hunks = [h for h in hunks if not h.is_new_file]
        total_lines = sum(h.count for h in old_hunks)
        class1_lines = sum(
            c.hunk.count for c in oracle if c.cls == 1
        )
        class_counts = defaultdict(int)
        for classification in oracle:
            class_counts[classification.cls] += 1
        instances.append(
            {
                "instance_id": instance_id,
                "n_hunks": len(hunks),
                "class_counts": dict(class_counts),
                "cls1_line_share": (
                    class1_lines / total_lines if total_lines else None
                ),
                "gt_nodes": len(gt_ids),
                "floor_oracle": min_token_cover(oracle, nodes, cost_of),
                "variants": variant_results,
            }
        )

    present = [i for i in instances if not i.get("graph_missing")]
    n_hunks_1 = sum(i["class_counts"].get(1, 0) for i in present)
    n_hunks_23 = sum(
        i["class_counts"].get(cls, 0) for i in present for cls in (2, 3)
    )
    line_shares = [
        i["cls1_line_share"] for i in present if i["cls1_line_share"] is not None
    ]
    floors_oracle = [i["floor_oracle"]["tokens"] for i in present]
    variant_names = [
        k
        for k in (present[0]["variants"] if present else {}).keys()
    ]
    variant_aggregates = {}
    for name in variant_names:
        recalls = [
            i["variants"][name]["pool_recall"]
            for i in present
            if i["variants"][name]["pool_recall"] is not None
        ]
        floors = [i["variants"][name]["floor"]["tokens"] for i in present]
        variant_aggregates[name] = {
            "macro_pool_recall": (
                sum(recalls) / len(recalls) if recalls else None
            ),
            "floor_tokens": token_stats(floors),
            "fully_covered_share": (
                sum(
                    1
                    for i in present
                    if i["variants"][name]["floor"]["fallback_hunks"] == 0
                )
                / len(present)
                if present
                else None
            ),
            "median_pool_size": (
                sorted(i["variants"][name]["pool_size"] for i in present)[
                    len(present) // 2
                ]
                if present
                else None
            ),
        }
    aggregates = {
        "n_instances": len(rows),
        "n_with_graph": len(present),
        "cls1_hunk_share": (
            n_hunks_1 / (n_hunks_1 + n_hunks_23)
            if (n_hunks_1 + n_hunks_23)
            else None
        ),
        "cls1_line_share_macro": (
            sum(line_shares) / len(line_shares) if line_shares else None
        ),
        "floor_oracle_tokens": token_stats(floors_oracle),
        "fully_coverable": (
            sum(1 for i in present if i["floor_oracle"]["fallback_hunks"] == 0)
            / len(present)
            if present
            else None
        ),
        "variants": variant_aggregates,
    }
    return {"aggregates": aggregates, "instances": instances}


def _variant_label(name: str) -> str:
    return "t∩f5" if name == "touched_file5" else name


def render_markdown(report: dict) -> str:
    agg = report["aggregates"]
    variants = list(agg.get("variants", {}))
    lines = [
        "# Node-cover report (oracle ceiling + min-token floor)",
        "",
        f"Instances: {agg['n_instances']} ({agg['n_with_graph']} with graphs)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Hunks fully inside a node (class 1) | {agg['cls1_hunk_share']:.1%} |",
        f"| GT lines inside a node (macro) | {agg['cls1_line_share_macro']:.1%} |",
        f"| Floor tokens oracle — med/mean/p90 | "
        f"{agg['floor_oracle_tokens']['median']:,.0f} / "
        f"{agg['floor_oracle_tokens']['mean']:,.0f} / "
        f"{agg['floor_oracle_tokens']['p90']:,.0f} |",
        f"| Instances fully coverable (no fallback) | {agg['fully_coverable']:.1%} |",
        "",
        "| Pool variant | macro recall | floor med/mean/p90 |"
        " fully covered | median pool size |",
        "|---|---|---|---|---|",
    ]
    for name in variants:
        variant = agg["variants"][name]
        floor = variant["floor_tokens"]
        lines.append(
            f"| {_variant_label(name)} | {variant['macro_pool_recall']:.1%} | "
            f"{floor['median']:,.0f} / {floor['mean']:,.0f} / "
            f"{floor['p90']:,.0f} | "
            f"{variant['fully_covered_share']:.1%} | "
            f"{variant['median_pool_size']:,} |"
        )
    header = (
        "| Instance | hunks | c1/c2/c3/new | GT nodes | oracle |"
        + "".join(f" {_variant_label(name)} |" for name in variants)
    )
    lines += [
        header,
        "|---|---|---|---|---|" + "---|" * len(variants),
    ]
    for instance in report["instances"]:
        if instance.get("graph_missing"):
            lines.append(
                f"| {instance['instance_id']} | graph missing |"
                + " |" * (4 + len(variants))
            )
            continue
        counts = instance["class_counts"]
        cells = [
            f" {instance['floor_oracle']['tokens']:,} |",
        ]
        for name in variants:
            variant = instance["variants"][name]
            recall = variant["pool_recall"]
            cells.append(f" {variant['floor']['tokens']:,}/{recall:.0%} |")
        lines.append(
            f"| {instance['instance_id']} | {instance['n_hunks']} | "
            f"{counts.get(1, 0)}/{counts.get(2, 0)}/{counts.get(3, 0)}/"
            f"{counts.get('new', 0)} | "
            f"{instance['gt_nodes']} |" + "".join(cells)
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kg-progress", required=True)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rerank-progress", required=True)
    parser.add_argument("--graphs-dir", default="outputs/kg_graphs")
    parser.add_argument("--tokenizer", default=None, help="tokenizer.json path")
    parser.add_argument(
        "--file-k",
        type=int,
        nargs="+",
        default=[1, 3, 5],
        help="top-N file counts for the file{k} pool variants",
    )
    parser.add_argument("--output", default=None, help="summary JSON path")
    args = parser.parse_args()

    shared = _sibling("tcr_shared_cli", "token-cost-report.py")
    rows = list(shared.read_jsonl(args.kg_progress))[args.skip :]
    if args.limit is not None:
        rows = rows[: args.limit]
    rerank_ranked = {
        row["instance_id"]: row.get("ranked_files", [])
        for row in shared.read_jsonl(args.rerank_progress)
    }
    counter = shared.make_counter(args.tokenizer)
    print(
        f"instances: {len(rows)}, rerank rows: {len(rerank_ranked)}, "
        f"counter: {counter.describe()}"
    )

    report = build_report(
        rows,
        rerank_ranked,
        args.graphs_dir,
        counter,
        file_ks=tuple(args.file_k),
    )
    markdown = render_markdown(report)
    print("\n" + markdown)
    if args.output:
        Path(args.output).write_text(
            json.dumps(report, indent=2, default=str) + "\n" + markdown
        )
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
