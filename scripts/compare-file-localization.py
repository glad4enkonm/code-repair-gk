#!/usr/bin/env python3
"""Compare file-level localization across BM25, Agentless, and KG.

Computes Recall@1/3/5 and Full Coverage@5 for each method on the
SWE-bench Lite split (dev by default, switchable to test).

Data sources (by default under artifacts/, see artifacts/README.md;
override the root with GK_ARTIFACTS_DIR):
  - BM25:      artifacts/baselines/bm25_file_name_and_contents.retrieval.jsonl
  - Agentless: artifacts/baselines/agentless_found_files_{dev|test}.jsonl
  - KG:        artifacts/localization/kg__k-auto__[variant__][mode].progress.jsonl

Metrics:
  - Recall@k   = |GT ∩ top-k predicted| / |GT|   (k = 1, 3, 5)
  - Cov@5      = all GT files in top-5?           (binary, averaged)
  - KG ceiling = unbounded recall (all KG files, labeled non-comparable)

Usage:
    python scripts/compare-file-localization.py [--split dev|test] [--out results.csv]
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Default data paths (repository artifacts layout, overridable)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_ARTIFACTS = Path(os.environ.get("GK_ARTIFACTS_DIR", _REPO_ROOT / "artifacts"))

BM25_RETRIEVAL = str(_ARTIFACTS / "baselines" / "bm25_file_name_and_contents.retrieval.jsonl")
AGENTLESS_PATHS = {
    "dev": str(_ARTIFACTS / "baselines" / "agentless_found_files_dev.jsonl"),
    "test": str(_ARTIFACTS / "baselines" / "agentless_found_files_test.jsonl"),
}
KG_BASE_NAME = "kg__k-auto"


def kg_progress_path(mode: str, variant: str = "plain") -> str:
    """Build the KG progress JSONL path for a context mode and run variant.

    Variants: ``plain`` (classic exploration) and ``embed``
    (embedding-augmented, produced by ``kg.run --embeddings``).
    Suffix order: variant first, then context mode — matches run.py's
    output naming (``...k-auto__embed``) and rerun-context.py's mode
    suffixing (``...k-auto__embed__file_snippet``).
    """
    name = KG_BASE_NAME
    if variant != "plain":
        name += f"__{variant}"
    if mode != "file_level":
        name += f"__{mode}"
    return str(_ARTIFACTS / "localization" / f"{name}.progress.jsonl")


KG_PROGRESS_PATHS = {
    "file_level": kg_progress_path("file_level"),
    "file_snippet": kg_progress_path("file_snippet"),
    "node_source": kg_progress_path("node_source"),
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_START_MARKER = re.compile(r"\[start of (.+?)\]")
_GT_MARKER = re.compile(r"(?m)^\+\+\+ b/(.+)$")
_KS = (1, 3, 5)
_K_COVERAGE = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _is_readme(path: str) -> bool:
    return Path(path).name.lower().startswith("readme")


def gt_files_from_patch(patch: str) -> set[str]:
    return set(_GT_MARKER.findall(patch))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def recall_at_k(predicted: list[str], gt: set[str], k: int) -> float:
    if not gt:
        return 0.0
    return len(set(predicted[:k]) & gt) / len(gt)


def full_coverage_at_k(predicted: list[str], gt: set[str], k: int) -> bool:
    if not gt:
        return False
    return gt.issubset(predicted[:k])


def unbounded_recall(predicted: list[str], gt: set[str]) -> float:
    if not gt:
        return 0.0
    return len(set(predicted) & gt) / len(gt)


# ---------------------------------------------------------------------------
# Loaders — each returns {instance_id: [file_path, ...]} ranked best-first
# ---------------------------------------------------------------------------
def _read_jsonl(path: str):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_bm25(path: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for row in _read_jsonl(path):
        hits = sorted(row["hits"], key=lambda h: h["score"], reverse=True)
        out[row["instance_id"]] = [
            h["docid"] for h in hits if not _is_readme(h["docid"])
        ]
    return out


def load_agentless(path: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for row in _read_jsonl(path):
        out[row["instance_id"]] = [
            f for f in row.get("found_files", []) if not _is_readme(f)
        ]
    return out


def load_kg(path: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for row in _read_jsonl(path):
        files = _START_MARKER.findall(row.get("text_inputs", ""))
        out[row["instance_id"]] = [f for f in files if not _is_readme(f)]
    return out


def load_gt(path: str) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for row in _read_jsonl(path):
        gt = gt_files_from_patch(row.get("patch", ""))
        if gt:
            out[row["instance_id"]] = gt
    return out


# ---------------------------------------------------------------------------
# Split identification
# ---------------------------------------------------------------------------
def get_split_ids(split: str) -> set[str]:
    if split not in AGENTLESS_PATHS:
        raise ValueError(f"Unknown split: {split}")
    return set(load_agentless(AGENTLESS_PATHS[split]).keys())


# ---------------------------------------------------------------------------
# Per-method evaluation
# ---------------------------------------------------------------------------
def evaluate_method(
    predicted_by_id: dict[str, list[str]],
    gt_by_id: dict[str, set[str]],
    instance_ids: list[str],
    method_label: str,
) -> list[dict]:
    rows = []
    for iid in instance_ids:
        predicted = predicted_by_id.get(iid, [])
        gt = gt_by_id.get(iid, set())
        missed = sorted(gt - set(predicted[:_K_COVERAGE]))
        row = {
            "instance_id": iid,
            "method": method_label,
            "n_predicted": len(predicted),
            "n_gt": len(gt),
            "recall@1": recall_at_k(predicted, gt, 1),
            "recall@3": recall_at_k(predicted, gt, 3),
            "recall@5": recall_at_k(predicted, gt, 5),
            "cov@5": int(full_coverage_at_k(predicted, gt, _K_COVERAGE)),
            "unbounded_recall": unbounded_recall(predicted, gt),
            "missed_gt_files": "; ".join(missed),
        }
        rows.append(row)
    return rows


def aggregate(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {}
    n = len(rows)
    return {
        "recall@1": sum(r["recall@1"] for r in rows) / n,
        "recall@3": sum(r["recall@3"] for r in rows) / n,
        "recall@5": sum(r["recall@5"] for r in rows) / n,
        "cov@5": sum(r["cov@5"] for r in rows) / n,
        "unbounded_recall": sum(r["unbounded_recall"] for r in rows) / n,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def print_summary(summary: dict[str, dict[str, float]], n_instances: int):
    header = f"{'Method':<22} {'Recall@1':>9} {'Recall@3':>9} {'Recall@5':>9} {'Cov@5':>7} {'KG ceil':>9}"
    print(f"\nFile-localization comparison ({n_instances} instances)\n")
    print(header)
    print("-" * len(header))
    for method in summary:
        s = summary[method]
        ceil = f"{s['unbounded_recall']:.1%}" if method.startswith("kg_") else "—"
        print(
            f"{method:<22} {s['recall@1']:>8.1%} {s['recall@3']:>8.1%} "
            f"{s['recall@5']:>8.1%} {s['cov@5']:>6.1%} {ceil:>9}"
        )
    print()
    print(
        "KG ceil = unbounded recall over ALL surfaced files (non-comparable, labeled reference)"
    )


def write_csv(all_rows: list[dict], path: str):
    fieldnames = [
        "instance_id",
        "method",
        "n_predicted",
        "n_gt",
        "recall@1",
        "recall@3",
        "recall@5",
        "cov@5",
        "unbounded_recall",
        "missed_gt_files",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nPer-instance CSV written to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compare file-level localization: BM25 vs Agentless vs KG"
    )
    parser.add_argument(
        "--split",
        choices=["dev", "test"],
        default="dev",
        help="SWE-bench Lite split (default: dev)",
    )
    parser.add_argument(
        "--variant",
        choices=["plain", "embed"],
        default="plain",
        help="KG run variant (default: plain; embed = embedding-augmented)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="CSV output path (default: none, stdout only)",
    )
    args = parser.parse_args()

    # --- Identify instances for this split ---
    split_ids = get_split_ids(args.split)
    print(
        f"Split '{args.split}': {len(split_ids)} instances (from Agentless {args.split} file)"
    )

    # --- Load GT (from the variant's file_level progress) ---
    gt_by_id = load_gt(kg_progress_path("file_level", args.variant))
    instance_ids = sorted(i for i in split_ids if i in gt_by_id)
    missing_gt = split_ids - set(instance_ids)
    if missing_gt:
        print(
            f"WARNING: {len(missing_gt)} instances missing GT, excluded: {sorted(missing_gt)}"
        )
    print(f"Evaluating {len(instance_ids)} instances with GT")

    # --- Load each method ---
    methods_data: dict[str, dict[str, list[str]]] = {}
    methods_data["bm25"] = load_bm25(BM25_RETRIEVAL)
    methods_data["agentless"] = load_agentless(AGENTLESS_PATHS[args.split])
    for mode in ("file_level", "file_snippet", "node_source"):
        progress = kg_progress_path(mode, args.variant)
        if not os.path.exists(progress):
            print(f"WARNING: kg progress file missing, skipping: {progress}")
            continue
        method = f"kg_{mode}"
        if args.variant != "plain":
            method = f"kg_{args.variant}_{mode}"
        methods_data[method] = load_kg(progress)

    # --- Check coverage ---
    for method, data in methods_data.items():
        present = sum(1 for i in instance_ids if i in data)
        if present < len(instance_ids):
            print(f"WARNING: {method} covers {present}/{len(instance_ids)} instances")

    # --- Evaluate ---
    all_rows: list[dict] = []
    summary: dict[str, dict[str, float]] = {}
    for method in methods_data:
        rows = evaluate_method(methods_data[method], gt_by_id, instance_ids, method)
        all_rows.extend(rows)
        summary[method] = aggregate(rows)

    # --- Output ---
    print_summary(summary, len(instance_ids))
    if args.out:
        write_csv(all_rows, args.out)


if __name__ == "__main__":
    sys.exit(main())
