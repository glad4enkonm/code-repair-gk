#!/usr/bin/env python3
"""Re-assemble context from exploration + list-rerank results.

Reads the embed progress JSONL (the source of truth for prompts and file
contents) plus the list-rerank progress JSONL, then rebuilds each
instance's ``<code>`` section with the top-k ranked files in rerank
order. No LLM and no graph access — pure ``text_inputs`` surgery: the
``<issue>`` block, the header and the tail instructions stay verbatim.

Usage:
    python scripts/build-rerank-dataset.py [--split dev|test] [--top-k 5]
        [--model MODEL] [--kg-progress PATH] [--output-root DIR]

``--model`` selects the list-rerank progress to read (must match the
``--model`` used for llm-rerank-files.py) and, when it differs from the
legacy qwen default, is appended to the output names so model-separated
runs never overwrite each other. ``--kg-progress``/``--output-root``
point the source and outputs at a separated tree (e.g. outputs/gemma26b
produced by a config with its own ``output_dir``).

Run from the experiment root. Writes ``<root>/<base>[__model].progress.jsonl``
and the HF dataset dir next to it (schema-identical to the chain output).
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "src")

KG_BASE_NAME = "data__swe-bench_lite__style-3__fs-kg__k-auto"
EMBED_VARIANT = "embed"
# Legacy default chat model: its outputs keep the original untagged names.
DEFAULT_MODEL = "qwen3-coder-next-Q8_0"
RERANK_STEM = (
    "llm_rerank__file_level__embed__list__freq-simmax-simmean-parts"
    "__{split}__{model}"
)
# The chain writes the 23 dev instances first, then the 300 test ones.
DEV_ROW_COUNT = 23

_BLOCK_RE = re.compile(r"\[start of (.+?)\]\n(.*?)\n\[end of \1\]", re.DOTALL)


def parse_blocks(text):
    """Split ``text`` into header/footer/separator and verbatim blocks.

    Returns ``(header, footer, separator, blocks, order)`` where ``blocks``
    maps file path -> the exact ``[start of p] ... [end of p]`` slice and
    ``order`` is the original block order (first occurrence wins on
    duplicate paths). With no blocks, the whole text is the header.
    """
    spans = [(m.group(1), m.start(), m.end()) for m in _BLOCK_RE.finditer(text)]
    if not spans:
        return text, "", "\n", {}, []
    header = text[: spans[0][1]]
    footer = text[spans[-1][2] :]
    if len(spans) > 1:
        separator = text[spans[0][2] : spans[1][1]]
    else:
        separator = "\n"
    blocks = {}
    order = []
    for path, start, end in spans:
        if path not in blocks:
            blocks[path] = text[start:end]
            order.append(path)
    return header, footer, separator, blocks, order


def rebuild_context(text, ranked_files, top_k):
    """Rebuild the code section from a ranked file list.

    Returns ``(new_text, used_paths, missing_paths)``. Ranked paths with
    no matching block are reported in ``missing_paths`` and skipped; if
    nothing is usable the original text is returned unchanged.
    """
    header, footer, separator, blocks, _order = parse_blocks(text)
    wanted = list(dict.fromkeys(ranked_files or []))[:top_k]
    used = []
    missing = []
    for path in wanted:
        if path in blocks:
            used.append(path)
        else:
            missing.append(path)
    if not used:
        return text, [], missing
    body = separator.join(blocks[path] for path in used)
    return header + body + footer, used, missing


def load_progress(path, skip_failed=False):
    """Load a progress JSONL into ``{instance_id: row}`` (last write wins)."""
    rows = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if skip_failed and row.get("failed"):
                continue
            rows[row["instance_id"]] = row
    return rows


def split_rows(rows, split):
    """Select dev (first 23 written) or test (the rest) progress rows."""
    ordered = list(rows.values())
    if split == "dev":
        return ordered[:DEV_ROW_COUNT]
    return ordered[DEV_ROW_COUNT:]


def output_names(split, top_k, model=DEFAULT_MODEL, output_root="outputs"):
    """Return (progress file, dataset dir) for the rerank{k} variant.

    The legacy default model keeps the original untagged names; any other
    model is appended (``...__rerank5__<model>``) so separated runs never
    overwrite each other's datasets.
    """
    base = f"{KG_BASE_NAME}__{EMBED_VARIANT}__rerank{top_k}"
    if model != DEFAULT_MODEL:
        base += f"__{model}"
    root = Path(output_root)
    return root / f"{base}.progress.jsonl", root / base


def rerank_progress_path(split, model=DEFAULT_MODEL, output_root="outputs"):
    """Progress file of the list rerank for ``split`` and ``model``."""
    name = f"{RERANK_STEM.format(split=split, model=model)}.progress.jsonl"
    return Path(output_root) / name


def build_entries(kg_rows, rerank_rows, top_k):
    """Rebuild text_inputs for KG rows using rerank rankings.

    Instances without a usable rerank row keep their original context and
    are marked ``rerank_used: False`` so the dataset stays complete.
    """
    entries = []
    rebuilt = 0
    fallback = 0
    total_missing = 0
    for row in kg_rows:
        entry = dict(row)
        ranked = rerank_rows.get(entry["instance_id"], {}).get("ranked_files")
        if ranked is None:
            entry["rerank_used"] = False
            fallback += 1
            entries.append(entry)
            continue
        new_text, used, missing = rebuild_context(
            entry.get("text_inputs", ""), ranked, top_k
        )
        total_missing += len(missing)
        entry["text_inputs"] = new_text
        entry["rerank_used"] = bool(used)
        if used:
            rebuilt += 1
        else:
            fallback += 1
        entries.append(entry)
    stats = {
        "rebuilt": rebuilt,
        "fallback": fallback,
        "missing_blocks": total_missing,
    }
    return entries, stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--top-k", type=int, default=5, dest="top_k")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "Chat model whose list-rerank progress to read (must match "
            "llm-rerank-files.py --model); non-default models get their "
            f"name appended to the outputs. Default: {DEFAULT_MODEL}"
        ),
    )
    parser.add_argument(
        "--kg-progress",
        default=None,
        help=(
            "Source KG embed progress JSONL override (default: "
            "outputs/<base>__embed.progress.jsonl)"
        ),
    )
    parser.add_argument(
        "--output-root",
        default="outputs",
        help="Root dir for the rerank progress input and both outputs "
        "(e.g. outputs/gemma26b for a model-separated tree)",
    )
    args = parser.parse_args(argv)

    src_progress = (
        Path(args.kg_progress)
        if args.kg_progress
        else Path(f"outputs/{KG_BASE_NAME}__{EMBED_VARIANT}.progress.jsonl")
    )
    rerank_progress = rerank_progress_path(
        args.split, model=args.model, output_root=args.output_root
    )
    out_progress, out_dir = output_names(
        args.split, args.top_k, model=args.model, output_root=args.output_root
    )

    for label, path in (("source", src_progress), ("rerank", rerank_progress)):
        if not path.exists():
            parser.error(f"{label} progress not found: {path}")

    kg_rows = split_rows(load_progress(src_progress), args.split)
    rerank_rows = load_progress(rerank_progress, skip_failed=True)
    entries, stats = build_entries(kg_rows, rerank_rows, args.top_k)

    print(f"Split {args.split}: {len(entries)} instances, top-{args.top_k}")
    print(
        f"Rebuilt: {stats['rebuilt']}, fallback: {stats['fallback']}, "
        f"ranked paths without blocks: {stats['missing_blocks']}"
    )
    print(f"Output: {out_progress}")

    with open(out_progress, "w", encoding="utf-8") as handle:
        for entry in entries:
            print(json.dumps(entry, ensure_ascii=False), file=handle)

    from kg.run import _convert_to_dataset

    _convert_to_dataset(out_progress, out_dir, args.split)
    print(f"Done. Dataset: {out_dir}")


if __name__ == "__main__":
    main()
