#!/usr/bin/env python3
"""Oracle dataset builder: GT-file contexts for repair-ceiling measurement.

Reads SWE-bench_Lite, extracts ground-truth file paths from each gold
patch, assembles a style-3 prompt with exactly those files (oracle
localization), and writes a progress JSONL + HF dataset.

Output is schema-compatible with the KG/BM25 datasets, so
``run_inference.py`` (diff) and ``run_repair_inference.py``
(sr/node/tool) work on it unchanged.

Usage::

    python src/entrypoints/run_oracle_dataset.py --split dev
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

# Add src/ to path for direct script execution
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root / "src"))

from datasets import load_dataset
from tqdm.auto import tqdm

from kg.config import KGConfig
from kg.oracle import assemble_oracle_context
from kg.run import _convert_to_dataset, _load_existing


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument(
        "--config",
        type=str,
        default="config/kg_config.json",
        help="Path to kg_config.json",
    )
    return parser


def main():
    args = build_parser().parse_args()
    config = KGConfig.from_file(args.config)

    instances = load_dataset("princeton-nlp/SWE-bench_Lite", split=args.split)
    print(f"Dataset: princeton-nlp/SWE-bench_Lite/{args.split} — {len(instances)}")

    output_name = "data__swe-bench_lite__style-3__fs-oracle__k-gt"
    output_dir = Path(config.output_dir) / output_name
    progress_file = Path(str(output_dir) + f".{args.split}.progress.jsonl")
    progress_file.parent.mkdir(parents=True, exist_ok=True)

    existing = _load_existing(progress_file)
    if existing:
        print(f"Resuming: {len(existing)} already processed")

    for instance in tqdm(instances, desc=f"Oracle context ({args.split})"):
        instance_id = instance["instance_id"]
        if instance_id in existing:
            continue

        repo = instance["repo"]
        base_commit = instance["base_commit"]
        problem_statement = instance["problem_statement"]
        patch = instance.get("patch", "")

        print(f"\n[{instance_id}] {repo}", flush=True)

        try:
            oracle = assemble_oracle_context(
                repo, base_commit, problem_statement, patch, config
            )
            if oracle.dropped_files:
                print(
                    f"  WARNING: dropped GT files: {oracle.dropped_files}",
                    flush=True,
                )

            entry = {
                "instance_id": instance_id,
                "repo": repo,
                "base_commit": base_commit,
                "problem_statement": problem_statement,
                "patch": patch,
                "test_patch": instance.get("test_patch", ""),
                "hints_text": instance.get("hints_text", ""),
                "created_at": instance.get("created_at", ""),
                "version": instance.get("version", ""),
                "FAIL_TO_PASS": instance.get("FAIL_TO_PASS", ""),
                "PASS_TO_PASS": instance.get("PASS_TO_PASS", ""),
                "environment_setup_commit": instance.get(
                    "environment_setup_commit", ""
                ),
                "text_inputs": oracle.text,
                "touched_nodes": oracle.gt_files,
                "gt_files": oracle.gt_files,
                "gt_files_dropped": oracle.dropped_files,
            }

            with open(progress_file, "a") as f:
                print(json.dumps(entry, ensure_ascii=False), file=f, flush=True)
            existing.add(instance_id)

            print(f"  context: {len(oracle.text):,} chars", flush=True)

        except Exception:
            print(f"  FAILED: {instance_id}", flush=True)
            traceback.print_exc()

    _convert_to_dataset(progress_file, output_dir, args.split)

    print(f"\nDone. Dataset saved to {output_dir}")


if __name__ == "__main__":
    main()
