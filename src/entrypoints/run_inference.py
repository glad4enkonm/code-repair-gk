#!/usr/bin/env python3
"""
SWE-bench inference entry point with Qwen model override.

Applies runtime overrides to register qwen3-coder-next-Q8_0, then delegates
to the upstream swebench inference main() with pre-parsed arguments
(bypasses argparse choices validation).

Usage:
    python3 src/entrypoints/run_inference.py                          \
        --dataset_name_or_path outputs/data__...__fs-kg__k-auto       \
        --split test                                                   \
        --model_name_or_path qwen3-coder-next-Q8_0                      \
        --output_dir outputs                                           \
        --model_args "max_tokens=8192,temperature=0.0"
"""

import argparse
import sys
from pathlib import Path

# Add src/ to path for direct script execution (pip install -e . not required in dev)
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root / "src"))

from entrypoints.swebench_override import apply_swebench_overrides

apply_swebench_overrides()

from swebench.inference.run_api import main as swebench_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        required=True,
        help="HuggingFace dataset name or local path",
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="qwen3-coder-next-Q8_0",
    )
    parser.add_argument("--shard_id", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--model_args",
        type=str,
        default=None,
        help="key=value,key=value (e.g. 'max_tokens=8192,temperature=0.0')",
    )
    parser.add_argument("--max_cost", type=float, default=None)
    return parser


def main():
    args = build_parser().parse_args()
    swebench_main(
        dataset_name_or_path=args.dataset_name_or_path,
        split=args.split,
        model_name_or_path=args.model_name_or_path,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        output_dir=args.output_dir,
        model_args=args.model_args,
        max_cost=args.max_cost,
    )


if __name__ == "__main__":
    main()
