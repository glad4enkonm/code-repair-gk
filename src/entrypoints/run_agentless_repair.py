#!/usr/bin/env python3
"""
Agentless repair entry point — delegates to the Qwen-adapted fork.

Usage:
    python -m entrypoints.run_agentless_repair \
        --loc_file results/edit_location_individual/loc_merged_0-0_outputs.jsonl \
        --output_folder results/repair --dataset ../data/swe-bench_lite --split dev \
        --loc_interval --top_n 3 --context_window 10 --max_samples 5 \
        --cot --diff_format --gen_and_process \
        --backend openai --model qwen3-coder-next-Q8_0 --num_threads 2

All flags are passed through to agentless.repair.repair:main.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentless.repair.repair import main

if __name__ == "__main__":
    main()
