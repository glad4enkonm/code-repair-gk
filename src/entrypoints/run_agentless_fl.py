#!/usr/bin/env python3
"""
Agentless fault localization entry point — delegates to the Qwen-adapted fork.

Usage (file-level):
    python -m entrypoints.run_agentless_fl --file_level \
        --dataset ../data/swe-bench_lite --split dev \
        --backend openai --model qwen3-coder-next-Q8_0 \
        --output_folder results/file_level --skip_existing

Usage (related_level, fine_grain_line_level, merge):
    Same as agentless.fl.localize -- all flags are passed through.
    See agentless-minimal.sh for the full pipeline.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentless.fl.localize import main

if __name__ == "__main__":
    main()
