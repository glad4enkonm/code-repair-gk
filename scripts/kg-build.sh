#!/bin/bash
# Stage K1: Build knowledge graphs for all instances in a dataset split.
# Usage: bash scripts/kg-build.sh [config_path] [split] [worker_id] [num_workers]
set -e

CONFIG="${1:-config/kg_config.json}"
SPLIT="${2:-test}"
WORKER_ID="${3:-0}"
NUM_WORKERS="${4:-1}"

export PYTHONPATH="$(dirname "$0")/../src:$PYTHONPATH"

python3 -c "
import sys
sys.path.insert(0, 'src')
from kg.config import KGConfig
from kg.build_graphs import build_all
from datasets import load_dataset

config = KGConfig.from_file('${CONFIG}')
config.split = '${SPLIT}'
dataset = load_dataset('princeton-nlp/SWE-bench_Lite', split=config.split)
build_all(dataset, config, worker_id=${WORKER_ID}, num_workers=${NUM_WORKERS})
"
echo "=== Graphs built ==="
