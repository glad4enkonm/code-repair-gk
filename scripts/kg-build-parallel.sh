#!/bin/bash
# Launch parallel KG build workers (one nohup process per worker).
# Usage: bash scripts/kg-build-parallel.sh [num_workers] [config_path] [split]
set -e

NUM_WORKERS="${1:-12}"
CONFIG="${2:-config/kg_config.json}"
SPLIT="${3:-test}"

mkdir -p logs

for i in $(seq 0 $((NUM_WORKERS - 1))); do
    nohup bash scripts/kg-build.sh "$CONFIG" "$SPLIT" $i $NUM_WORKERS \
        > "logs/kg-build-${SPLIT}-w${i}.log" 2>&1 &
    echo "Worker $i: PID $!"
done

echo "Launched $NUM_WORKERS parallel workers."
echo "Monitor: grep -c 'Graph built' logs/kg-build-${SPLIT}-w*.log"
