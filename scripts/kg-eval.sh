#!/bin/bash
# Stage K3+K4: Run inference + evaluation on KG preprocessed dataset.
# Usage: bash scripts/kg-eval.sh [split] [context_mode] [patch_format]
#   split:        dev (default) | test
#   context_mode: file_level (default) | file_snippet | node_source | oracle
#   patch_format: diff (default) | sr | node | tool
#
# Env overrides (defaults reproduce the original qwen behavior):
#   MODEL        chat model served on OPENAI_BASE_URL
#                (default: qwen3-coder-next-Q8_0)
#   OUTPUT_ROOT  tree for dataset lookup, inference output, predictions
#                (default: outputs; e.g. outputs/gemma26b for a
#                model-separated run)
#   DATASET      full dataset dir override (e.g. the rerank5 dataset;
#                otherwise derived from context_mode under OUTPUT_ROOT)
#   RUN_ID       swebench run id (default: kg-${SPLIT}-${MODE}-${PATCH};
#                make it model-distinct for non-qwen runs)
#   DISABLE_THINKING  set to 1 for reasoning models (gemma): appends
#                --disable-thinking to sr/node inference so reasoning
#                does not eat the completion budget
set -e

SPLIT="${1:-dev}"
MODE="${2:-file_level}"
PATCH="${3:-diff}"
MODEL="${MODEL:-qwen3-coder-next-Q8_0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
DATASET_OVERRIDE="${DATASET:-}"
RUN_ID="${RUN_ID:-kg-${SPLIT}-${MODE}-${PATCH}}"

export PYTHONPATH="$(dirname "$0")/../src:$PYTHONPATH"
export OPENAI_API_KEY=any
export OPENAI_BASE_URL=http://localhost:8081/v1

if [ "$MODE" = "oracle" ]; then
    BASE="data__swe-bench_lite__style-3__fs-oracle__k-gt"
else
    BASE="data__swe-bench_lite__style-3__fs-kg__k-auto"
fi
if [ -n "$DATASET_OVERRIDE" ]; then
    DATASET="$DATASET_OVERRIDE"
elif [ "$MODE" = "file_level" ] || [ "$MODE" = "oracle" ]; then
    DATASET="${OUTPUT_ROOT}/${BASE}"
else
    DATASET="${OUTPUT_ROOT}/${BASE}__${MODE}"
fi

if [ ! -d "$DATASET" ]; then
    echo "Error: preprocessed dataset not found at $DATASET"
    if [ "$MODE" = "oracle" ]; then
        echo "Run src/entrypoints/run_oracle_dataset.py --split $SPLIT first."
    else
        echo "Run kg-run.sh (file_level) or rerun-context.py ($MODE) first."
    fi
    exit 1
fi

echo "=== Stage K3: Running inference ($MODE / $PATCH / $MODEL) ==="
EXTRA_ARGS=()
if [ "${DISABLE_THINKING:-0}" = "1" ]; then
    EXTRA_ARGS+=(--disable-thinking)
fi
if [ "$PATCH" = "diff" ]; then
    python src/entrypoints/run_inference.py \
        --dataset_name_or_path "$DATASET" \
        --split "$SPLIT" \
        --model_name_or_path "$MODEL" \
        --output_dir "$OUTPUT_ROOT" \
        --model_args "max_tokens=8192,temperature=0.0"
else
    python src/entrypoints/run_repair_inference.py \
        --dataset_name_or_path "$DATASET" \
        --split "$SPLIT" \
        --mode "$PATCH" \
        --model_name_or_path "$MODEL" \
        --output_dir "$OUTPUT_ROOT" \
        --max_tokens 8192 \
        --temperature 0.0 \
        "${EXTRA_ARGS[@]}"
fi

echo "=== Stage K4: Running evaluation ($MODE / $PATCH / $MODEL) ==="
# run_*_inference names predictions {output_dir}/{model}__{dataset_basename}
# [__{patch}]__{split}.jsonl — derive from $DATASET so every variant
# (file_snippet, node_source, rerank5, model-tagged trees) resolves.
PRED_FILE="${OUTPUT_ROOT}/${MODEL}__$(basename "$DATASET")"
if [ "$PATCH" != "diff" ]; then
    PRED_FILE="${PRED_FILE}__${PATCH}"
fi
PRED_FILE="${PRED_FILE}__${SPLIT}.jsonl"

python -m swebench.harness.run_evaluation \
    --dataset_name SWE-bench/SWE-bench_Lite \
    --split "$SPLIT" \
    --predictions_path "$PRED_FILE" \
    --max_workers 4 \
    --run_id "$RUN_ID" \
    --cache_level env

echo "=== Done ($MODE / $PATCH / $MODEL) ==="
