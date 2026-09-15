#!/usr/bin/env bash
# Idempotent KG chain: kg.run (resumes from progress file) -> completeness
# check with retries -> LLM list rerank (resumes from its own progress file).
# Safe to re-run after a crash: every stage skips already-processed work.
set -u

# Raise fd limit: chroma rust bindings panic (EMFILE) under the default 1024
ulimit -n 65535 2>/dev/null || true

cd "$(dirname "$0")/.."
PY=".venv/bin/python"
PROGRESS="outputs/data__swe-bench_lite__style-3__fs-kg__k-auto__embed.progress.jsonl"
EXPECTED_ROWS=323
MAX_ATTEMPTS=10
RETRY_SLEEP_SEC=600

count_done() {
    "$PY" - "$PROGRESS" <<'PYEOF'
import json, sys

done = 0
with open(sys.argv[1]) as progress_file:
    for line in progress_file:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue  # crash-truncated tail line
        if entry.get("text_inputs"):
            done += 1
print(done)
PYEOF
}

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    echo "[kg-run-all] attempt $attempt/$MAX_ATTEMPTS: starting kg.run" | tee -a logs/kg_run_all.log
    # --subprocess-per-instance: each instance runs in a fresh python that
    # exits after writing its row — chroma/glibc memory returns to the OS,
    # so RSS cannot accumulate across instances (leak fix 2026-08-31)
    PYTHONPATH=src "$PY" -m kg.run config/kg_config.json \
        --embeddings --subprocess-per-instance
    done_rows=$(count_done)
    echo "[kg-run-all] attempt $attempt: $done_rows/$EXPECTED_ROWS complete rows" | tee -a logs/kg_run_all.log
    if [ "$done_rows" -ge "$EXPECTED_ROWS" ]; then
        break
    fi
    echo "[kg-run-all] incomplete; retrying in $RETRY_SLEEP_SEC s" | tee -a logs/kg_run_all.log
    sleep "$RETRY_SLEEP_SEC"
done

done_rows=$(count_done)
if [ "$done_rows" -lt "$EXPECTED_ROWS" ]; then
    echo "[kg-run-all] incomplete after $MAX_ATTEMPTS attempts ($done_rows/$EXPECTED_ROWS); ABORTING before rerank" | tee -a logs/kg_run_all.log
    exit 1
fi

echo "[kg-run-all] starting LLM rerank (list mode)" | tee -a logs/kg_run_all.log
PYTHONPATH=src "$PY" scripts/llm-rerank-files.py \
    --split test \
    --variant embed \
    --mode list \
    --signals freq,sim_max,sim_mean,parts \
    --model qwen3-coder-next-Q8_0 \
    --config config/kg_config.json
rerank_exit=$?

echo "[kg-run-all] chain finished (rerank exit=$rerank_exit)" | tee -a logs/kg_run_all.log
exit "$rerank_exit"
