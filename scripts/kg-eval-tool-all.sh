#!/bin/bash
# Run tool-mode repair + evaluation on all 3 context modes.
# Usage: bash scripts/kg-eval-tool-all.sh [split]
#   split: dev (default)
set -e

SPLIT="${1:-dev}"

for MODE in file_level file_snippet node_source; do
    echo "=== Tool mode: $MODE ==="
    bash scripts/kg-eval.sh "$SPLIT" "$MODE" tool
done

echo "=== All done ==="
