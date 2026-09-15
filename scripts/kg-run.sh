#!/bin/bash
# Stage K2: Run KG exploration + context assembly → preprocessed dataset.
# Usage: bash scripts/kg-run.sh [config_path] [split]
set -e

CONFIG="${1:-config/kg_config.json}"
SPLIT="${2:-test}"

export PYTHONPATH="$(dirname "$0")/../src:$PYTHONPATH"

python3 -c "
import sys
sys.path.insert(0, 'src')
from kg.config import KGConfig
from kg.run import main

config = KGConfig.from_file('${CONFIG}')
config.split = '${SPLIT}'
main(config)
"
echo "=== KG dataset created ==="
