#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/cod/thesis
PYTHON="$ROOT/unsloth/bin/python3"
CONFIG="$ROOT/_clean/configs/chapter4.json"

cd "$ROOT"

"$PYTHON" _clean/src/chapter4/run_explainability.py --config "$CONFIG"
"$PYTHON" _clean/src/chapter4/plot_explainability.py --config "$CONFIG"
