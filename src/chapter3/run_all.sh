#!/usr/bin/env bash
set -euo pipefail

PYTHON=/root/cod/thesis/unsloth/bin/python3
CHAPTER_DIR=/root/cod/thesis/_clean/src/chapter3
CONFIG=/root/cod/thesis/_clean/configs/chapter3.json
OUTPUT=/root/cod/thesis/_clean/outputs/chapter3

export HF_HOME=/root/cod/thesis/_clean/.cache/huggingface
export XDG_CACHE_HOME=/root/cod/thesis/_clean/.cache
export MPLCONFIGDIR=/root/cod/thesis/_clean/.cache/matplotlib
export TORCH_HOME=/root/cod/thesis/_clean/.cache/torch
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUTPUT/embeddings"

"$PYTHON" -u "$CHAPTER_DIR/create_splits.py" --config "$CONFIG"
"$PYTHON" -u "$CHAPTER_DIR/embed_reports.py" --config "$CONFIG" prepare

"$PYTHON" -u "$CHAPTER_DIR/embed_reports.py" --config "$CONFIG" \
  shard --index 0 --count 2 --device cuda:0 \
  > "$OUTPUT/embeddings/shard_0.log" 2>&1 &
PID0=$!

"$PYTHON" -u "$CHAPTER_DIR/embed_reports.py" --config "$CONFIG" \
  shard --index 1 --count 2 --device cuda:1 \
  > "$OUTPUT/embeddings/shard_1.log" 2>&1 &
PID1=$!

wait "$PID0"
wait "$PID1"

"$PYTHON" -u "$CHAPTER_DIR/embed_reports.py" --config "$CONFIG" merge
"$PYTHON" -u "$CHAPTER_DIR/train_classifiers.py" --config "$CONFIG"
"$PYTHON" -u "$CHAPTER_DIR/evaluate_classifiers.py" --config "$CONFIG"
