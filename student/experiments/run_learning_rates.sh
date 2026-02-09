#!/usr/bin/env bash
set -euo pipefail

ROOT="../../"
PY="$ROOT/venv/bin/python"

TRAIN="$ROOT/data/tinystories_train_uint16.bin"
VAL="$ROOT/data/tinystories_valid_uint16.bin"
VOCAB_SIZE=10000

# Base TinyStories config (17M params)
CTX=256
LAYERS=4
DMODEL=512
HEADS=16
DFF=1344
BATCH=32
MAX_ITERS=5000
EVAL_ITERS=50
EVAL_INTERVAL=200
LOG_INTERVAL=50
DEVICE=cuda

# Learning rate sweep
$PY -m student.experiments.sweep_learning_rates \
  --train-data "$TRAIN" \
  --val-data "$VAL" \
  --vocab-size "$VOCAB_SIZE" \
  --context-length "$CTX" \
  --num-layers "$LAYERS" \
  --d-model "$DMODEL" \
  --num-heads "$HEADS" \
  --d-ff "$DFF" \
  --batch-size "$BATCH" \
  --max-iters "$MAX_ITERS" \
  --eval-iters "$EVAL_ITERS" \
  --eval-interval "$EVAL_INTERVAL" \
  --log-interval "$LOG_INTERVAL" \
  --device "$DEVICE"