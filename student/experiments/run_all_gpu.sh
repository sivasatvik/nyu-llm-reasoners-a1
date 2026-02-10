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

# Batch size sweep (adjust --lr if needed)
$PY -m student.experiments.sweep_batch_sizes \
  --train-data "$TRAIN" \
  --val-data "$VAL" \
  --vocab-size "$VOCAB_SIZE" \
  --context-length "$CTX" \
  --num-layers "$LAYERS" \
  --d-model "$DMODEL" \
  --num-heads "$HEADS" \
  --d-ff "$DFF" \
  --max-iters "$MAX_ITERS" \
  --eval-iters "$EVAL_ITERS" \
  --eval-interval "$EVAL_INTERVAL" \
  --log-interval "$LOG_INTERVAL" \
  --lr 1e-3 \
  --device "$DEVICE"

# Ablations: no RMSNorm, post-norm, NoPE, SiLU FFN
$PY -m student.experiments.run_ablation \
  --train-data "$TRAIN" \
  --val-data "$VAL" \
  --vocab-size "$VOCAB_SIZE" \
  --context-length "$CTX" \
  --num-layers "$LAYERS" \
  --d-model "$DMODEL" \
  --num-heads "$HEADS" \
  --d-ff "$DFF" \
  --max-iters "$MAX_ITERS" \
  --batch-size "$BATCH" \
  --eval-iters "$EVAL_ITERS" \
  --eval-interval "$EVAL_INTERVAL" \
  --log-interval "$LOG_INTERVAL" \
  --lr 1e-3 \
  --device "$DEVICE"

echo "All GPU experiment sweeps completed. Logs are in student/experiments/logs."