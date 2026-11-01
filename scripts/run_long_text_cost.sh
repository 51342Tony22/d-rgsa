#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-8}"
WARMUP="${WARMUP:-10}"
REPEATS="${REPEATS:-50}"
LENGTHS="${LENGTHS:-128 256 512 1024 2048}"
METHODS="${METHODS:-full local d-rgsa d-rgsa-mlp d-rgsa-student}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/long-text-extension/cost/raw}"

for length in $LENGTHS; do
  for method in $METHODS; do
    output="$OUTPUT_ROOT/$method/length-$length.json"
    if [[ -f "$output" && "${FORCE:-0}" != "1" ]]; then
      echo "Skipping completed benchmark: $output"
      continue
    fi
    PYTHONPATH="$PROJECT_ROOT/src" "$PYTHON_BIN" -m rgsa.benchmark_cost \
      --method "$method" \
      --length "$length" \
      --batch-size "$BATCH_SIZE" \
      --window-radius 16 \
      --anchor-budget 4 \
      --hidden-size 128 \
      --num-layers 2 \
      --num-heads 4 \
      --ffn-size 512 \
      --warmup "$WARMUP" \
      --repeats "$REPEATS" \
      --device "$DEVICE" \
      --dtype float32 \
      --seed 42 \
      --output "$output"
  done
done

PYTHONPATH="$PROJECT_ROOT/src" "$PYTHON_BIN" -m rgsa.extension_summary \
  --project-root "$PROJECT_ROOT" \
  --baseline-root \
    "${BASELINE_ROOT:?Set BASELINE_ROOT to the frozen experiment project}"

PYTHONPATH="$PROJECT_ROOT/src" "$PYTHON_BIN" -m rgsa.extension_figures \
  --tables-dir "$PROJECT_ROOT/outputs/long-text-extension/tables" \
  --output-dir "$PROJECT_ROOT/outputs/long-text-extension/figures"
