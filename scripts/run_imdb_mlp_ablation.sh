#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEEDS="${SEEDS:-13 42 87 101 123}"
READY_DIR="${READY_DIR:-$PROJECT_ROOT/data/imdb-6k}"
PREDICTED_DIR="${PREDICTED_DIR:-$PROJECT_ROOT/data/distilled/deepseek/imdb-6k-mlp}"
SELECTOR_DIR="${SELECTOR_DIR:-$PROJECT_ROOT/outputs/long-text-extension/selectors/imdb-6k/deepseek/mlp/seed-42}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/long-text-extension/quality/imdb-6k-mlp}"
TOKENIZER="${TOKENIZER:-$PROJECT_ROOT/outputs/distilled-selectors/sst2-6k/deepseek/bigru/seed-42}"

if [[ ! -f "$SELECTOR_DIR/selector.pt" ]]; then
  HF_HOME="$PROJECT_ROOT/.cache/huggingface" \
  TRANSFORMERS_OFFLINE=1 \
  HF_DATASETS_OFFLINE=1 \
  PYTHONPATH="$PROJECT_ROOT/src" \
  "$PYTHON_BIN" -m rgsa.distill \
    --train-file "$READY_DIR/train.ready.jsonl" \
    --validation-file "$READY_DIR/validation.ready.jsonl" \
    --test-file "$READY_DIR/test.ready.jsonl" \
    --output-dir "$SELECTOR_DIR" \
    --predicted-data-dir "$PREDICTED_DIR" \
    --teacher-name deepseek \
    --tokenizer-name "$TOKENIZER" \
    --architecture mlp \
    --max-length 512 \
    --anchor-budget 4 \
    --epochs 8 \
    --seed 42
fi

for seed in $SEEDS; do
  output_dir="$OUTPUT_ROOT/distilled/seed-$seed"
  if [[ -f "$output_dir/metrics.json" ]]; then
    echo "Skipping completed run: $output_dir"
    continue
  fi
  HF_HOME="$PROJECT_ROOT/.cache/huggingface" \
  TRANSFORMERS_OFFLINE=1 \
  HF_DATASETS_OFFLINE=1 \
  PYTHONPATH="$PROJECT_ROOT/src" \
  "$PYTHON_BIN" -m rgsa.train \
    --train-file "$PREDICTED_DIR/train.jsonl" \
    --validation-file "$PREDICTED_DIR/validation.jsonl" \
    --test-file "$PREDICTED_DIR/test.jsonl" \
    --strategy distilled \
    --tokenizer-name "$TOKENIZER" \
    --max-length 512 \
    --hidden-size 128 \
    --num-layers 2 \
    --num-heads 4 \
    --ffn-size 512 \
    --window-radius 16 \
    --anchor-budget 4 \
    --epochs 10 \
    --learning-rate 3e-4 \
    --batch-size 16 \
    --disable-tqdm \
    --seed "$seed" \
    --output-dir "$output_dir"
done
