#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEEDS="${SEEDS:-13 42 87 101 123}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data/imdb-6k}"
READY_DIR="${READY_DIR:-$PROJECT_ROOT/data/imdb-6k}"
PREDICTED_DIR="${PREDICTED_DIR:-$PROJECT_ROOT/data/distilled/deepseek/imdb-6k}"
SELECTOR_DIR="${SELECTOR_DIR:-$PROJECT_ROOT/outputs/long-text-extension/selectors/imdb-6k/deepseek/bigru/seed-42}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/long-text-extension/quality/imdb-6k}"
TOKENIZER="${TOKENIZER:-$PROJECT_ROOT/outputs/distilled-selectors/sst2-6k/deepseek/bigru/seed-42}"

for split in train validation test; do
  if [[ ! -f "$DATA_DIR/$split.teacher.jsonl" ]]; then
    echo "Missing teacher annotations: $DATA_DIR/$split.teacher.jsonl" >&2
    exit 1
  fi
  PYTHONPATH="$PROJECT_ROOT/src" "$PYTHON_BIN" -m rgsa.tfidf \
    --input "$DATA_DIR/$split.teacher.jsonl" \
    --fit-input "$DATA_DIR/train.teacher.jsonl" \
    --output "$READY_DIR/$split.ready.jsonl" \
    --top-k 4
done

PYTHONPATH="$PROJECT_ROOT/src" "$PYTHON_BIN" -m rgsa.audit \
  --input \
    "$DATA_DIR/train.teacher.jsonl" \
    "$DATA_DIR/validation.teacher.jsonl" \
    "$DATA_DIR/test.teacher.jsonl" \
  --output "$DATA_DIR/teacher-audit.json"

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
    --architecture bigru \
    --max-length 512 \
    --anchor-budget 4 \
    --epochs 8 \
    --seed 42
fi

for seed in $SEEDS; do
  for strategy in full local random tfidf rationale distilled; do
    if [[ "$strategy" == "distilled" ]]; then
      train_file="$PREDICTED_DIR/train.jsonl"
      validation_file="$PREDICTED_DIR/validation.jsonl"
      test_file="$PREDICTED_DIR/test.jsonl"
    else
      train_file="$READY_DIR/train.ready.jsonl"
      validation_file="$READY_DIR/validation.ready.jsonl"
      test_file="$READY_DIR/test.ready.jsonl"
    fi
    output_dir="$OUTPUT_ROOT/$strategy/seed-$seed"
    if [[ -f "$output_dir/metrics.json" ]]; then
      echo "Skipping completed run: $output_dir"
      continue
    fi
    HF_HOME="$PROJECT_ROOT/.cache/huggingface" \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    PYTHONPATH="$PROJECT_ROOT/src" \
    "$PYTHON_BIN" -m rgsa.train \
      --train-file "$train_file" \
      --validation-file "$validation_file" \
      --test-file "$test_file" \
      --strategy "$strategy" \
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
done

PYTHONPATH="$PROJECT_ROOT/src" "$PYTHON_BIN" -m rgsa.extension_summary \
  --project-root "$PROJECT_ROOT" \
  --baseline-root \
    "${BASELINE_ROOT:?Set BASELINE_ROOT to the frozen experiment project}"

PYTHONPATH="$PROJECT_ROOT/src" "$PYTHON_BIN" -m rgsa.extension_figures \
  --tables-dir "$PROJECT_ROOT/outputs/long-text-extension/tables" \
  --output-dir "$PROJECT_ROOT/outputs/long-text-extension/figures"
