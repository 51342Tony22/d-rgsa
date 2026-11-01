#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/sst2-6k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/sst2-6k}"
SEEDS="${SEEDS:-13 42 87}"

for split in train validation test; do
  rgsa-tfidf \
    --input "$DATA_DIR/${split}.teacher.jsonl" \
    --fit-input "$DATA_DIR/train.teacher.jsonl" \
    --output "$DATA_DIR/${split}.ready.jsonl" \
    --top-k 4
done

for seed in $SEEDS; do
  for strategy in full local random tfidf rationale; do
    rgsa-train \
      --train-file "$DATA_DIR/train.ready.jsonl" \
      --validation-file "$DATA_DIR/validation.ready.jsonl" \
      --test-file "$DATA_DIR/test.ready.jsonl" \
      --strategy "$strategy" \
      --tokenizer-name distilbert/distilbert-base-uncased \
      --max-length 128 \
      --hidden-size 128 \
      --num-layers 2 \
      --num-heads 4 \
      --ffn-size 512 \
      --window-radius 4 \
      --anchor-budget 4 \
      --epochs 10 \
      --learning-rate 3e-4 \
      --batch-size 16 \
      --disable-tqdm \
      --seed "$seed" \
      --output-dir "$OUTPUT_ROOT/$strategy/seed-$seed"
  done
done

rgsa-summarize --input-root "$OUTPUT_ROOT" --output "$OUTPUT_ROOT/results.csv"
