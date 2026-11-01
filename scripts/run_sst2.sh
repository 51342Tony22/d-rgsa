#!/usr/bin/env bash
set -euo pipefail

# Examples:
#   TEACHER_PROVIDER=deepseek TEACHER_MODEL=deepseek-chat bash scripts/run_sst2.sh
#   TEACHER_PROVIDER=qwen TEACHER_MODEL=qwen-plus bash scripts/run_sst2.sh
# Use LIMIT=100 for a cheap smoke run; leave it empty for the full split.

: "${TEACHER_PROVIDER:?Set TEACHER_PROVIDER (openai/deepseek/qwen/siliconflow/compatible/local-hf)}"
: "${TEACHER_MODEL:?Set TEACHER_MODEL}"

LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi

for split in train validation; do
  rgsa-annotate \
    --dataset sst2 \
    --split "$split" \
    --provider "$TEACHER_PROVIDER" \
    --model "$TEACHER_MODEL" \
    --provide-gold-label \
    --max-rationale-tokens 6 \
    --output "data/sst2/${split}.teacher.jsonl" \
    "${LIMIT_ARGS[@]}"
done

rgsa-tfidf \
  --input data/sst2/train.teacher.jsonl \
  --output data/sst2/train.ready.jsonl \
  --top-k 4

rgsa-tfidf \
  --input data/sst2/validation.teacher.jsonl \
  --fit-input data/sst2/train.teacher.jsonl \
  --output data/sst2/validation.ready.jsonl \
  --top-k 4

for seed in 13 42 87; do
  for strategy in full local random tfidf rationale; do
    rgsa-train \
      --train-file data/sst2/train.ready.jsonl \
      --validation-file data/sst2/validation.ready.jsonl \
      --strategy "$strategy" \
      --tokenizer-name distilbert/distilbert-base-uncased \
      --hidden-size 128 --num-layers 2 --num-heads 4 --ffn-size 512 \
      --window-radius 4 \
      --anchor-budget 4 \
      --seed "$seed" \
      --output-dir "outputs/sst2/${strategy}/seed-${seed}"
  done
done

rgsa-summarize --input-root outputs/sst2 --output outputs/sst2/results.csv
