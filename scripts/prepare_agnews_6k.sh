#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_ROOT:=$(pwd)}"

rgsa-prepare-agnews \
  --output-dir "$PROJECT_ROOT/data/agnews-6k" \
  --train-size 4800 \
  --validation-size 600 \
  --test-size 600 \
  --seed 42

for split in train validation test; do
  rgsa-batch prepare \
    --input "$PROJECT_ROOT/data/agnews-6k/${split}.jsonl" \
    --output "$PROJECT_ROOT/data/agnews-6k/batch/${split}.requests.jsonl" \
    --dataset-name agnews-6k \
    --split "$split" \
    --model deepseek-ai/DeepSeek-V3 \
    --prompt-profile agnews
done
