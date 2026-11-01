#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_ROOT:=$(pwd)}"
: "${SILICONFLOW_API_KEY:?Set SILICONFLOW_API_KEY before collecting batches}"

for split in train validation test; do
  rgsa-batch collect \
    --job "$PROJECT_ROOT/data/agnews-6k/batch/${split}.job.json" \
    --source "$PROJECT_ROOT/data/agnews-6k/${split}.jsonl" \
    --output "$PROJECT_ROOT/data/agnews-6k/${split}.teacher.jsonl" \
    --dataset-name agnews-6k \
    --split "$split" \
    --model deepseek-ai/DeepSeek-V3 \
    --prompt-version rgsa-v1-agnews-guided
done

rgsa-audit \
  --input "$PROJECT_ROOT/data/agnews-6k/train.teacher.jsonl" \
          "$PROJECT_ROOT/data/agnews-6k/validation.teacher.jsonl" \
          "$PROJECT_ROOT/data/agnews-6k/test.teacher.jsonl" \
  --output "$PROJECT_ROOT/data/agnews-6k/teacher-audit.json"
