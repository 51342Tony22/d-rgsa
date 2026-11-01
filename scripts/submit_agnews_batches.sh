#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_ROOT:=$(pwd)}"
: "${SILICONFLOW_API_KEY:?Set SILICONFLOW_API_KEY before submitting batches}"

for split in train validation test; do
  rgsa-batch submit \
    --requests "$PROJECT_ROOT/data/agnews-6k/batch/${split}.requests.jsonl" \
    --job "$PROJECT_ROOT/data/agnews-6k/batch/${split}.job.json" \
    --description "RGSA AG News 6k ${split}"
done
