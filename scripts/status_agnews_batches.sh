#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_ROOT:=$(pwd)}"
: "${SILICONFLOW_API_KEY:?Set SILICONFLOW_API_KEY before checking batch status}"

for split in train validation test; do
  echo "== $split =="
  rgsa-batch status \
    --job "$PROJECT_ROOT/data/agnews-6k/batch/${split}.job.json"
done
