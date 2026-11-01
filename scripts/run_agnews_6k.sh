#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_ROOT:=$(pwd)}"

DATA_DIR="$PROJECT_ROOT/data/agnews-6k" \
OUTPUT_ROOT="$PROJECT_ROOT/outputs/agnews-6k" \
SEEDS="${SEEDS:-13 42 87 101 123}" \
EPOCHS="${EPOCHS:-10}" \
"$PROJECT_ROOT/scripts/run_dataset.sh"
