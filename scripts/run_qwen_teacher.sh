#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
if [[ -f "$PROJECT_ROOT/.env" ]]; then
  set -a
  source "$PROJECT_ROOT/.env"
  set +a
fi
: "${SILICONFLOW_API_KEY:?Set SILICONFLOW_API_KEY before running Qwen annotation}"

QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3.6-27B}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
WORKERS="${WORKERS:-6}"
LIMIT="${LIMIT:-}"
DATASETS="${DATASETS:-sst2-6k trec agnews-6k}"
SPLITS="${SPLITS:-train validation test}"

for dataset in $DATASETS; do
  case "$dataset" in
    trec) profile="trec" ;;
    agnews-6k) profile="agnews" ;;
    *) profile="generic" ;;
  esac
  for split in $SPLITS; do
    args=(
      --input "$PROJECT_ROOT/data/$dataset/$split.jsonl"
      --output "$PROJECT_ROOT/data/qwen/$dataset/$split.teacher.jsonl"
      --dataset-name "$dataset"
      --split "$split"
      --provider siliconflow
      --model "$QWEN_MODEL"
      --prompt-profile "$profile"
      --max-rationale-tokens 6
      --workers "$WORKERS"
    )
    if [[ -n "$LIMIT" ]]; then
      args+=(--limit "$LIMIT")
    fi
    PYTHONPATH="$PROJECT_ROOT/src" "$PYTHON_BIN" -m rgsa.annotate_concurrent "${args[@]}"
  done
done
