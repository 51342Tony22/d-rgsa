#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data/imdb-6k}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-V3}"
ACTION="${1:-status}"

if [[ -f "$PROJECT_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/.env"
  set +a
fi

: "${SILICONFLOW_API_KEY:?Set SILICONFLOW_API_KEY in the shell or ignored .env file}"

for split in train validation test; do
  requests="$DATA_DIR/batch/$split.requests.jsonl"
  job="$DATA_DIR/batch/$split.job.json"
  teacher="$DATA_DIR/$split.teacher.jsonl"
  case "$ACTION" in
    submit)
      if [[ -f "$job" ]]; then
        echo "Skipping existing job: $job"
      else
        PYTHONPATH="$PROJECT_ROOT/src" "$PYTHON_BIN" -m rgsa.batch submit \
          --requests "$requests" \
          --job "$job" \
          --description "D-RGSA IMDb 6K $split"
      fi
      ;;
    status)
      if [[ ! -f "$job" ]]; then
        echo "No submitted job for $split"
      else
        PYTHONPATH="$PROJECT_ROOT/src" "$PYTHON_BIN" -m rgsa.batch status \
          --job "$job"
      fi
      ;;
    collect)
      PYTHONPATH="$PROJECT_ROOT/src" "$PYTHON_BIN" -m rgsa.batch collect \
        --job "$job" \
        --source "$DATA_DIR/$split.jsonl" \
        --output "$teacher" \
        --dataset-name imdb-6k \
        --split "$split" \
        --model "$MODEL" \
        --max-rationale-tokens 6
      ;;
    *)
      echo "Usage: $0 {submit|status|collect}" >&2
      exit 2
      ;;
  esac
done
