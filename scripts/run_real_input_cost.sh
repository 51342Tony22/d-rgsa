#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
BASELINE_ROOT="${BASELINE_ROOT:-/Users/starsfalling/Desktop/Project/Distilling-Teacher-Rationales-into-Global-Anchors-for-Sparse-Text-Classification}"
DEVICE="${DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-8}"
WARMUP="${WARMUP:-10}"
PASSES="${PASSES:-5}"
SEED="${SEED:-42}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/long-text-extension/real-cost/raw}"
RUN_AG="${RUN_AG:-1}"
RUN_IMDB="${RUN_IMDB:-1}"

run_benchmark() {
  local dataset="$1"
  local method="$2"
  local data_file="$3"
  local student_dir="$4"
  local window_radius="$5"
  local selector_dir="${6:-}"
  local output="$OUTPUT_ROOT/$dataset/$method.json"
  if [[ -f "$output" && "${FORCE:-0}" != "1" ]]; then
    echo "Skipping completed benchmark: $output"
    return
  fi
  args=(
    --dataset "$dataset"
    --data-file "$data_file"
    --method "$method"
    --student-dir "$student_dir"
    --batch-size "$BATCH_SIZE"
    --window-radius "$window_radius"
    --anchor-budget 4
    --warmup "$WARMUP"
    --passes "$PASSES"
    --seed "$SEED"
    --device "$DEVICE"
    --dtype float32
    --output "$output"
  )
  if [[ -n "$selector_dir" ]]; then
    args+=(--selector-dir "$selector_dir")
  fi
  HF_HOME="$PROJECT_ROOT/.cache/huggingface" \
  TRANSFORMERS_OFFLINE=1 \
  HF_DATASETS_OFFLINE=1 \
  PYTHONPATH="$PROJECT_ROOT/src" \
  "$PYTHON_BIN" -m rgsa.benchmark_real_data "${args[@]}"
}

AG_DATA="$PROJECT_ROOT/data/agnews-6k/test.ready.jsonl"
AG_BIGRU_STUDENT="$PROJECT_ROOT/outputs/d-rgsa-pilot/agnews-6k/deepseek/bigru/seed-42"
AG_MLP_STUDENT="$PROJECT_ROOT/outputs/long-text-extension/quality/agnews-6k-mlp/distilled/seed-42"
AG_BIGRU_SELECTOR="$PROJECT_ROOT/outputs/distilled-selectors/agnews-6k/deepseek/bigru/seed-42"
AG_MLP_SELECTOR="$PROJECT_ROOT/outputs/long-text-extension/selectors/agnews-6k/deepseek/mlp/seed-42"

if [[ "$RUN_AG" == "1" ]]; then
  run_benchmark "AG News" full "$AG_DATA" \
    "$BASELINE_ROOT/outputs/agnews-6k/full/seed-42" 4
  run_benchmark "AG News" local "$AG_DATA" \
    "$BASELINE_ROOT/outputs/agnews-6k/local/seed-42" 4
  run_benchmark "AG News" d-rgsa "$AG_DATA" \
    "$AG_BIGRU_STUDENT" 4 "$AG_BIGRU_SELECTOR"
  run_benchmark "AG News" d-rgsa-mlp "$AG_DATA" \
    "$AG_MLP_STUDENT" 4 "$AG_MLP_SELECTOR"
fi

IMDB_DATA="$PROJECT_ROOT/data/imdb-6k/test.ready.jsonl"
IMDB_QUALITY="$PROJECT_ROOT/outputs/long-text-extension/quality/imdb-6k"
IMDB_BIGRU_SELECTOR="$PROJECT_ROOT/outputs/long-text-extension/selectors/imdb-6k/deepseek/bigru/seed-42"
IMDB_MLP_SELECTOR="$PROJECT_ROOT/outputs/long-text-extension/selectors/imdb-6k/deepseek/mlp/seed-42"

if [[ "$RUN_IMDB" == "1" ]]; then
  run_benchmark IMDb full "$IMDB_DATA" "$IMDB_QUALITY/full/seed-42" 16
  run_benchmark IMDb local "$IMDB_DATA" "$IMDB_QUALITY/local/seed-42" 16
  run_benchmark IMDb d-rgsa "$IMDB_DATA" \
    "$IMDB_QUALITY/distilled/seed-42" 16 "$IMDB_BIGRU_SELECTOR"
  run_benchmark IMDb d-rgsa-mlp "$IMDB_DATA" \
    "$PROJECT_ROOT/outputs/long-text-extension/quality/imdb-6k-mlp/distilled/seed-42" \
    16 "$IMDB_MLP_SELECTOR"
fi
