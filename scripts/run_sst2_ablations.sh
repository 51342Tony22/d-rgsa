#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/sst2-6k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/sst2-6k-ablations}"
SEEDS="${SEEDS:-13 42 87}"

COMMON=(
  --train-file "$DATA_DIR/train.ready.jsonl"
  --validation-file "$DATA_DIR/validation.ready.jsonl"
  --test-file "$DATA_DIR/test.ready.jsonl"
  --tokenizer-name distilbert/distilbert-base-uncased
  --max-length 128
  --hidden-size 128
  --num-layers 2
  --num-heads 4
  --ffn-size 512
  --epochs 10
  --learning-rate 3e-4
  --batch-size 16
  --disable-tqdm
)

for seed in $SEEDS; do
  # budget=4 is already the rationale main experiment.
  for budget in 1 2 3 5; do
    run_dir="$OUTPUT_ROOT/anchor-budget-$budget/seed-$seed"
    if [[ -f "$run_dir/metrics.json" ]]; then
      echo "Skipping completed run: $run_dir"
      continue
    fi
    rgsa-train \
      "${COMMON[@]}" \
      --strategy rationale \
      --window-radius 4 \
      --anchor-budget "$budget" \
      --seed "$seed" \
      --output-dir "$run_dir"
  done

  # radius=4 is already covered by the local/rationale main experiments.
  for radius in 2 8; do
    for strategy in local rationale; do
      run_dir="$OUTPUT_ROOT/window-radius-$radius/$strategy/seed-$seed"
      if [[ -f "$run_dir/metrics.json" ]]; then
        echo "Skipping completed run: $run_dir"
        continue
      fi
      rgsa-train \
        "${COMMON[@]}" \
        --strategy "$strategy" \
        --window-radius "$radius" \
        --anchor-budget 4 \
        --seed "$seed" \
        --output-dir "$run_dir"
    done
  done
done
