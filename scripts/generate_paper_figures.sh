#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_ROOT:=$(pwd)}"

FIGURE_DIR="$PROJECT_ROOT/outputs/paper-figures"

rgsa-aggregate-figures \
  --dataset "SST-2=$PROJECT_ROOT/outputs/sst2-6k-preliminary/results.summary.json" \
  --dataset "TREC=$PROJECT_ROOT/outputs/trec/results.summary.json" \
  --dataset "AG News=$PROJECT_ROOT/outputs/agnews-6k/results.summary.json" \
  --output-dir "$FIGURE_DIR"

for dataset in sst2-6k trec agnews-6k; do
  input="$PROJECT_ROOT/data/$dataset/test.ready.jsonl"
  if [[ -f "$input" ]]; then
    for strategy in full local random tfidf rationale; do
      rgsa-visualize \
        --input "$input" \
        --sample-index 0 \
        --strategy "$strategy" \
        --tokenizer-name distilbert/distilbert-base-uncased \
        --window-radius 4 \
        --anchor-budget 4 \
        --max-length 128 \
        --output "$FIGURE_DIR/heatmap_${dataset}_${strategy}.png"
    done
  fi
done
