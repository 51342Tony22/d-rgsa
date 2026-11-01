#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEEDS="${SEEDS:-13 42 87 101 123}"
DATASETS="${DATASETS:-sst2-6k trec agnews-6k}"

for dataset in $DATASETS; do
  teacher_dir="$PROJECT_ROOT/data/qwen/$dataset"
  ready_dir="$PROJECT_ROOT/data/qwen-ready/$dataset"
  predicted_dir="$PROJECT_ROOT/data/distilled/qwen/$dataset"
  selector_dir="$PROJECT_ROOT/outputs/distilled-selectors/$dataset/qwen/bigru/seed-42"
  tokenizer="${TOKENIZER:-$PROJECT_ROOT/outputs/distilled-selectors/$dataset/deepseek/bigru/seed-42}"

  for split in train validation test; do
    PYTHONPATH="$PROJECT_ROOT/src" "$PYTHON_BIN" -m rgsa.tfidf \
      --input "$teacher_dir/$split.teacher.jsonl" \
      --fit-input "$teacher_dir/train.teacher.jsonl" \
      --output "$ready_dir/$split.ready.jsonl" \
      --top-k 4
  done

  PYTHONPATH="$PROJECT_ROOT/src" "$PYTHON_BIN" -m rgsa.audit \
    --input \
      "$teacher_dir/train.teacher.jsonl" \
      "$teacher_dir/validation.teacher.jsonl" \
      "$teacher_dir/test.teacher.jsonl" \
    --output "$teacher_dir/teacher-audit.json"

  HF_HOME="$PROJECT_ROOT/.cache/huggingface" \
  TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH="$PROJECT_ROOT/src" \
  "$PYTHON_BIN" -m rgsa.teacher_compare \
    --teacher-a \
      "$PROJECT_ROOT/data/$dataset/train.ready.jsonl" \
      "$PROJECT_ROOT/data/$dataset/validation.ready.jsonl" \
      "$PROJECT_ROOT/data/$dataset/test.ready.jsonl" \
    --teacher-b \
      "$teacher_dir/train.teacher.jsonl" \
      "$teacher_dir/validation.teacher.jsonl" \
      "$teacher_dir/test.teacher.jsonl" \
    --teacher-a-name DeepSeek \
    --teacher-b-name Qwen \
    --tokenizer-name "$tokenizer" \
    --output "$PROJECT_ROOT/outputs/qwen-teacher/$dataset/teacher_comparison.json"

  HF_HOME="$PROJECT_ROOT/.cache/huggingface" \
  TRANSFORMERS_OFFLINE=1 \
  HF_DATASETS_OFFLINE=1 \
  PYTHONPATH="$PROJECT_ROOT/src" \
  "$PYTHON_BIN" -m rgsa.distill \
    --train-file "$ready_dir/train.ready.jsonl" \
    --validation-file "$ready_dir/validation.ready.jsonl" \
    --test-file "$ready_dir/test.ready.jsonl" \
    --output-dir "$selector_dir" \
    --predicted-data-dir "$predicted_dir" \
    --teacher-name qwen \
    --tokenizer-name "$tokenizer" \
    --architecture bigru \
    --anchor-budget 4 \
    --epochs 8 \
    --seed 42

  for seed in $SEEDS; do
    for method in direct distilled; do
      if [[ "$method" == "direct" ]]; then
        strategy="rationale"
        train_file="$ready_dir/train.ready.jsonl"
        validation_file="$ready_dir/validation.ready.jsonl"
        test_file="$ready_dir/test.ready.jsonl"
        tokenizer="$selector_dir"
      else
        strategy="distilled"
        train_file="$predicted_dir/train.jsonl"
        validation_file="$predicted_dir/validation.jsonl"
        test_file="$predicted_dir/test.jsonl"
        tokenizer="$selector_dir"
      fi
      output_dir="$PROJECT_ROOT/outputs/qwen-teacher/$dataset/$method/seed-$seed"
      if [[ -f "$output_dir/metrics.json" ]]; then
        echo "Skipping completed run: $output_dir"
        continue
      fi
      HF_HOME="$PROJECT_ROOT/.cache/huggingface" \
      TRANSFORMERS_OFFLINE=1 \
      HF_DATASETS_OFFLINE=1 \
      PYTHONPATH="$PROJECT_ROOT/src" \
      "$PYTHON_BIN" -m rgsa.train \
        --train-file "$train_file" \
        --validation-file "$validation_file" \
        --test-file "$test_file" \
        --strategy "$strategy" \
        --tokenizer-name "$tokenizer" \
        --max-length 128 \
        --hidden-size 128 \
        --num-layers 2 \
        --num-heads 4 \
        --ffn-size 512 \
        --window-radius 4 \
        --anchor-budget 4 \
        --epochs 10 \
        --learning-rate 3e-4 \
        --batch-size 16 \
        --disable-tqdm \
        --seed "$seed" \
        --output-dir "$output_dir"
    done
  done
done
