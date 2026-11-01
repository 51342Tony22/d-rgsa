#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_ROOT:=$(pwd)}"

rgsa-altair-figures \
  --project-root "$PROJECT_ROOT" \
  --output-dir "$PROJECT_ROOT/outputs/altair-figures" \
  --max-heatmap-length "${MAX_HEATMAP_LENGTH:-64}" \
  --sample-index "${SAMPLE_INDEX:-0}"
