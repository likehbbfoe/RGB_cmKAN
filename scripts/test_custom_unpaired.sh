#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${1:-${CMKAN_DATA_ROOT:-/home/share/y50063074/data}}"
SOURCE_DOMAIN="${2:-${CMKAN_SOURCE_DOMAIN:-source}}"
TARGET_DOMAIN="${3:-${CMKAN_TARGET_DOMAIN:-target}}"
CONFIG_PATH="${4:-${CMKAN_CONFIG_PATH:-configs/custom_unpaired.example.yaml}}"
WEIGHTS="${5:-${CMKAN_WEIGHTS:-logs/checkpoints/last.ckpt}}"
OUTPUT_ROOT="${6:-${CMKAN_RESULTS_ROOT:-../experiment/results/custom_unpaired}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$PROJECT_ROOT"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config does not exist: $CONFIG_PATH" >&2
  exit 1
fi

EVAL_SPLIT="val"
if [[ -d "$DATA_ROOT/test/$SOURCE_DOMAIN" && -d "$DATA_ROOT/test/$TARGET_DOMAIN" ]]; then
  EVAL_SPLIT="test"
fi

SOURCE_INPUT="$DATA_ROOT/$EVAL_SPLIT/$SOURCE_DOMAIN"
TARGET_INPUT="$DATA_ROOT/$EVAL_SPLIT/$TARGET_DOMAIN"

if [[ ! -d "$SOURCE_INPUT" ]]; then
  echo "Source evaluation directory does not exist: $SOURCE_INPUT" >&2
  exit 1
fi
if [[ ! -d "$TARGET_INPUT" ]]; then
  echo "Target evaluation directory does not exist: $TARGET_INPUT" >&2
  exit 1
fi

echo "[1/3] Evaluate cycle and identity losses on $EVAL_SPLIT"
"$PYTHON_BIN" main.py test \
  --config "$CONFIG_PATH" \
  --weights "$WEIGHTS" \
  --data-root "$DATA_ROOT" \
  --source-domain "$SOURCE_DOMAIN" \
  --target-domain "$TARGET_DOMAIN"

echo "[2/3] Predict $SOURCE_DOMAIN -> $TARGET_DOMAIN"
"$PYTHON_BIN" main.py predict \
  --config "$CONFIG_PATH" \
  --weights "$WEIGHTS" \
  --input "$SOURCE_INPUT" \
  --output "$OUTPUT_ROOT/${SOURCE_DOMAIN}_to_${TARGET_DOMAIN}" \
  --batch_size 1

echo "[3/3] Predict $TARGET_DOMAIN -> $SOURCE_DOMAIN"
"$PYTHON_BIN" main.py predict \
  --config "$CONFIG_PATH" \
  --weights "$WEIGHTS" \
  --input "$TARGET_INPUT" \
  --output "$OUTPUT_ROOT/${TARGET_DOMAIN}_to_${SOURCE_DOMAIN}" \
  --batch_size 1 \
  --reverse

echo "Done. Predictions are in: $OUTPUT_ROOT"
