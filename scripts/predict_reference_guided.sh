#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${CMKAN_DATA_ROOT:-/home/share/y50063074/data}"
REFERENCE_PATH="${1:-${CMKAN_REFERENCE:-}}"
INPUT_PATH="${2:-${CMKAN_INPUT:-$DATA_ROOT/val/source}}"
CONFIG_PATH="${3:-${CMKAN_CONFIG_PATH:-configs/custom_unpaired_reference.server.yaml}}"
WEIGHTS="${4:-${CMKAN_WEIGHTS:-logs/checkpoints/last.ckpt}}"
OUTPUT_PATH="${5:-${CMKAN_OUTPUT:-results/reference_guided/source_to_target}}"
BATCH_SIZE="${CMKAN_BATCH_SIZE:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$PROJECT_ROOT"

if [[ -z "$REFERENCE_PATH" ]]; then
  echo "Usage: $0 REFERENCE_IMAGE_OR_DIR [INPUT_IMAGE_OR_DIR] [CONFIG] [WEIGHTS] [OUTPUT_DIR]" >&2
  echo "You may also set CMKAN_REFERENCE instead of the first argument." >&2
  exit 2
fi

if [[ ! -f "$INPUT_PATH" && ! -d "$INPUT_PATH" ]]; then
  echo "Input image or directory does not exist: $INPUT_PATH" >&2
  exit 1
fi
if [[ ! -f "$REFERENCE_PATH" && ! -d "$REFERENCE_PATH" ]]; then
  echo "Reference image or directory does not exist: $REFERENCE_PATH" >&2
  exit 1
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config does not exist: $CONFIG_PATH" >&2
  exit 1
fi

"$PYTHON_BIN" main.py predict \
  --config "$CONFIG_PATH" \
  --weights "$WEIGHTS" \
  --input "$INPUT_PATH" \
  --reference "$REFERENCE_PATH" \
  --output "$OUTPUT_PATH" \
  --batch_size "$BATCH_SIZE"

echo "Done. Reference-guided predictions are in: $OUTPUT_PATH"
