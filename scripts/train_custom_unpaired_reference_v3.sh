#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${1:-${CMKAN_DATA_ROOT:-/home/share/y50063074/data}}"
SOURCE_DOMAIN="${2:-${CMKAN_SOURCE_DOMAIN:-source}}"
TARGET_DOMAIN="${3:-${CMKAN_TARGET_DOMAIN:-target}}"
CONFIG_PATH="${4:-${CMKAN_CONFIG_PATH:-configs/custom_unpaired_reference_v3.server.yaml}}"

cd "$PROJECT_ROOT"
python main.py train \
  --config "$CONFIG_PATH" \
  --data-root "$DATA_ROOT" \
  --source-domain "$SOURCE_DOMAIN" \
  --target-domain "$TARGET_DOMAIN"
