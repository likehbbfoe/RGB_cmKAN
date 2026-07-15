#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${1:-data/custom_unpaired}"
SOURCE_DOMAIN="${2:-source}"
TARGET_DOMAIN="${3:-target}"
CONFIG_PATH="${4:-configs/custom_unpaired.example.yaml}"

cd "$PROJECT_ROOT"
python main.py train \
  --config "$CONFIG_PATH" \
  --data-root "$DATA_ROOT" \
  --source-domain "$SOURCE_DOMAIN" \
  --target-domain "$TARGET_DOMAIN"
