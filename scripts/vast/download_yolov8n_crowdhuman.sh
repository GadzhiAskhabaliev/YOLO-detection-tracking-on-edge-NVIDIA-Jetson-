#!/usr/bin/env bash
set -euo pipefail
# Official-ish CrowdHuman-trained YOLOv8n weights (akanametov).

export MODEL_DIR="${MODEL_DIR:-/workspace/models}"
mkdir -p "${MODEL_DIR}"

URL="https://github.com/akanametov/yolov8-crowdhuman/releases/download/v1.0/yolov8n_crowdhuman.pt"
DEST="${MODEL_DIR}/yolov8n_crowdhuman.pt"

if [[ -f "${DEST}" ]]; then
  echo "Already exists: ${DEST}"
  exit 0
fi

wget -O "${DEST}" "${URL}"
echo "Saved: ${DEST}"
