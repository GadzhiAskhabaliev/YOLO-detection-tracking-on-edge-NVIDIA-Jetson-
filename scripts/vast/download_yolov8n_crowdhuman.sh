#!/usr/bin/env bash
set -euo pipefail
# CrowdHuman-trained YOLOv8n: yakhyo/yolov8-crowdhuman release weights → saved as yolov8n_crowdhuman.pt.
# (Old akanametov/yolov8-crowdhuman release URLs currently 404.)
# Truncated downloads cause torch EOFError — minimum size enforced.

export MODEL_DIR="${MODEL_DIR:-/workspace/models}"
mkdir -p "${MODEL_DIR}"

URL="https://github.com/yakhyo/yolov8-crowdhuman/releases/download/weights/yolov8n_best.pt"
DEST="${MODEL_DIR}/yolov8n_crowdhuman.pt"
# yolov8n*.pt is usually ~6 MiB; reject obviously broken files
MIN_BYTES=$((4 * 1024 * 1024))

bytes() {
  stat -c%s "$1" 2>/dev/null || stat -f%z "$1"
}

if [[ -f "${DEST}" ]]; then
  sz=$(bytes "${DEST}")
  if (( sz >= MIN_BYTES )); then
    echo "Already present (${sz} bytes): ${DEST}"
    exit 0
  fi
  echo "Removing truncated/corrupt file (${sz} bytes): ${DEST}"
  rm -f "${DEST}"
fi

echo "Downloading → ${DEST}"
wget --tries=5 --timeout=60 --continue -O "${DEST}.part" "${URL}"
mv -f "${DEST}.part" "${DEST}"

sz=$(bytes "${DEST}")
if (( sz < MIN_BYTES )); then
  echo "Download failed: ${DEST} is only ${sz} bytes (expected ≥ ${MIN_BYTES}). Try again or check URL." >&2
  rm -f "${DEST}"
  exit 1
fi

echo "Saved: ${DEST} (${sz} bytes)"
