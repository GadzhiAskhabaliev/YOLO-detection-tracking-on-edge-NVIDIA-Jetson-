#!/usr/bin/env bash
set -euo pipefail
# Official-ish CrowdHuman-trained YOLOv8n weights (akanametov).
# Truncated downloads cause torch EOFError — we enforce a minimum file size.

export MODEL_DIR="${MODEL_DIR:-/workspace/models}"
mkdir -p "${MODEL_DIR}"

URL="https://github.com/akanametov/yolov8-crowdhuman/releases/download/v1.0/yolov8n_crowdhuman.pt"
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
