#!/usr/bin/env bash
set -euo pipefail
# 3-YOLO detection benchmark driver:
#   1) YOLOv8n CrowdHuman
#   2) FreeYOLO tiny CrowdHuman
#   3) FreeYOLO nano CrowdHuman

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

YOLO_WEIGHTS="${YOLO_WEIGHTS:-${MODEL_DIR:-/workspace/models}/yolov8n_crowdhuman.pt}"
YOLO_DATA="${YOLO_DATA:-configs/datasets/crowdhuman_val.yaml}"

echo "--- [1/3] YOLOv8n CrowdHuman ---"
if [[ -f "${YOLO_WEIGHTS}" ]]; then
  python3 scripts/bench_runner.py \
    --model-name yolov8n_crowdhuman \
    --weights "${YOLO_WEIGHTS}" \
    --weights-hub yakhyo/yolov8-crowdhuman \
    --bench-mode all \
    --data-yaml "${YOLO_DATA}" \
    --group B \
    --detector-id 6 \
    --detector-label "YOLOv8n-CrowdHuman"
else
  echo "Skipping YOLOv8n-CrowdHuman: missing ${YOLO_WEIGHTS}"
fi

echo "--- [2/3] FreeYOLO tiny CrowdHuman ---"
FREEYOLO_VARIANT=yolo_free_tiny \
FREEYOLO_WEIGHT_URL=https://github.com/yjh0410/FreeYOLO/releases/download/weight/yolo_free_tiny_ch.pth \
FREEYOLO_WEIGHT_PATH="${MODEL_DIR:-/workspace/models}/yolo_free_tiny_ch.pth" \
FREEYOLO_BENCH_MODEL=freeyolo_ch_tiny \
bash scripts/yolo_detectors/run_freeyolo_crowdhuman.sh

echo "--- [3/3] FreeYOLO nano CrowdHuman ---"
FREEYOLO_VARIANT=yolo_free_nano \
FREEYOLO_WEIGHT_URL=https://github.com/yjh0410/FreeYOLO/releases/download/weight/yolo_free_nano_ch.pth \
FREEYOLO_WEIGHT_PATH="${MODEL_DIR:-/workspace/models}/yolo_free_nano_ch.pth" \
FREEYOLO_BENCH_MODEL=freeyolo_yolox_mot17 \
bash scripts/yolo_detectors/run_freeyolo_crowdhuman.sh
