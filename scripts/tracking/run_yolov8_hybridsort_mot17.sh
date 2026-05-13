#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/scripts/tracking/path_defaults.sh"

MOT17_ROOT="${MOT17_ROOT:-${TRACKING_DATA_ROOT}/mot17}"
MOT17_SEQ="${MOT17_SEQ:-MOT17-02-FRCNN}"
MOT17_SOURCE="${MOT17_SOURCE:-${MOT17_ROOT}/MOT17/train/${MOT17_SEQ}/img1}"

WEIGHTS="${WEIGHTS:-${TRACKING_MODEL_ROOT}/yolov8n_crowdhuman.pt}"
REID_WEIGHTS="${REID_WEIGHTS:-osnet_x0_25_msmt17.pt}"
DEVICE="${DEVICE:-cuda:0}"
IMGSZ="${IMGSZ:-640}"
CONF="${CONF:-0.25}"
IOU="${IOU:-0.7}"

PROJECT="${PROJECT:-results/tracking/runs}"
NAME="${NAME:-yolov8_hybridsort_${MOT17_SEQ}}"
REPORT_DIR="${REPORT_DIR:-results/tracking}"

[[ -e "${WEIGHTS}" ]] || { echo "Missing weights: ${WEIGHTS}" >&2; exit 1; }
[[ -d "${MOT17_SOURCE}" || -f "${MOT17_SOURCE}" ]] || { echo "Missing source: ${MOT17_SOURCE}" >&2; exit 1; }

mkdir -p "${PROJECT}" "${REPORT_DIR}"

python3 "${REPO_ROOT}/scripts/tracking/run_yolov8_boxmot_mot17.py" \
  --weights "${WEIGHTS}" \
  --source "${MOT17_SOURCE}" \
  --imgsz "${IMGSZ}" \
  --conf "${CONF}" \
  --iou "${IOU}" \
  --device "${DEVICE}" \
  --tracker-type hybridsort \
  --tracker-label hybridsort \
  --reid-weights "${REID_WEIGHTS}" \
  --project "${PROJECT}" \
  --name "${NAME}" \
  --report-dir "${REPORT_DIR}" \
  --save-mot-txt
