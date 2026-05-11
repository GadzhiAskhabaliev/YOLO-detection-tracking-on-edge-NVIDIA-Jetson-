#!/usr/bin/env bash
set -euo pipefail
# FreeYOLO on MOT17 train (FRCNN GT): dump COCO DT + eval_coco_predictions.py with full log (tee).
#
# Prerequisites: MOT17 downloaded, mot17_train_frcnn_gt.json built, FreeYOLO venv + clone + weights
# (same layout as run_freeyolo_crowdhuman.sh).
#
# Example (tiny):
#   MOT17_ROOT=/root/data/mot17 \\
#   FREEYOLO_HOME=/root/group_b/FreeYOLO \\
#   FREEYOLO_VENV=/root/group_b/venv_freeyolo \\
#   FREEYOLO_VARIANT=yolo_free_tiny \\
#   FREEYOLO_WEIGHT_PATH=/root/models/yolo_free_tiny_ch.pth \\
#   bash scripts/group_b/run_freeyolo_mot17_unified_eval.sh
#
# Example (nano → bench slug freeyolo_yolox_mot17):
#   FREEYOLO_VARIANT=yolo_free_nano \\
#   FREEYOLO_WEIGHT_PATH=/root/models/yolo_free_nano_ch.pth \\
#   FREEYOLO_DT_STEM=freeyolo_nano_mot17_train \\
#   bash scripts/group_b/run_freeyolo_mot17_unified_eval.sh
#
# If venv is elsewhere: FREEYOLO_VENV=/path/to/venv  OR  FREEYOLO_PYTHON=/path/to/venv/bin/python
# Create venv + install deps: bash scripts/group_b/run_freeyolo_crowdhuman.sh (stop after venv step if needed).

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

# If FreeYOLO was just cloned: python3 scripts/group_b/patch_freeyolo_torch_load.py + patch_freeyolo_numpy_aliases.py
MOT17_ROOT="${MOT17_ROOT:-/workspace/data/mot17}"
GT_JSON="${MOT17_GT_JSON:-${MOT17_ROOT}/annotations/mot17_train_frcnn_gt.json}"
TRAIN_ROOT="${MOT17_ROOT}/MOT17/train"

FREEYOLO_HOME="${FREEYOLO_HOME:-${GROUP_B_ROOT:-/workspace/group_b}/FreeYOLO}"
VENV="${FREEYOLO_VENV:-${GROUP_B_ROOT:-/workspace/group_b}/venv_freeyolo}"
FREEYOLO_VARIANT="${FREEYOLO_VARIANT:-yolo_free_tiny}"
WEIGHT_PATH="${FREEYOLO_WEIGHT_PATH:?Set FREEYOLO_WEIGHT_PATH to .pth}"

if [[ "${FREEYOLO_VARIANT}" == "yolo_free_nano" ]]; then
  FREEYOLO_DT_STEM="${FREEYOLO_DT_STEM:-freeyolo_nano_mot17_train}"
else
  FREEYOLO_DT_STEM="${FREEYOLO_DT_STEM:-freeyolo_${FREEYOLO_VARIANT#yolo_free_}_mot17_train}"
fi

OUT_DT="/tmp/${FREEYOLO_DT_STEM}_dt.json"
OUT_MET="/tmp/${FREEYOLO_DT_STEM}_unified_metrics.json"
OUT_PATCH="/tmp/${FREEYOLO_DT_STEM}_unified_patch.json"
MOT_DET_ROOT="${MOT17_ROOT}/detections/${FREEYOLO_DT_STEM}"

mkdir -p "${ROOT}/results/logs"
LOG="${ROOT}/results/logs/${FREEYOLO_DT_STEM}_unified_$(date -u +%Y%m%dT%H%M%SZ).log"

if [[ ! -f "${GT_JSON}" ]]; then
  echo "Missing GT: ${GT_JSON} — run mot17_gt_to_coco.py first." >&2
  exit 1
fi

if [[ -n "${FREEYOLO_PYTHON:-}" ]]; then
  PY="${FREEYOLO_PYTHON}"
  if [[ ! -x "${PY}" ]]; then
    echo "FREEYOLO_PYTHON is not executable: ${PY}" >&2
    exit 1
  fi
else
  if [[ ! -f "${VENV}/bin/activate" ]]; then
    echo "FreeYOLO venv not found: ${VENV}" >&2
    echo "Fix: export FREEYOLO_VENV=/path/to/venv   (directory with bin/activate)" >&2
    echo "  or FREEYOLO_PYTHON=/path/to/venv/bin/python" >&2
    echo "Create venv + deps: bash scripts/group_b/run_freeyolo_crowdhuman.sh" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "${VENV}/bin/activate"
  PY="${VENV}/bin/python"
fi

{
  echo "========== $(date -u +%Y-%m-%dT%H:%M:%SZ) dump_freeyolo_mot17 =========="
  "${PY}" "${ROOT}/scripts/group_b/dump_freeyolo_mot17.py" \
    --freeyolo-home "${FREEYOLO_HOME}" \
    --variant "${FREEYOLO_VARIANT}" \
    --weights "${WEIGHT_PATH}" \
    --gt-json "${GT_JSON}" \
    --mot17-train-root "${TRAIN_ROOT}" \
    --out-coco-dt-json "${OUT_DT}" \
    --mot-det-root "${MOT_DET_ROOT}"

  echo "========== $(date -u +%Y-%m-%dT%H:%M:%SZ) eval_coco_predictions =========="
  "${PY}" "${ROOT}/scripts/eval_coco_predictions.py" \
    --gt-json "${GT_JSON}" \
    --dt-json "${OUT_DT}" \
    --strict \
    --out-metrics-json "${OUT_MET}" \
    --out-patch-json "${OUT_PATCH}"

  echo "========== metrics file =========="
  cat "${OUT_MET}"
} 2>&1 | tee "${LOG}"

echo "--- Full log: ${LOG} ---"
