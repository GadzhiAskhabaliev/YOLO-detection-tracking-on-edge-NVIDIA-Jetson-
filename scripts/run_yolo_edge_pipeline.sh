#!/usr/bin/env bash
set -euo pipefail

# One-command YOLO edge benchmark pipeline:
# 1) install deps, 2) bootstrap datasets/weights, 3) run YOLO benchmark,
# 4) dump MOT17 detections for tracking, 5) run unified eval + merge metrics.
#
# Usage:
#   bash scripts/run_yolo_edge_pipeline.sh
#
# Common overrides:
#   SKIP_BOOTSTRAP=1 SKIP_INSTALL=1 \
#   WEIGHTS=/workspace/models/yolov8n_crowdhuman.pt \
#   DEVICE=cuda:0 \
#   bash scripts/run_yolo_edge_pipeline.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_NAME="${MODEL_NAME:-yolov8n_crowdhuman}"
MODEL_DIR="${MODEL_DIR:-/workspace/models}"
WEIGHTS="${WEIGHTS:-${MODEL_DIR}/yolov8n_crowdhuman.pt}"
WEIGHTS_HUB="${WEIGHTS_HUB:-yakhyo/yolov8-crowdhuman}"
DATA_YAML="${DATA_YAML:-${REPO_ROOT}/configs/datasets/crowdhuman_val.yaml}"
DEVICE="${DEVICE:-cuda:0}"
IMGSZ="${IMGSZ:-640}"
BATCH_SIZE="${BATCH_SIZE:-8}"
WARMUP="${WARMUP:-50}"
ITERS="${ITERS:-200}"

MOT17_ROOT="${MOT17_ROOT:-/workspace/data/mot17}"
MOT17_TRAIN_ROOT="${MOT17_TRAIN_ROOT:-${MOT17_ROOT}/MOT17/train}"
MOT17_SUFFIX="${MOT17_SUFFIX:-FRCNN}"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-/workspace/artifacts/yolo-edge-bench}"
mkdir -p "${ARTIFACT_ROOT}"

MOT17_GT_JSON="${MOT17_GT_JSON:-${ARTIFACT_ROOT}/mot17_train_${MOT17_SUFFIX,,}_gt.json}"
MOT17_DT_JSON="${MOT17_DT_JSON:-${ARTIFACT_ROOT}/mot17_${MODEL_NAME}_dt.json}"
MOT17_DET_ROOT="${MOT17_DET_ROOT:-${ARTIFACT_ROOT}/mot17_det/${MODEL_NAME}}"
MOT17_EVAL_PATCH_JSON="${MOT17_EVAL_PATCH_JSON:-${ARTIFACT_ROOT}/mot17_${MODEL_NAME}_unified_patch.json}"

if [[ "${SKIP_BOOTSTRAP:-0}" != "1" ]]; then
  bash "${REPO_ROOT}/scripts/vast/run_cloud_bootstrap.sh"
elif [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
  # Bootstrap skipped, so run install separately if requested.
  bash "${REPO_ROOT}/scripts/vast/install_deps.sh"
fi

if [[ ! -f "${WEIGHTS}" ]]; then
  echo "Missing weights: ${WEIGHTS}" >&2
  exit 1
fi
if [[ ! -f "${DATA_YAML}" ]]; then
  echo "Missing dataset YAML: ${DATA_YAML}" >&2
  exit 1
fi

python3 "${REPO_ROOT}/scripts/bench_runner.py" \
  --model-name "${MODEL_NAME}" \
  --weights "${WEIGHTS}" \
  --weights-hub "${WEIGHTS_HUB}" \
  --bench-mode all \
  --data-yaml "${DATA_YAML}" \
  --device "${DEVICE}" \
  --imgsz "${IMGSZ}" \
  --batch-size "${BATCH_SIZE}" \
  --warmup "${WARMUP}" \
  --iters "${ITERS}" \
  --backend ultralytics_yolo

if [[ ! -f "${MOT17_GT_JSON}" ]]; then
  python3 "${REPO_ROOT}/scripts/mot17_gt_to_coco.py" \
    --mot17-train-root "${MOT17_TRAIN_ROOT}" \
    --det-subdir-suffix "${MOT17_SUFFIX}" \
    --out-json "${MOT17_GT_JSON}"
fi

python3 "${REPO_ROOT}/scripts/dump_ultralytics_mot17.py" \
  --gt-json "${MOT17_GT_JSON}" \
  --mot17-train-root "${MOT17_TRAIN_ROOT}" \
  --weights "${WEIGHTS}" \
  --out-coco-dt-json "${MOT17_DT_JSON}" \
  --mot-det-root "${MOT17_DET_ROOT}" \
  --imgsz "${IMGSZ}" \
  --device "${DEVICE}"

python3 "${REPO_ROOT}/scripts/eval_coco_predictions.py" \
  --gt-json "${MOT17_GT_JSON}" \
  --dt-json "${MOT17_DT_JSON}" \
  --strict \
  --out-patch-json "${MOT17_EVAL_PATCH_JSON}"

LATEST_RUN_JSON="$(
python3 - <<PY
from pathlib import Path
repo = Path(r"${REPO_ROOT}")
model = "${MODEL_NAME}"
runs = sorted((repo / "results" / "runs").glob(f"{model}_*.json"), key=lambda p: p.stat().st_mtime)
print(runs[-1] if runs else "")
PY
)"

if [[ -z "${LATEST_RUN_JSON}" ]]; then
  echo "Could not find results/runs/${MODEL_NAME}_*.json after benchmark." >&2
  exit 1
fi

python3 "${REPO_ROOT}/scripts/bench_runner.py" \
  --merge-json "${LATEST_RUN_JSON}" \
  --patch-json "${MOT17_EVAL_PATCH_JSON}"

echo "Pipeline complete."
echo "Run JSON: ${LATEST_RUN_JSON}"
echo "MOT17 COCO DT: ${MOT17_DT_JSON}"
echo "MOT17 tracker det root: ${MOT17_DET_ROOT}"
echo "Unified eval patch: ${MOT17_EVAL_PATCH_JSON}"
