#!/usr/bin/env bash
set -euo pipefail
# FreeYOLO: CrowdHuman val eval → results/runs/ JSON (Group B, detector_id=7).
# Separate venv with torch cu124; NumPy pinned <2 (FreeYOLO legacy aliases).
# If CUDA mismatches: set TORCH_INDEX_URL to cu118 wheels, etc.
#
#   CROWDHUMAN_ROOT=/workspace/data/crowdhuman MODEL_DIR=/workspace/models \\
#     bash scripts/group_b/run_freeyolo_crowdhuman.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

GROUP_B_ROOT="${GROUP_B_ROOT:-/workspace/group_b}"
FREEYOLO_HOME="${FREEYOLO_HOME:-${GROUP_B_ROOT}/FreeYOLO}"
VENV="${FREEYOLO_VENV:-${GROUP_B_ROOT}/venv_freeyolo}"
CROWDHUMAN_ROOT="${CROWDHUMAN_ROOT:-/workspace/data/crowdhuman}"
MODEL_DIR="${MODEL_DIR:-/workspace/models}"
BRIDGE="${FREEYOLO_CH_BRIDGE:-${GROUP_B_ROOT}/freeyolo_crowdhuman_bridge}"

FREEYOLO_VARIANT="${FREEYOLO_VARIANT:-yolo_free_nano}"
WEIGHT_URL="${FREEYOLO_WEIGHT_URL:-https://github.com/yjh0410/FreeYOLO/releases/download/weight/yolo_free_nano_ch.pth}"
WEIGHT_PATH="${FREEYOLO_WEIGHT_PATH:-${MODEL_DIR}/yolo_free_nano_ch.pth}"

# JSON/README row name: default nano unchanged; else freeyolo_ch_<tiny|large|...>
if [[ -z "${FREEYOLO_BENCH_MODEL:-}" ]]; then
  if [[ "${FREEYOLO_VARIANT}" == "yolo_free_nano" ]]; then
    FREEYOLO_BENCH_MODEL="freeyolo_yolox_mot17"
  else
    FREEYOLO_BENCH_MODEL="freeyolo_ch_${FREEYOLO_VARIANT#yolo_free_}"
  fi
fi
FREEYOLO_DETECTOR_LABEL="${FREEYOLO_DETECTOR_LABEL:-FreeYOLO ${FREEYOLO_VARIANT} CrowdHuman}"

mkdir -p "${GROUP_B_ROOT}" "${MODEL_DIR}"

if [[ ! -d "${FREEYOLO_HOME}/.git" ]]; then
  echo "--- Clone FreeYOLO ---"
  git clone --depth 1 https://github.com/yjh0410/FreeYOLO.git "${FREEYOLO_HOME}"
fi

echo "--- Patch FreeYOLO: torch.load(weights_only=False) for PyTorch 2.6+ ---"
python3 scripts/group_b/patch_freeyolo_torch_load.py --freeyolo-home "${FREEYOLO_HOME}"

echo "--- Patch FreeYOLO: np.int / np.float / np.bool → built-ins (NumPy 2.x) ---"
python3 scripts/group_b/patch_freeyolo_numpy_aliases.py --freeyolo-home "${FREEYOLO_HOME}"

if [[ ! -f "${WEIGHT_PATH}" ]]; then
  echo "--- Download FreeYOLO weights ---"
  wget -O "${WEIGHT_PATH}.part" "${WEIGHT_URL}"
  mv "${WEIGHT_PATH}.part" "${WEIGHT_PATH}"
fi

if [[ ! -d "${VENV}" ]]; then
  echo "--- Create FreeYOLO venv ---"
  python3 -m venv "${VENV}"
fi
# shellcheck disable=SC1090
source "${VENV}/bin/activate"
PY="${VENV}/bin/python"
PIP="${VENV}/bin/pip"
"${PIP}" install -q --upgrade pip wheel
"${PIP}" install -q "numpy>=1.23,<2"
"${PIP}" install -q torch torchvision --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
"${PIP}" install -q opencv-python scipy matplotlib pycocotools loguru thop Pillow
# Above deps may pull numpy 2.x — pin back <2 (sources still patched for safety)
"${PIP}" install -q "numpy>=1.23,<2" --force-reinstall --no-deps

echo "--- NumPy version in venv (expect 1.x; 2.x triggers alias patches) ---"
"${PY}" -c "import numpy as np; print('numpy', np.__version__)"

echo "--- Prepare CrowdHuman bridge for FreeYOLO ---"
FREEYOLO_CH_BRIDGE="${BRIDGE}" CROWDHUMAN_ROOT="${CROWDHUMAN_ROOT}" \
  "${PY}" scripts/group_b/freeyolo_prepare_crowdhuman.py \
  --crowdhuman-root "${CROWDHUMAN_ROOT}" \
  --bridge-root "${BRIDGE}"

VAL_JSON="${BRIDGE}/CrowdHuman/annotations/val.json"
NIMG="$("${PY}" -c "import json; print(len(json.load(open('${VAL_JSON}'))['images']))")"

mkdir -p "${ROOT}/results/logs"
FREEYOLO_LOG="${ROOT}/results/logs/freeyolo_${FREEYOLO_VARIANT}_$(date -u +%Y%m%dT%H%M%SZ).log"
SEC0="$(date +%s)"
set +e
cd "${FREEYOLO_HOME}"
"${PY}" eval.py \
  --cuda \
  -d crowdhuman \
  -v "${FREEYOLO_VARIANT}" \
  --img_size 640 \
  --weight "${WEIGHT_PATH}" \
  --root "${BRIDGE}" \
  2>&1 | tee "${FREEYOLO_LOG}"
EC="${PIPESTATUS[0]}"
set -e
cd "${ROOT}"
SEC1="$(date +%s)"
WALL=$((SEC1 - SEC0))

if [[ "${EC}" -ne 0 ]]; then
  echo "eval.py exited ${EC}; full log: ${FREEYOLO_LOG}" >&2
  exit "${EC}"
fi

"${PY}" scripts/group_b/freeyolo_save_run.py \
  --eval-log "${FREEYOLO_LOG}" \
  --weights "${WEIGHT_PATH}" \
  --variant "${FREEYOLO_VARIANT}" \
  --weights-uri "${WEIGHT_URL}" \
  --model-name "${FREEYOLO_BENCH_MODEL}" \
  --detector-label "${FREEYOLO_DETECTOR_LABEL}" \
  --freeyolo-home "${FREEYOLO_HOME}" \
  --imgsz 640 \
  --wall-seconds "${WALL}" \
  --num-images "${NIMG}"

echo "--- FreeYOLO done; log: ${FREEYOLO_LOG}; plots: python3 scripts/plot_group_b_results.py ---"
