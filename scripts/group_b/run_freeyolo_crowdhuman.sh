#!/usr/bin/env bash
set -euo pipefail
# FreeYOLO: eval на CrowdHuman val → JSON в results/runs/ (группа B, detector_id=7).
# Использует отдельный venv с torch cu124. NumPy закреплён <2 (FreeYOLO использует np.int и др.).
# При проблемах совместимости: TORCH_INDEX_URL=.../cu118 и т.д.
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

mkdir -p "${GROUP_B_ROOT}" "${MODEL_DIR}"

if [[ ! -d "${FREEYOLO_HOME}/.git" ]]; then
  echo "--- Клонирование FreeYOLO ---"
  git clone --depth 1 https://github.com/yjh0410/FreeYOLO.git "${FREEYOLO_HOME}"
fi

echo "--- Патч FreeYOLO: torch.load(weights_only=False) для PyTorch 2.6+ ---"
python3 scripts/group_b/patch_freeyolo_torch_load.py --freeyolo-home "${FREEYOLO_HOME}"

if [[ ! -f "${WEIGHT_PATH}" ]]; then
  echo "--- Скачивание весов FreeYOLO (CrowdHuman nano) ---"
  wget -O "${WEIGHT_PATH}.part" "${WEIGHT_URL}"
  mv "${WEIGHT_PATH}.part" "${WEIGHT_PATH}"
fi

if [[ ! -d "${VENV}" ]]; then
  echo "--- Создание venv FreeYOLO ---"
  python3 -m venv "${VENV}"
fi
# shellcheck disable=SC1090
source "${VENV}/bin/activate"
pip install -q --upgrade pip wheel
pip install -q "numpy>=1.23,<2"
pip install -q torch torchvision --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
pip install -q opencv-python scipy matplotlib pycocotools loguru thop Pillow
# зависимости выше могут подтянуть numpy 2.x — вернуть <2 для FreeYOLO
pip install -q "numpy>=1.23,<2" --force-reinstall --no-deps

echo "--- Подготовка CrowdHuman для FreeYOLO ---"
FREEYOLO_CH_BRIDGE="${BRIDGE}" CROWDHUMAN_ROOT="${CROWDHUMAN_ROOT}" \
  python scripts/group_b/freeyolo_prepare_crowdhuman.py \
  --crowdhuman-root "${CROWDHUMAN_ROOT}" \
  --bridge-root "${BRIDGE}"

VAL_JSON="${BRIDGE}/CrowdHuman/annotations/val.json"
NIMG="$(python -c "import json; print(len(json.load(open('${VAL_JSON}'))['images']))")"

LOG="$(mktemp)"
SEC0="$(date +%s)"
set +e
cd "${FREEYOLO_HOME}"
python eval.py \
  --cuda \
  -d crowdhuman \
  -v "${FREEYOLO_VARIANT}" \
  --img_size 640 \
  --weight "${WEIGHT_PATH}" \
  --root "${BRIDGE}" \
  2>&1 | tee "${LOG}"
EC="${PIPESTATUS[0]}"
set -e
cd "${ROOT}"
SEC1="$(date +%s)"
WALL=$((SEC1 - SEC0))

if [[ "${EC}" -ne 0 ]]; then
  echo "eval.py завершился с кодом ${EC}" >&2
  rm -f "${LOG}"
  exit "${EC}"
fi

python scripts/group_b/freeyolo_save_run.py \
  --eval-log "${LOG}" \
  --weights "${WEIGHT_PATH}" \
  --variant "${FREEYOLO_VARIANT}" \
  --weights-uri "${WEIGHT_URL}" \
  --wall-seconds "${WALL}" \
  --num-images "${NIMG}"

rm -f "${LOG}"
echo "--- FreeYOLO готов; обновите графики: python3 scripts/plot_group_b_results.py ---"
