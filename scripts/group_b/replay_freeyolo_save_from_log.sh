#!/usr/bin/env bash
set -euo pipefail
# После успешного eval.py — записать JSON с полными метриками (speed bench + recall из лога),
# без повторного прогона валидации. Нужен venv FreeYOLO и GPU для микробенча.
#
# Пример (nano):
#   FREEYOLO_HOME=/workspace/group_b/FreeYOLO PY=/workspace/group_b/venv_freeyolo/bin/python \\
#     bash scripts/group_b/replay_freeyolo_save_from_log.sh \\
#       results/logs/freeyolo_yolo_free_nano_*.log \\
#       /workspace/models/yolo_free_nano_ch.pth \\
#       yolo_free_nano \\
#       freeyolo_yolox_mot17 \\
#       'https://github.com/yjh0410/FreeYOLO/releases/download/weight/yolo_free_nano_ch.pth'

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

if [[ "${#}" -lt 5 ]]; then
  echo "Usage: $0 <eval-log> <weights.pth> <FREEYOLO_VARIANT> <FREEYOLO_BENCH_MODEL> <weights-uri-url>" >&2
  exit 1
fi

LOG="$1"
WEIGHTS="$2"
VARIANT="$3"
MODEL_NAME="$4"
WEIGHTS_URI="$5"

FREEYOLO_HOME="${FREEYOLO_HOME:?Set FREEYOLO_HOME}"
if [[ -z "${PY:-}" ]]; then
  PY="${FREEYOLO_HOME}/../venv_freeyolo/bin/python"
fi
if [[ ! -x "${PY}" ]]; then
  echo "Python not found: ${PY} — задайте PY=/path/to/venv_freeyolo/bin/python" >&2
  exit 1
fi

VAL_JSON="${FREEYOLO_CH_BRIDGE:-/workspace/group_b/freeyolo_crowdhuman_bridge}/CrowdHuman/annotations/val.json"
NIMG="$("${PY}" -c "import json; print(len(json.load(open('${VAL_JSON}'))['images']))")"

"${PY}" scripts/group_b/freeyolo_save_run.py \
  --eval-log "${LOG}" \
  --weights "${WEIGHTS}" \
  --variant "${VARIANT}" \
  --weights-uri "${WEIGHTS_URI}" \
  --model-name "${MODEL_NAME}" \
  --detector-label "FreeYOLO ${VARIANT} CrowdHuman" \
  --freeyolo-home "${FREEYOLO_HOME}" \
  --imgsz 640 \
  --wall-seconds 0 \
  --num-images "${NIMG}"

echo "--- Обновите сводки локально: python3 -c \"…bench_runner generate_benchmark_summary_md…\" ---"
