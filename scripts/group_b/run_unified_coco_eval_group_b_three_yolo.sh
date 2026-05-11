#!/usr/bin/env bash
# CrowdHuman val — same metrics for all three YOLO-family Group B rows:
#   1) Ultralytics yolov8n_crowdhuman  (dump_ultralytics_coco_dt.py)
#   2) FreeYOLO yolo_free_tiny         (dump_freeyolo_crowdhuman_coco_dt.py)
#   3) FreeYOLO yolo_free_nano         (same dumper)
# Then scripts/eval_coco_predictions.py --strict → bench_runner --merge-json (keeps FPS).
#
# Prerequisites:
#   - Bridge val.json:  python3 scripts/group_b/freeyolo_prepare_crowdhuman.py ...
#   - Ultralytics + CUDA for slot (1); FreeYOLO clone + venv for (2)(3)
#   - Existing results/runs/*.json from prior bench (for merge targets)
#
# Typical GPU host:
#   export CROWDHUMAN_ROOT=/workspace/data/crowdhuman
#   export MODEL_DIR=/workspace/models
#   export GROUP_B_ROOT=/workspace/group_b
#   export FREEYOLO_HOME="${GROUP_B_ROOT}/FreeYOLO"
#   export FREEYOLO_VENV="${GROUP_B_ROOT}/venv_freeyolo"
#   bash scripts/group_b/run_unified_coco_eval_group_b_three_yolo.sh
#
# Skip steps: SKIP_YOLOV8=1  SKIP_FREEYOLO_TINY=1  SKIP_FREEYOLO_NANO=1

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

CROWDHUMAN_ROOT="${CROWDHUMAN_ROOT:-/workspace/data/crowdhuman}"
MODEL_DIR="${MODEL_DIR:-/workspace/models}"
GROUP_B_ROOT="${GROUP_B_ROOT:-/workspace/group_b}"
BRIDGE="${FREEYOLO_CH_BRIDGE:-${GROUP_B_ROOT}/freeyolo_crowdhuman_bridge}"
VAL_JSON="${VAL_JSON:-${BRIDGE}/CrowdHuman/annotations/val.json}"
IMAGES_DIR="${IMAGES_DIR:-${CROWDHUMAN_ROOT}/Images}"

FREEYOLO_HOME="${FREEYOLO_HOME:-${GROUP_B_ROOT}/FreeYOLO}"
FREEYOLO_PY="${FREEYOLO_PYTHON:-${FREEYOLO_VENV:-${GROUP_B_ROOT}/venv_freeyolo}/bin/python}"

YOLOV8_WEIGHTS="${YOLOV8_WEIGHTS:-${MODEL_DIR}/yolov8n_crowdhuman.pt}"
MERGE_YOLOV8="${MERGE_YOLOV8:-${ROOT}/results/runs/yolov8n_crowdhuman_2026-05-09T143848Z.json}"

TINY_WEIGHTS="${FREEYOLO_TINY_WEIGHTS:-${MODEL_DIR}/yolo_free_tiny_ch.pth}"
MERGE_TINY="${MERGE_FREEYOLO_TINY:-${ROOT}/results/runs/freeyolo_ch_tiny_2026-05-09T143328Z.json}"

NANO_WEIGHTS="${FREEYOLO_NANO_WEIGHTS:-${MODEL_DIR}/yolo_free_nano_ch.pth}"
MERGE_NANO="${MERGE_FREEYOLO_NANO:-${ROOT}/results/runs/freeyolo_yolox_mot17_2026-05-09T144753Z.json}"

WORKDIR="${UNIFIED_EVAL_WORKDIR:-/tmp/group_b_unified_yolo_three}"
mkdir -p "${WORKDIR}" "${ROOT}/results/logs"

stamp_utc() { date -u +%Y-%m-%dT%H%M%SZ; }

append_log_note_to_patch() {
  local patch_path="$1" log_path="$2"
  python3 -c "
import json
from pathlib import Path
p = Path('${patch_path}')
log = '${log_path}'
patch = json.loads(p.read_text(encoding='utf-8'))
rel = log
if rel.startswith('${ROOT}/'):
    rel = rel[len('${ROOT}/'):]
msg = 'Unified CrowdHuman val: dump + scripts/eval_coco_predictions.py (strict); full tee: ' + rel
patch.setdefault('notes', []).insert(0, msg)
p.write_text(json.dumps(patch, indent=2), encoding='utf-8')
"
}

if [[ ! -f "${VAL_JSON}" ]]; then
  echo "Missing bridge GT: ${VAL_JSON}" >&2
  echo "Run: FREEYOLO_CH_BRIDGE=${BRIDGE} CROWDHUMAN_ROOT=${CROWDHUMAN_ROOT} \\" >&2
  echo "  python3 scripts/group_b/freeyolo_prepare_crowdhuman.py" >&2
  exit 1
fi
if [[ ! -d "${IMAGES_DIR}" ]]; then
  echo "Missing images dir: ${IMAGES_DIR}" >&2
  exit 1
fi

run_yolov8() {
  [[ "${SKIP_YOLOV8:-0}" == "1" ]] && { echo "--- SKIP_YOLOV8=1 ---"; return 0; }
  [[ -f "${MERGE_YOLOV8}" ]] || { echo "Missing merge target: ${MERGE_YOLOV8}" >&2; exit 1; }
  [[ -f "${YOLOV8_WEIGHTS}" ]] || { echo "Missing weights: ${YOLOV8_WEIGHTS}" >&2; exit 1; }

  local dt="${WORKDIR}/yolov8n_ch_val_dt.json"
  local patch="${WORKDIR}/patch_yolov8n.json"
  local log="${ROOT}/results/logs/yolov8n_crowdhuman_unified_cocoeval_$(stamp_utc).log"

  {
    echo "--- dump_ultralytics_coco_dt ---"
    python3 "${ROOT}/scripts/dump_ultralytics_coco_dt.py" \
      --gt-json "${VAL_JSON}" \
      --images-dir "${IMAGES_DIR}" \
      --weights "${YOLOV8_WEIGHTS}" \
      --out-json "${dt}"
    echo "--- eval_coco_predictions ---"
    python3 "${ROOT}/scripts/eval_coco_predictions.py" \
      --gt-json "${VAL_JSON}" \
      --dt-json "${dt}" \
      --strict \
      --out-patch-json "${patch}"
  } | tee "${log}"

  append_log_note_to_patch "${patch}" "${log}"
  python3 "${ROOT}/scripts/bench_runner.py" \
    --merge-json "${MERGE_YOLOV8}" \
    --patch-json "${patch}"
  echo "--- yolov8n merge OK: ${MERGE_YOLOV8} ---"
}

run_freeyolo_variant() {
  local variant="$1" weights="$2" merge_json="$3" skip_flag="${4:-0}"
  [[ "${skip_flag}" == "1" ]] && { echo "--- skip ${variant} (SKIP flag) ---"; return 0; }
  [[ -f "${merge_json}" ]] || { echo "Missing merge target: ${merge_json}" >&2; exit 1; }
  [[ -f "${weights}" ]] || { echo "Missing weights: ${weights}" >&2; exit 1; }
  if [[ ! -f "${FREEYOLO_PY}" ]] && ! command -v "${FREEYOLO_PY}" >/dev/null; then
    echo "FreeYOLO python not found: ${FREEYOLO_PY}" >&2
    exit 1
  fi
  [[ -d "${FREEYOLO_HOME}" ]] || { echo "Missing FREEYOLO_HOME: ${FREEYOLO_HOME}" >&2; exit 1; }

  local safe="${variant//[^a-zA-Z0-9_]/_}"
  local dt="${WORKDIR}/freeyolo_${safe}_ch_val_dt.json"
  local patch="${WORKDIR}/patch_freeyolo_${safe}.json"
  local log="${ROOT}/results/logs/freeyolo_${safe}_unified_cocoeval_$(stamp_utc).log"

  {
    echo "--- dump_freeyolo_crowdhuman_coco_dt (${variant}) ---"
    "${FREEYOLO_PY}" "${ROOT}/scripts/group_b/dump_freeyolo_crowdhuman_coco_dt.py" \
      --freeyolo-home "${FREEYOLO_HOME}" \
      --variant "${variant}" \
      --weights "${weights}" \
      --gt-json "${VAL_JSON}" \
      --images-dir "${IMAGES_DIR}" \
      --out-coco-dt-json "${dt}"
    echo "--- eval_coco_predictions ---"
    python3 "${ROOT}/scripts/eval_coco_predictions.py" \
      --gt-json "${VAL_JSON}" \
      --dt-json "${dt}" \
      --strict \
      --out-patch-json "${patch}"
  } | tee "${log}"

  append_log_note_to_patch "${patch}" "${log}"
  python3 "${ROOT}/scripts/bench_runner.py" \
    --merge-json "${merge_json}" \
    --patch-json "${patch}"
  echo "--- FreeYOLO ${variant} merge OK: ${merge_json} ---"
}

run_yolov8
run_freeyolo_variant "yolo_free_tiny" "${TINY_WEIGHTS}" "${MERGE_TINY}" "${SKIP_FREEYOLO_TINY:-0}"
run_freeyolo_variant "yolo_free_nano" "${NANO_WEIGHTS}" "${MERGE_NANO}" "${SKIP_FREEYOLO_NANO:-0}"

echo ""
echo "--- Paste into docs/benchmark_unified_cocoeval.md (verify numbers) ---"
python3 "${ROOT}/scripts/group_b/print_crowdhuman_unified_md_rows.py" \
  "${MERGE_YOLOV8}" "${MERGE_TINY}" "${MERGE_NANO}" || true

echo "--- Done ---"
