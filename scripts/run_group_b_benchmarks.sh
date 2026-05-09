#!/usr/bin/env bash
set -euo pipefail
# Full Group B driver:
#   slot 6 YOLOv8n-CH (bench_runner), slot 7 FreeYOLO (separate venv + CrowdHuman val),
#   slots 4/5/8 — documentation only unless you wire upstream binaries.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

GROUP_B_EXTRA_MODELS="${GROUP_B_EXTRA_MODELS:-1}"

echo "--- Group B / slot 6 YOLOv8n-CrowdHuman ---"

YOLO_WEIGHTS="${YOLO_WEIGHTS:-${MODEL_DIR:-/workspace/models}/yolov8n_crowdhuman.pt}"
YOLO_DATA="${YOLO_DATA:-configs/datasets/crowdhuman_val.yaml}"

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
  echo "Skipping YOLOv8n-CrowdHuman: weights missing ${YOLO_WEIGHTS}"
fi

if [[ "${GROUP_B_EXTRA_MODELS}" == "1" ]]; then
  echo "--- Other Group B slots (FreeYOLO + CrowdDet / Pedestron / PeopleNet notes) ---"
  bash scripts/group_b/run_remaining_models.sh
else
  echo "GROUP_B_EXTRA_MODELS=0 — skipping scripts/group_b/run_remaining_models.sh"
  if python3 scripts/plot_group_b_results.py 2>/dev/null; then
    echo "Plots: results/group_b_report.md"
  else
    echo "Plots: pip install matplotlib pyyaml && python3 scripts/plot_group_b_results.py"
  fi
fi
