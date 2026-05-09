#!/usr/bin/env bash
set -euo pipefail
# Полный прогон группы B:
#   №6 YOLOv8n-CrowdHuman (Ultralytics bench_runner)
#   №7 FreeYOLO (отдельный venv + eval CrowdHuman), слоти 4/5/8 — инструкции или опциональный тест
#
#   MODEL_DIR=/workspace/models DATA_YAML=configs/datasets/crowdhuman_val.yaml \\
#     bash scripts/run_group_b_benchmarks.sh
#
# Только YOLOv8 без FreeYOLO и без напоминаний по остальным:
#   GROUP_B_EXTRA_MODELS=0 bash scripts/run_group_b_benchmarks.sh
#
# Без FreeYOLO (долгий eval):
#   GROUP_B_FREEYOLO=0 bash scripts/group_b/run_remaining_models.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL_DIR="${MODEL_DIR:-/workspace/models}"
DATA_YAML="${DATA_YAML:-configs/datasets/crowdhuman_val.yaml}"

YOLO_WEIGHTS="${MODEL_DIR}/yolov8n_crowdhuman.pt"
if [[ -f "${YOLO_WEIGHTS}" ]]; then
  echo "--- Группа B / №6 YOLOv8n-CrowdHuman ---"
  python3 scripts/bench_runner.py \
    --model-name yolov8n_crowdhuman \
    --weights "${YOLO_WEIGHTS}" \
    --weights-hub yakhyo/yolov8-crowdhuman \
    --group B \
    --detector-id 6 \
    --detector-label "YOLOv8n-CrowdHuman" \
    --bench-mode all \
    --data-yaml "${DATA_YAML}"
else
  echo "Пропуск YOLOv8n-CrowdHuman: нет весов ${YOLO_WEIGHTS}"
fi

if [[ "${GROUP_B_EXTRA_MODELS:-1}" == "1" ]]; then
  echo ""
  echo "--- Остальные слоты группы B (FreeYOLO + памятки CrowdDet / Pedestron / PeopleNet) ---"
  bash scripts/group_b/run_remaining_models.sh
else
  echo ""
  echo "GROUP_B_EXTRA_MODELS=0 — пропуск scripts/group_b/run_remaining_models.sh"
  if python3 scripts/plot_group_b_results.py 2>/dev/null; then
    echo "Графики: results/group_b_report.md"
  else
    echo "Графики: pip install matplotlib pyyaml && python3 scripts/plot_group_b_results.py"
  fi
fi
