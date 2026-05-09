#!/usr/bin/env bash
set -euo pipefail
# Прогон всех «остальных» моделей группы B (кроме YOLOv8n — см. scripts/run_group_b_benchmarks.sh).
#
#   GROUP_B_FREEYOLO=1      # по умолчанию 1 — долго на полном val
#   GROUP_B_RUN_PEDESTRON=0 # включите после установки mmcv + весов
#   bash scripts/group_b/run_remaining_models.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

export GROUP_B_ROOT="${GROUP_B_ROOT:-/workspace/group_b}"
GROUP_B_FREEYOLO="${GROUP_B_FREEYOLO:-1}"

bash scripts/group_b/run_crowddet.sh

bash scripts/group_b/run_pedestron_crowdhuman.sh

if [[ "${GROUP_B_FREEYOLO}" == "1" ]]; then
  bash scripts/group_b/run_freeyolo_crowdhuman.sh
else
  echo "Пропуск FreeYOLO (GROUP_B_FREEYOLO!=1)"
fi

bash scripts/group_b/run_peoplenet.sh

if command -v python3 >/dev/null 2>&1; then
  python3 scripts/plot_group_b_results.py || true
fi

echo "--- Конец run_remaining_models ---"
