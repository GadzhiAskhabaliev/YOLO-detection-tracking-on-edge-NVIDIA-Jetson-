#!/usr/bin/env bash
set -euo pipefail
# CrowdDet: официально через Docker + Google Drive / Baidu веса.
# Автоматизации полного eval в этом репозитории нет — только напоминание и опциональный clone.

GROUP_B_ROOT="${GROUP_B_ROOT:-/workspace/group_b}"
CROWDDET_HOME="${CROWDDET_HOME:-${GROUP_B_ROOT}/CrowdDet}"

echo "=== CrowdDet (группа B, слот 4) ==="
echo "Репозиторий: https://github.com/xg-chu/CrowdDet"
echo "Docker: в корне репозитория  docker build . -t crowddet && docker run --gpus all --shm-size=8g -it crowddet"
echo "Веса RCNN EMD Refine (crowdhuman): см. таблицу Models в README (Google Drive)."
echo ""
echo "После получения AP/MR на CrowdHuman сохраните JSON с:"
echo "  model=crowddet_r50_fpn_emd  group=B  detector_id=4"

if [[ ! -d "${CROWDDET_HOME}/.git" ]] && [[ "${CROWDDET_CLONE:-0}" == "1" ]]; then
  mkdir -p "${GROUP_B_ROOT}"
  git clone --depth 1 https://github.com/xg-chu/CrowdDet.git "${CROWDDET_HOME}"
  echo "Клонировано в ${CROWDDET_HOME}"
fi
