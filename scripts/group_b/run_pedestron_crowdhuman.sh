#!/usr/bin/env bash
set -euo pipefail
# Pedestron (CrowdHuman, Cascade HRNet): нужен отдельный mmcv/mmdet пайплайн.
#
# 1) Установка: https://github.com/hasanirtiza/Pedestron/blob/master/INSTALL.md
# 2) Скачайте вес CrowdHuman Cascade HRNet вручную (Google Drive в README Pedestron).
# 3) Задайте переменные и включите прогон:
#    PEDESTRON_ROOT=/workspace/group_b/Pedestron \\
#    PEDESTRON_CKPT=/path/to/epoch_19.pth.stu \\
#    RUN_PEDESTRON_TEST=1 \\
#    bash scripts/group_b/run_pedestron_crowdhuman.sh

PEDESTRON_ROOT="${PEDESTRON_ROOT:-${GROUP_B_ROOT:-/workspace/group_b}/Pedestron}"
RUN_PEDESTRON_TEST="${RUN_PEDESTRON_TEST:-0}"

echo "=== Pedestron (группа B, слот 5) ==="
echo "Репозиторий: https://github.com/hasanirtiza/Pedestron"
echo "Пример multi-GPU теста в README:"
echo "  ./tools/dist_test.sh configs/elephant/crowdhuman/cascade_hrnet.py ./models_pretrained/epoch_19.pth.stu 8 --eval bbox"
echo ""
echo "После eval добавьте метрики в results/runs через ручной JSON или расширьте парсер лога."
echo "bench_slug: pedestron_cascade_hrnet_w32 | detector_id: 5 | group: B"

if [[ "${RUN_PEDESTRON_TEST}" != "1" ]]; then
  echo "Пропуск dist_test (RUN_PEDESTRON_TEST!=1)."
  exit 0
fi

if [[ -z "${PEDESTRON_CKPT:-}" || ! -f "${PEDESTRON_CKPT}" ]]; then
  echo "Задайте PEDESTRON_CKPT на существующий .pth/.pth.stu" >&2
  exit 1
fi

if [[ ! -d "${PEDESTRON_ROOT}/tools" ]]; then
  echo "Клонируйте Pedestron в PEDESTRON_ROOT=${PEDESTRON_ROOT}" >&2
  exit 1
fi

cd "${PEDESTRON_ROOT}"
GPUS="${PEDESTRON_GPUS:-1}"
./tools/dist_test.sh configs/elephant/crowdhuman/cascade_hrnet.py "${PEDESTRON_CKPT}" "${GPUS}" --eval bbox
