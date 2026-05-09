#!/usr/bin/env bash
set -euo pipefail
# FreeYOLO (group B) + plots. CrowdDet / Pedestron / PeopleNet live outside this repo.
# See docs/group_b_pedestrian_detectors.yaml and docs/group_b_benchmarks.md.
#
#   GROUP_B_FREEYOLO=1 bash scripts/group_b/run_remaining_models.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

export GROUP_B_ROOT="${GROUP_B_ROOT:-/workspace/group_b}"
GROUP_B_FREEYOLO="${GROUP_B_FREEYOLO:-1}"

echo "=== CrowdDet / Pedestron / PeopleNet: manual eval in upstream stacks (see docs) ==="

if [[ "${GROUP_B_FREEYOLO}" == "1" ]]; then
  bash scripts/group_b/run_freeyolo_crowdhuman.sh
else
  echo "Skipping FreeYOLO (GROUP_B_FREEYOLO!=1)"
fi

if command -v python3 >/dev/null 2>&1; then
  python3 scripts/plot_group_b_results.py || true
fi

echo "--- run_remaining_models done ---"
