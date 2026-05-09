#!/usr/bin/env bash
set -euo pipefail
# ODGT → YOLO txt labels under labels_val/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export CROWDHUMAN_ROOT="${CROWDHUMAN_ROOT:-/workspace/data/crowdhuman}"

python3 "${REPO_ROOT}/scripts/vast/crowdhuman_odgt_to_yolo.py" \
  --odgt "${CROWDHUMAN_ROOT}/annotation_val.odgt" \
  --images-dir "${CROWDHUMAN_ROOT}/Images" \
  --out-dir "${CROWDHUMAN_ROOT}/labels_val"
