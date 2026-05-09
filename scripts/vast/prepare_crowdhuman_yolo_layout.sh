#!/usr/bin/env bash
set -euo pipefail
# Symlink CrowdHuman Images + labels_val into Ultrantics-compatible layout:
#   ${CROWDHUMAN_ROOT}/yolo/images/val/*.jpg
#   ${CROWDHUMAN_ROOT}/yolo/labels/val/*.txt
# Then use configs/datasets/crowdhuman_val.yaml

export CROWDHUMAN_ROOT="${CROWDHUMAN_ROOT:-/workspace/data/crowdhuman}"
YOLO="${CROWDHUMAN_ROOT}/yolo"

rm -rf "${YOLO}"
mkdir -p "${YOLO}/images/val" "${YOLO}/labels/val"

shopt -s nullglob
for f in "${CROWDHUMAN_ROOT}/Images"/*; do
  [[ -f "$f" ]] || continue
  ln -sf "$(realpath "$f")" "${YOLO}/images/val/$(basename "$f")"
done
for f in "${CROWDHUMAN_ROOT}/labels_val"/*.txt; do
  [[ -f "$f" ]] || continue
  ln -sf "$(realpath "$f")" "${YOLO}/labels/val/$(basename "$f")"
done

echo "YOLO layout ready: ${YOLO}"
echo "Images: $(find "${YOLO}/images/val" -type l | wc -l) symlinks"
echo "Labels: $(find "${YOLO}/labels/val" -type l | wc -l) symlinks"
