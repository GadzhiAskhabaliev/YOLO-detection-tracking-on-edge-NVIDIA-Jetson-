#!/usr/bin/env bash
set -euo pipefail
# One-shot cloud bootstrap (tmux recommended). Override roots via env if needed.
#
#   export CROWDHUMAN_ROOT=/workspace/data/crowdhuman
#   export MOT17_ROOT=/workspace/data/mot17
#   export MODEL_DIR=/workspace/models
#   bash scripts/vast/run_cloud_bootstrap.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/install_deps.sh"
bash "${SCRIPT_DIR}/download_crowdhuman_val.sh"
bash "${SCRIPT_DIR}/download_mot17.sh"
bash "${SCRIPT_DIR}/convert_crowdhuman_odgt.sh"
bash "${SCRIPT_DIR}/prepare_crowdhuman_yolo_layout.sh"
bash "${SCRIPT_DIR}/download_yolov8n_crowdhuman.sh"

echo "Bootstrap done. Next:"
echo "  bash scripts/vast/bench_yolo_fps.py --weights \${MODEL_DIR:-/workspace/models}/yolov8n_crowdhuman.pt --out-json /workspace/bench.json"
echo "  bash scripts/vast/val_yolov8_crowdhuman.sh"
