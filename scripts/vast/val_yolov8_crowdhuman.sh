#!/usr/bin/env bash
set -euo pipefail
# Ultralytics val on CrowdHuman val (YOLO layout required).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export MODEL_DIR="${MODEL_DIR:-/workspace/models}"
export WEIGHTS="${WEIGHTS:-${MODEL_DIR}/yolov8n_crowdhuman.pt}"
export DATA_YAML="${DATA_YAML:-${REPO_ROOT}/configs/datasets/crowdhuman_val.yaml}"
export BATCH="${BATCH:-8}"
export IMGSZ="${IMGSZ:-640}"

python3 <<PY
from ultralytics import YOLO
import os
w = os.environ["WEIGHTS"]
d = os.environ["DATA_YAML"]
model = YOLO(w)
metrics = model.val(data=d, imgsz=int(os.environ.get("IMGSZ", "640")), batch=int(os.environ["BATCH"]), plots=False, verbose=True)
rd = getattr(metrics, "results_dict", {}) or {}
print("results_dict:", rd)
PY
