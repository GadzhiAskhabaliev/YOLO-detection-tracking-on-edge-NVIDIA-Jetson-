#!/usr/bin/env bash
set -euo pipefail
trap 'echo "[FAIL] line ${LINENO}: ${BASH_COMMAND}" >&2; exit 1' ERR

# Safe, container-first runner for the current winning setup:
# yolov8n_crowdhuman + yolov8_bytetrack (conf=0.35, iou=0.7).
#
# Designed for Jetson lab usage with fail-fast checks and reproducible outputs.

REPO_URL="${REPO_URL:-https://github.com/GadzhiAskhabaliev/YOLO-detection-tracking-on-edge-NVIDIA-Jetson-.git}"
REPO_PARENT="${REPO_PARENT:-/workspace/repo}"
REPO_DIR_NAME="${REPO_DIR_NAME:-YOLO-detection-tracking-on-edge-NVIDIA-Jetson-}"
REPO_ROOT="${REPO_ROOT:-${REPO_PARENT}/${REPO_DIR_NAME}}"
PIN_COMMIT="${PIN_COMMIT:-c2a52ab}"

MOT17_ROOT="${MOT17_ROOT:-/workspace/data/mot17}"
MOT17_SEQ="${MOT17_SEQ:-MOT17-02-FRCNN}"
MODEL_DIR="${MODEL_DIR:-/workspace/models}"
WEIGHTS="${WEIGHTS:-${MODEL_DIR}/yolov8n_crowdhuman.pt}"

CONF="${CONF:-0.35}"
IOU="${IOU:-0.7}"
IMGSZ="${IMGSZ:-640}"
DEVICE="${DEVICE:-cuda:0}"

REPORT_DIR="${REPORT_DIR:-/workspace/results/tracking}"
PROJECT="${PROJECT:-/workspace/results/tracking/runs}"
TRACKEVAL_OUT_DIR="${TRACKEVAL_OUT_DIR:-/workspace/results/tracking/trackeval}"
LOG_DIR="${LOG_DIR:-/workspace/logs}"

INSTALL_DEPS="${INSTALL_DEPS:-0}" # set 1 only if explicitly approved
PIP_INSTALL="${PIP_INSTALL:-1}"

cfg_suffix() {
  local conf="$1"
  local iou="$2"
  local c="${conf//./}"
  local i="${iou//./}"
  printf "c%s_i%s" "${c}" "${i}"
}

CFG_SUFFIX="$(cfg_suffix "${CONF}" "${IOU}")"
RUN_NAME="yolov8_bytetrack_${MOT17_SEQ}_${CFG_SUFFIX}"
TRACKER_NAME="${TRACKER_NAME:-yolov8_bytetrack_${CFG_SUFFIX}}"

mkdir -p "${REPO_PARENT}" "${MODEL_DIR}" "${REPORT_DIR}" "${PROJECT}" "${TRACKEVAL_OUT_DIR}" "${LOG_DIR}"

echo "=== Preflight (host/container) ==="
echo "timestamp_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "arch: $(uname -m)"
echo "kernel: $(uname -a)"
if command -v tegrastats >/dev/null 2>&1; then
  echo "tegrastats: available"
else
  echo "tegrastats: not found (ok on non-Jetson hosts)"
fi
nvidia-smi >/dev/null 2>&1 && echo "nvidia-smi: available" || echo "nvidia-smi: not available (often normal on Jetson)"

python3 - <<'PY'
import torch
print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
PY

if [[ -d "${REPO_ROOT}/.git" ]]; then
  echo "=== Repo exists, fetching and checking out ==="
  cd "${REPO_ROOT}"
  git fetch --all --tags
  git checkout "${PIN_COMMIT}"
elif [[ -d "${REPO_ROOT}" ]] && [[ -n "$(ls -A "${REPO_ROOT}" 2>/dev/null)" ]]; then
  echo "ERROR: ${REPO_ROOT} is not empty and is not a git repo." >&2
  exit 1
else
  echo "=== Cloning repo ==="
  git clone "${REPO_URL}" "${REPO_ROOT}"
  cd "${REPO_ROOT}"
  git checkout "${PIN_COMMIT}"
fi

[[ -f "${REPO_ROOT}/README.md" ]] || {
  echo "ERROR: README.md not found under ${REPO_ROOT}. Check REPO_URL/REPO_DIR_NAME." >&2
  exit 1
}
[[ -d "${REPO_ROOT}/scripts/tracking" ]] || {
  echo "ERROR: scripts/tracking not found under ${REPO_ROOT}. Wrong repository checkout?" >&2
  exit 1
}

if [[ "${INSTALL_DEPS}" == "1" ]]; then
  echo "=== INSTALL_DEPS=1: running install_deps.sh (container only) ==="
  bash scripts/vast/install_deps.sh
fi

if [[ "${PIP_INSTALL}" == "1" ]]; then
  echo "=== Installing tracking requirements ==="
  pip3 install -r requirements-tracking.txt
fi

if [[ ! -f "${WEIGHTS}" ]]; then
  echo "=== Weights missing, downloading ==="
  MODEL_DIR="${MODEL_DIR}" bash scripts/vast/download_yolov8n_crowdhuman.sh
fi
[[ -f "${WEIGHTS}" ]] || { echo "ERROR: weights not found: ${WEIGHTS}" >&2; exit 1; }

[[ -d "${MOT17_ROOT}/MOT17/train/${MOT17_SEQ}/img1" ]] || {
  echo "ERROR: MOT17 sequence not found: ${MOT17_ROOT}/MOT17/train/${MOT17_SEQ}/img1" >&2
  echo "Hint: run scripts/vast/download_mot17.sh first." >&2
  exit 1
}

python3 scripts/tracking/check_dataset_layout.py \
  --mot17-root "${MOT17_ROOT}" \
  --out-json "${REPORT_DIR}/dataset_layout_check_${MOT17_SEQ}.json"

echo "=== Running winner tracking pipeline ==="
MOT17_ROOT="${MOT17_ROOT}" \
MOT17_SEQ="${MOT17_SEQ}" \
WEIGHTS="${WEIGHTS}" \
CONF="${CONF}" \
IOU="${IOU}" \
IMGSZ="${IMGSZ}" \
DEVICE="${DEVICE}" \
REPORT_DIR="${REPORT_DIR}" \
PROJECT="${PROJECT}" \
NAME="${RUN_NAME}" \
bash scripts/tracking/run_yolov8_bytetrack_mot17.sh

RUN_JSON="$(ls -t "${REPORT_DIR}/${RUN_NAME}"_*_run_report.json 2>/dev/null | head -n 1 || true)"
[[ -n "${RUN_JSON}" ]] || { echo "ERROR: No run_report found for ${RUN_NAME}" >&2; exit 1; }

PRED_TXT="$(RUN_JSON="${RUN_JSON}" python3 - <<'PY'
import json
import os
from pathlib import Path
p = Path(os.environ["RUN_JSON"])
obj = json.loads(p.read_text(encoding="utf-8"))
print(obj["artifacts"]["mot_txt"])
PY
)"
[[ -f "${PRED_TXT}" ]] || { echo "ERROR: MOT txt missing: ${PRED_TXT}" >&2; exit 1; }

echo "=== Running TrackEval ==="
MOT17_ROOT="${MOT17_ROOT}" \
MOT17_SEQ="${MOT17_SEQ}" \
TRACKER_NAME="${TRACKER_NAME}" \
PRED_TXT="${PRED_TXT}" \
OUT_DIR="${TRACKEVAL_OUT_DIR}" \
bash scripts/tracking/eval_trackeval_mot17.sh

METRICS_JSON="${TRACKEVAL_OUT_DIR}/${TRACKER_NAME}_${MOT17_SEQ}_metrics.json"
SUMMARY_MD="${TRACKEVAL_OUT_DIR}/${TRACKER_NAME}_${MOT17_SEQ}_summary.md"
[[ -f "${METRICS_JSON}" ]] || { echo "ERROR: TrackEval metrics missing: ${METRICS_JSON}" >&2; exit 1; }
[[ -f "${SUMMARY_MD}" ]] || { echo "ERROR: TrackEval summary missing: ${SUMMARY_MD}" >&2; exit 1; }

PYTHON_VERSION="$(python3 -V 2>&1 || true)"
TORCH_VERSION="$(python3 - <<'PY'
try:
    import torch
    print(torch.__version__)
except Exception:
    print("unavailable")
PY
)"
L4T_RELEASE="$(cat /etc/nv_tegra_release 2>/dev/null || echo "unavailable")"

RUN_NOTE="${LOG_DIR}/run_note_${TRACKER_NAME}_${MOT17_SEQ}_$(date -u +%Y%m%dT%H%M%SZ).md"
export PYTHON_VERSION TORCH_VERSION L4T_RELEASE
python3 - <<'PY' > "${RUN_NOTE}"
import json
import os
import subprocess
from pathlib import Path

run_json = Path(os.environ["RUN_JSON"])
metrics_json = Path(os.environ["METRICS_JSON"])
summary_md = Path(os.environ["SUMMARY_MD"])
repo_root = Path(os.environ["REPO_ROOT"])
mot17_seq = os.environ["MOT17_SEQ"]
conf = os.environ["CONF"]
iou = os.environ["IOU"]
imgsz = os.environ["IMGSZ"]
device = os.environ["DEVICE"]
tracker_name = os.environ["TRACKER_NAME"]

run = json.loads(run_json.read_text(encoding="utf-8"))
met = json.loads(metrics_json.read_text(encoding="utf-8"))
m = met.get("metrics", {})
hota = m.get("HOTA___AUC", m.get("HOTA"))

commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(repo_root), text=True).strip()

print(f"- date_utc: {subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()}")
print(f"- git_commit: {commit}")
print(f"- image_tag: {os.environ.get('IMAGE_TAG', 'unknown')}")
print(f"- python_version: {os.environ.get('PYTHON_VERSION', 'unknown')}")
print(f"- torch_version: {os.environ.get('TORCH_VERSION', 'unknown')}")
print(f"- l4t_release: {os.environ.get('L4T_RELEASE', 'unknown')}")
print(f"- sequence: {mot17_seq}")
print(f"- tracker: {tracker_name}")
print(f"- conf: {conf}")
print(f"- iou: {iou}")
print(f"- imgsz: {imgsz}")
print(f"- device: {device}")
print(f"- fps_e2e: {run.get('fps_e2e')}")
print(f"- latency_ms_e2e: {run.get('latency_ms_e2e')}")
print(f"- MOTA: {m.get('MOTA')}")
print(f"- IDF1: {m.get('IDF1')}")
print(f"- HOTA: {hota}")
print(f"- run_report: {run_json}")
print(f"- pred_txt: {run['artifacts']['mot_txt']}")
print(f"- trackeval_metrics: {metrics_json}")
print(f"- trackeval_summary: {summary_md}")
PY

echo "=== Done ==="
echo "run_report: ${RUN_JSON}"
echo "pred_txt: ${PRED_TXT}"
echo "trackeval_metrics: ${METRICS_JSON}"
echo "trackeval_summary: ${SUMMARY_MD}"
echo "run_note: ${RUN_NOTE}"
