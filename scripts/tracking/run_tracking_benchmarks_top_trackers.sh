#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

MOT17_SEQ="${MOT17_SEQ:-MOT17-02-FRCNN}"
WEIGHTS="${WEIGHTS:-}"
REID_WEIGHTS="${REID_WEIGHTS:-osnet_x0_25_msmt17.pt}"
DEVICE="${DEVICE:-cuda:0}"
IMGSZ="${IMGSZ:-640}"
MOT17_ROOT="${MOT17_ROOT:-}"
RESULTS_DIR="${RESULTS_DIR:-results/tracking}"

run_tracker() {
  local tracker_type="$1"
  local tracker_label="$2"

  local args=(
    "python3" "scripts/tracking/run_tracking_benchmarks_boxmot.py"
    "--tracker-type" "${tracker_type}"
    "--tracker-label" "${tracker_label}"
    "--mot17-seq" "${MOT17_SEQ}"
    "--reid-weights" "${REID_WEIGHTS}"
    "--device" "${DEVICE}"
    "--imgsz" "${IMGSZ}"
    "--results-dir" "${RESULTS_DIR}"
  )
  if [[ -n "${WEIGHTS}" ]]; then
    args+=("--weights" "${WEIGHTS}")
  fi
  if [[ -n "${MOT17_ROOT}" ]]; then
    args+=("--mot17-root" "${MOT17_ROOT}")
  fi
  "${args[@]}"
}

run_tracker "botsort" "botsort"
run_tracker "hybridsort" "hybridsort"
run_tracker "deepocsort" "deepocsort"
