# YOLO Edge Detection + Tracking (Jetson/Vast)

![Scope](https://img.shields.io/badge/scope-3%20YOLO%20detectors%20%2B%205%20trackers-blue)
![Platform](https://img.shields.io/badge/platform-Jetson%20%7C%20Vast.ai-green)
![Tracking](https://img.shields.io/badge/tracking-ByteTrack%20%7C%20StrongSORT%20%7C%20BoT--SORT%20%7C%20HybridSORT%20%7C%20DeepOCSORT-orange)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

Focused benchmark repository for pedestrian detection and MOT tracking on edge GPUs.

## Scope

This repository is intentionally cleaned to only keep:

- **3 detectors**
  - `yolov8n_crowdhuman`
  - `freeyolo_ch_tiny`
  - `freeyolo_yolox_mot17` (FreeYOLO nano)
- **5 trackers**
  - ByteTrack
  - StrongSORT
  - BoT-SORT
  - HybridSORT
  - DeepOCSORT

## Repository map

- Detection artifacts: `results/runs/`, `results/benchmark_summary.md`, `results/model_comparison.md`
- Tracking artifacts: `results/tracking/` (kept as-is, full benchmark history)
- Jetson safe runbook: `RUNBOOK.md`
- Safe runner script: `scripts/tracking/run_jetson_winner_safe.sh`

## Quick start (cloud / workstation)

```bash
bash scripts/vast/run_cloud_bootstrap.sh
bash scripts/run_yolo_detectors_benchmarks.sh
```

This runs detection benchmarks for all 3 YOLO detectors and updates run JSON artifacts in `results/runs/`.

## Tracking baseline (winner)

```bash
MOT17_SEQ=MOT17-02-FRCNN \
WEIGHTS=/workspace/models/yolov8n_crowdhuman.pt \
CONF=0.35 \
IOU=0.7 \
bash scripts/tracking/run_yolov8_bytetrack_mot17.sh
```

Evaluate with TrackEval:

```bash
MOT17_SEQ=MOT17-02-FRCNN \
TRACKER_NAME=yolov8_bytetrack_c035_i07 \
PRED_TXT=results/tracking/<RUN_TAG>.txt \
bash scripts/tracking/eval_trackeval_mot17.sh
```

## Tracking comparison

Unified tracking comparison across all detectors and trackers:

- `results/tracking/tracker_comparison.md`
- `results/tracking/tracker_comparison.json`

Per your latest requirement, tracking benchmark artifacts were not rewritten and remain preserved.

## Jetson no-risk mode

For lab-safe execution (Docker only, no host mutation):

```bash
bash scripts/tracking/run_jetson_winner_safe.sh
```

Detailed policy and stop conditions are documented in `RUNBOOK.md`.
