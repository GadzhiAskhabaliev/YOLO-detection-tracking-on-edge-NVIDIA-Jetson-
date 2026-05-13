# Vast.ai / cloud scripts

Cloud helpers for the focused stack:
- 3 detectors: `yolov8n_crowdhuman`, `freeyolo_ch_tiny`, `freeyolo_yolox_mot17`
- 5 trackers: ByteTrack, StrongSORT, BoT-SORT, HybridSORT, DeepOCSORT

## Quick bootstrap

```bash
bash scripts/vast/run_cloud_bootstrap.sh
```

## Core scripts

| Script | Role |
| --- | --- |
| `install_deps.sh` | System + Python deps for benchmark runs |
| `download_crowdhuman_val.sh` | Download CrowdHuman validation split |
| `download_mot17.sh` | Download MOT17 dataset |
| `convert_crowdhuman_odgt.sh` | ODGT to YOLO labels |
| `prepare_crowdhuman_yolo_layout.sh` | Build YOLO folder layout |
| `download_yolov8n_crowdhuman.sh` | Download YOLOv8n CrowdHuman weights |
| `../run_yolo_detectors_benchmarks.sh` | Run detection benchmarks for all 3 YOLO models |
| `../tracking/run_jetson_winner_safe.sh` | Safe one-command Jetson Docker baseline run |

## Tracking quick start

```bash
pip install -r requirements-tracking.txt
bash scripts/vast/download_mot17.sh
python3 scripts/tracking/check_dataset_layout.py --out-json results/tracking/dataset_layout_check.json
MOT17_SEQ=MOT17-02-FRCNN bash scripts/tracking/run_yolov8_bytetrack_mot17.sh
MOT17_SEQ=MOT17-02-FRCNN bash scripts/tracking/run_yolov8_strongsort_mot17.sh
MOT17_SEQ=MOT17-02-FRCNN bash scripts/tracking/run_yolov8_botsort_mot17.sh
MOT17_SEQ=MOT17-02-FRCNN bash scripts/tracking/run_yolov8_hybridsort_mot17.sh
MOT17_SEQ=MOT17-02-FRCNN bash scripts/tracking/run_yolov8_deepocsort_mot17.sh
```

Note: TrackEval metrics are computed on `MOT17 train` GT and used as development benchmarks.
