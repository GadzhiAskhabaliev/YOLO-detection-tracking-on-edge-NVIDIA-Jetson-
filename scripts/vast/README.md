# Vast.ai / cloud scripts

Run from repo root **or** use absolute paths below. Use **tmux** so SSH disconnects do not kill long jobs.

## Layout vs instances

Git holds **code and configs**. **Datasets and weights** stay on instance disk (for example `/workspace`) and must not be committed (see `.gitignore`). Clone once per machine, then `git pull` before sessions.

## Environment (optional)

| Variable | Default |
|----------|---------|
| `CROWDHUMAN_ROOT` | `/workspace/data/crowdhuman` |
| `MOT17_ROOT` | `/workspace/data/mot17` |
| `MODEL_DIR` | `/workspace/models` |
| `TORCH_INDEX_URL` | see `install_deps.sh` |

## Order (fresh GPU box)

1. `bash scripts/vast/install_deps.sh`
2. `bash scripts/vast/download_crowdhuman_val.sh`
3. `bash scripts/vast/download_mot17.sh`
4. `bash scripts/vast/convert_crowdhuman_odgt.sh`
5. `bash scripts/vast/prepare_crowdhuman_yolo_layout.sh`
6. Edit `configs/datasets/crowdhuman_val.yaml` if `path:` is not `/workspace/data/crowdhuman/yolo`
7. `bash scripts/vast/download_yolov8n_crowdhuman.sh`
8. `python3 scripts/vast/bench_yolo_fps.py --weights "$MODEL_DIR/yolov8n_crowdhuman.pt" --out-json /workspace/bench_yolov8n.json`
9. `bash scripts/vast/val_yolov8_crowdhuman.sh`

Or steps 1–7 via:

```bash
bash scripts/vast/run_cloud_bootstrap.sh
```

## Scripts

| Script | Role |
|--------|------|
| `install_deps.sh` | apt + pip + CUDA PyTorch + Ultralytics deps |
| `download_crowdhuman_val.sh` | HF CrowdHuman val zip + `annotation_val.odgt` |
| `download_mot17.sh` | MOT17 via kagglehub |
| `crowdhuman_odgt_to_yolo.py` | ODGT → `labels_val/` |
| `convert_crowdhuman_odgt.sh` | Wrapper |
| `prepare_crowdhuman_yolo_layout.sh` | Symlinks → `yolo/images/val`, `yolo/labels/val` |
| `download_yolov8n_crowdhuman.sh` | YOLOv8n CrowdHuman weights → `yolov8n_crowdhuman.pt` |
| `bench_yolo_fps.py` | Predict-loop FPS JSON; `--record-model-name` → `results/runs/` via `bench_runner.py` |
| `../bench_runner.py` | Unified bench (forward / predict / val), README + `results/benchmark_summary.md` |
| `../run_group_b_benchmarks.sh` | Group B driver (legacy orchestration helper) |
| `../group_b/run_remaining_models.sh` | FreeYOLO path + CrowdDet/Pedestron/PeopleNet notes |
| `../plot_group_b_results.py` | Group B figure generator (optional, local analysis only) |
| `val_yolov8_crowdhuman.sh` | `model.val` on CrowdHuman yaml |
| `run_cloud_bootstrap.sh` | install + datasets + convert + layout + weights |
| `../tracking/check_dataset_layout.py` | Validate CrowdHuman + MOT17 layout and print summary |
| `../tracking/run_yolov8_bytetrack_mot17.py` | YOLOv8n + ByteTrack run with FPS/latency report |
| `../tracking/run_yolov8_boxmot_mot17.py` | YOLOv8n + BoxMOT tracker run (BoT-SORT/HybridSORT/DeepOCSORT/StrongSORT) |
| `../tracking/run_yolov8_strongsort_mot17.py` | YOLOv8n + StrongSORT run with FPS/latency report |
| `../tracking/run_yolov8_botsort_mot17.sh` | BoT-SORT wrapper for MOT17 runs |
| `../tracking/run_yolov8_hybridsort_mot17.sh` | HybridSORT wrapper for MOT17 runs |
| `../tracking/run_yolov8_deepocsort_mot17.sh` | DeepOCSORT wrapper for MOT17 runs |
| `../tracking/export_ultralytics_to_mot.py` | Convert raw tracks JSON to MOTChallenge txt |
| `../tracking/eval_trackeval_mot17.sh` | TrackEval wrapper (HOTA/CLEAR/Identity, outputs json+md) |
| `../tracking/run_tracking_benchmarks.py` | 3-config tracking sweep and benchmark aggregation |
| `../tracking/run_tracking_benchmarks_strongsort.py` | 3-config StrongSORT sweep and benchmark aggregation |
| `../tracking/run_tracking_benchmarks_boxmot.py` | 3-config sweep for selected BoxMOT tracker |
| `../tracking/run_tracking_benchmarks_top_trackers.sh` | Sequential sweep for BoT-SORT + HybridSORT + DeepOCSORT |

Config: `configs/datasets/crowdhuman_val.yaml`.

## Tracking quick run (MOT17)

Default tracking paths auto-resolve by:
`EDGE_WORK_ROOT` -> `/workspace` -> `/root/workspace` -> `/root`.

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

TrackEval:

```bash
MOT17_SEQ=MOT17-02-FRCNN \
PRED_TXT=results/tracking/<run_tag>.txt \
TRACKER_NAME=yolov8_bytetrack \
bash scripts/tracking/eval_trackeval_mot17.sh
```

StrongSORT benchmark sweep:

```bash
python3 scripts/tracking/run_tracking_benchmarks_strongsort.py --mot17-seq MOT17-02-FRCNN
python3 scripts/tracking/run_tracking_benchmarks_boxmot.py --tracker-type botsort --mot17-seq MOT17-02-FRCNN
python3 scripts/tracking/run_tracking_benchmarks_boxmot.py --tracker-type hybridsort --mot17-seq MOT17-02-FRCNN
python3 scripts/tracking/run_tracking_benchmarks_boxmot.py --tracker-type deepocsort --mot17-seq MOT17-02-FRCNN
bash scripts/tracking/run_tracking_benchmarks_top_trackers.sh
```

Note: evaluation runs against `MOT17 train` GT for development iteration.
