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

Config: `configs/datasets/crowdhuman_val.yaml`.
