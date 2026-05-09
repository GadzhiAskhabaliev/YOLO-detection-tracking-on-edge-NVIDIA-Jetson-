# Vast.ai / cloud scripts

Run from repo root **or** use absolute paths as below. Prefer **tmux** so SSH drops do not kill downloads (`docs/VAST_WORKFLOW.md`).

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
6. Edit `configs/datasets/crowdhuman_val.yaml` if your `path` is not `/workspace/data/crowdhuman/yolo`
7. `bash scripts/vast/download_yolov8n_crowdhuman.sh`
8. `python3 scripts/vast/bench_yolo_fps.py --weights "$MODEL_DIR/yolov8n_crowdhuman.pt" --out-json /workspace/bench_yolov8n.json`
9. `bash scripts/vast/val_yolov8_crowdhuman.sh`

Or run steps 1–7 together:

```bash
bash scripts/vast/run_cloud_bootstrap.sh
```

## Files

| Script | Role |
|--------|------|
| `install_deps.sh` | apt + pip + CUDA PyTorch wheels + ultralytics deps |
| `download_crowdhuman_val.sh` | HF CrowdHuman val zip + `annotation_val.odgt` |
| `download_mot17.sh` | MOT17 via kagglehub |
| `crowdhuman_odgt_to_yolo.py` | ODGT → `labels_val/` (called by `convert_crowdhuman_odgt.sh`) |
| `convert_crowdhuman_odgt.sh` | Wrapper with default paths |
| `prepare_crowdhuman_yolo_layout.sh` | Symlinks → `yolo/images/val`, `yolo/labels/val` |
| `download_yolov8n_crowdhuman.sh` | CrowdHuman YOLOv8n (`yakhyo/yolov8-crowdhuman` → `yolov8n_best.pt`, saved as `yolov8n_crowdhuman.pt`) |
| `bench_yolo_fps.py` | Predict-loop FPS / latency JSON |
| `val_yolov8_crowdhuman.sh` | `model.val` on CrowdHuman yaml |
| `run_cloud_bootstrap.sh` | Runs install + datasets + convert + layout + weights |

Config: `configs/datasets/crowdhuman_val.yaml` (paths for Vast).
