# Pedestrian detection and tracking on edge hardware (YOLO)

Quantitative comparison of **Group B** pedestrian detectors for **Jetson-class** targets.
Timing runs are usually on a cloud GPU; on-board FPS/power stay out of this repo except as recorded numbers in run JSON.

## What we benchmark

All detectors tracked here belong to **Group B** — crowded-scene pedestrian models enumerated in [`docs/group_b_pedestrian_detectors.yaml`](docs/group_b_pedestrian_detectors.yaml):

| ID | Short name | Description |
|----|------------|-------------|
| 4 | **CrowdDet** | RCNN EMD Refine, ResNet-50 + FPN |
| 5 | **Pedestron** | Cascade Mask R-CNN, HRNet-W32 |
| 6 | **YOLOv8n-CH** | YOLOv8 nano trained on CrowdHuman (`integration: ultralytics`) |
| 7 | **FreeYOLO** | YOLOX-family; MOT17-oriented checkpoints (`integration: manual`) |
| 8 | **PeopleNet** | NVIDIA TAO / NGC pipeline |

**Persisted runs** in [`results/runs/`](results/runs/) are slots **6** and **7** (YOLOv8n-CrowdHuman, two FreeYOLO CrowdHuman checkpoints). Slots **4, 5, 8** are tracked in the manifest only; their training/inference code lives in **other repositories** (CrowdDet, Pedestron/MMDet, NVIDIA TAO), not here.

### What lives here vs elsewhere

| Here (this git repo) | Outside (other repos / host disk) |
|----------------------|-----------------------------------|
| `bench_runner.py`, `eval_coco_predictions.py`, dump scripts, `scripts/vast/*`, Group B shell drivers | Ultralytics install, CUDA wheels, datasets under `data/`, weights under `models/` ([`.gitignore`](.gitignore)) |
| CrowdHuman val YAML, MOT17→COCO GT scripts, unified metric docs | FreeYOLO **upstream tree** cloned on the GPU box (`FREEYOLO_HOME`), its venv, `eval.py` for CrowdHuman-native runs |
| `results/runs/*.json`, committed `results/logs/*.log` transcripts | CrowdDet / MMDet export pipelines (e.g. [CV-MMdetect](https://github.com/GadzhiAskhabaliev/CV-MMdetect)) |

Cross-model **AP** (keys **`AP50`**, **`AP50-95`**, …) use the same GT and **`scripts/eval_coco_predictions.py`** once each stack emits a COCO-style DT list. **FPS** is defined per backend in each run’s `notes` (`fps_forward` vs `fps_predict`).

## Canonical docs

- Main unified table (all models, logs, detailed artifacts): [`docs/crowdhuman_val_full_metrics_table.md`](docs/crowdhuman_val_full_metrics_table.md)
- Metric definitions and benchmark logic: [`docs/benchmark_metrics_schema.md`](docs/benchmark_metrics_schema.md)

## Layout

| Path | Purpose |
|------|---------|
| [`scripts/vast/`](scripts/vast/) | Cloud bootstrap, datasets, CrowdHuman→YOLO layout |
| [`configs/datasets/`](configs/datasets/) | Ultralytics dataset YAML (CrowdHuman val) |
| [`docs/group_b_pedestrian_detectors.yaml`](docs/group_b_pedestrian_detectors.yaml) | Canonical detector list & URLs |
| [`docs/benchmark_metrics_schema.md`](docs/benchmark_metrics_schema.md) | JSON schema & metric definitions |
| [`docs/crowdhuman_val_full_metrics_table.md`](docs/crowdhuman_val_full_metrics_table.md) | Unified benchmark table (all models + links to logs/artifacts) |
| [`results/runs/`](results/runs/) | One JSON file per benchmark run |
| [`scripts/bench_runner.py`](scripts/bench_runner.py) | Orchestration + README / summary refresh |
| [`scripts/eval_coco_predictions.py`](scripts/eval_coco_predictions.py) | Unified COCOeval on dumped predictions |

## Benchmark table (auto-generated)

After each `bench_runner.py` save/merge, the block below updates automatically.

<!-- TABLE_START -->

| Backend | Model | AP25 | AP50 | AP75 | AP50-95 | AR_coco | grP50 | grR50 | grF50 | cAR50 | FPS (forward) | FPS (predict) | MOTA | TRT FP16 | Date |
|---------|-------|------|------|------|---------|---------|-------|-------|-------|-------|---------------|---------------|------|----------|------|
| freeyolo | freeyolo_ch_tiny | 0.8608 | 0.7164 | 0.3084 | 0.3563 | 0.456 | 0.6894 | 0.7324 | 0.3106 | 0.8029 | 93.256 | 34.588 |  | no | 2026-05-09T14:33:28Z |
| freeyolo | freeyolo_yolox_mot17 | 0.8414 | 0.6819 | 0.2595 | 0.3202 | 0.4239 | 0.661 | 0.6948 | 0.339 | 0.7807 | 72.277 | 31.659 |  | no | 2026-05-11T21:16:29Z |
| ultralytics_yolo | yolov8n_crowdhuman | 0.8102 | 0.5703 | 0.2312 | 0.2716 | 0.4022 | 0.793 | 0.4334 | 0.207 | 0.7226 | 117.368 | 127.104 |  | no | 2026-05-09T14:37:40Z |
| mmdet | fcos_r50_crowdhuman | 0.5425 | 0.3284 | 0.1108 | 0.144 | 0.2938 | 0.7714 | 0.0845 | 0.2286 | 0.5899 |  |  |  | no | 2026-05-11T00:00:00Z |
| mmdet | ssd300_crowdhuman | 0.5976 | 0.2874 | 0.0473 | 0.0965 | 0.181 | 0.7132 | 0.1741 | 0.2868 | 0.4634 |  |  |  | no | 2026-05-11T00:00:00Z |
| onnx_runtime | peoplenet_crowdhuman | 0.3327 | 0.2076 | 0.0363 | 0.0717 | 0.1021 | 0.7642 | 0.1872 | 0.2358 | 0.2401 | 214.7649 | 147.8996 |  | no | 2026-05-12T15:09:19Z |

<!-- TABLE_END -->

### Unified AP / COCOeval (single evaluator)

Main source of truth (all model rows, full metric dictionary, run/log artifacts): [`docs/crowdhuman_val_full_metrics_table.md`](docs/crowdhuman_val_full_metrics_table.md).  
Metric math and protocol details: [`docs/benchmark_metrics_schema.md`](docs/benchmark_metrics_schema.md).

Per-run write-ups: [`results/benchmark_summary.md`](results/benchmark_summary.md). ASCII table: [`results/model_comparison.md`](results/model_comparison.md) (`scripts/generate_comparison_table.py`).

## Commands

Full Ultralytics run (forward + predict + validation):

```bash
python3 scripts/bench_runner.py \
  --model-name yolov8n_crowdhuman \
  --weights /workspace/models/yolov8n_crowdhuman.pt \
  --weights-hub yakhyo/yolov8-crowdhuman \
  --bench-mode all \
  --data-yaml configs/datasets/crowdhuman_val.yaml
```

Predict FPS helper (`bench_runner`-compatible JSON):

```bash
python3 scripts/vast/bench_yolo_fps.py \
  --weights /workspace/models/yolov8n_crowdhuman.pt \
  --record-model-name yolov8n_crowdhuman \
  --weights-hub yakhyo/yolov8-crowdhuman
```

Group B orchestrator:

```bash
bash scripts/run_group_b_benchmarks.sh
```

Merge tracking metrics:

```bash
python3 scripts/bench_runner.py \
  --merge-json results/runs/yolov8n_crowdhuman_2026-05-09T120000Z.json \
  --tracking-json '{"mot17_seq":"MOT17-02","MOTA":0.68,"HOTA":0.52,"IDF1":0.61}'
```

## Repository vs cloud disk

**Git** holds scripts and tiny configs. **Datasets, checkpoints, and large logs** stay on the instance (for example `/workspace`) per [`.gitignore`](.gitignore). Workflow notes: [`scripts/vast/README.md`](scripts/vast/README.md).
