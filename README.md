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
| `results/runs/*.json`, committed `results/logs/*.log` transcripts | CrowdDet / MMDet export pipelines (e.g. [CV-MMdetect](https://github.com/GadzhiAskhabaliev/CV-MMdetect) per `docs/group_b_remote_mmdet_bridge.md`) |

Cross-model **mAP** comparisons use the same GT and **`scripts/eval_coco_predictions.py`** once each stack emits a COCO-style DT list ([`docs/benchmark_metrics_schema.md`](docs/benchmark_metrics_schema.md)). **FPS** is defined per backend in each run’s `notes` (`fps_forward` vs `fps_predict`).

## Layout

| Path | Purpose |
|------|---------|
| [`scripts/vast/`](scripts/vast/) | Cloud bootstrap, datasets, CrowdHuman→YOLO layout |
| [`configs/datasets/`](configs/datasets/) | Ultralytics dataset YAML (CrowdHuman val) |
| [`docs/group_b_pedestrian_detectors.yaml`](docs/group_b_pedestrian_detectors.yaml) | Canonical detector list & URLs |
| [`docs/group_b_benchmarks.md`](docs/group_b_benchmarks.md) | Operational notes for Group B runs |
| [`docs/benchmark_metrics_schema.md`](docs/benchmark_metrics_schema.md) | JSON schema & metric definitions |
| [`docs/benchmark_unified_cocoeval.md`](docs/benchmark_unified_cocoeval.md) | CrowdHuman val, single `COCOeval` path |
| [`docs/benchmark_group_b_unified_two_domains.md`](docs/benchmark_group_b_unified_two_domains.md) | CrowdHuman val + MOT17 train (same evaluator) |
| [`results/runs/`](results/runs/) | One JSON file per benchmark run |
| [`scripts/bench_runner.py`](scripts/bench_runner.py) | Orchestration + README / summary refresh |
| [`scripts/eval_coco_predictions.py`](scripts/eval_coco_predictions.py) | Unified COCOeval on dumped predictions |

## Benchmark table (auto-generated)

After each `bench_runner.py` save/merge, the block below updates automatically.

<!-- TABLE_START -->

| Backend | Model | mAP50 | mAP50-95 | FPS (forward) | FPS (predict) | MOTA | TRT FP16 | Date |
|---------|-------|-------|----------|---------------|---------------|------|----------|------|
| freeyolo | freeyolo_ch_tiny | 0.7166 | 0.3564 | 93.256 | 34.588 |  | no | 2026-05-09T14:33:28Z |
| freeyolo | freeyolo_yolox_mot17 | 0.6822 | 0.3204 | 57.935 | 23.988 |  | no | 2026-05-09T14:47:53Z |
| ultralytics_yolo | yolov8n_crowdhuman | 0.5703 | 0.2716 | 117.368 | 127.104 |  | no | 2026-05-09T14:37:40Z |

<!-- TABLE_END -->

### Unified mAP (single evaluator)

CrowdHuman val (YOLOv8 dump + `eval_coco_predictions.py`; FreeYOLO numbers from the same `COCOeval` path in upstream `eval.py`): [`docs/benchmark_unified_cocoeval.md`](docs/benchmark_unified_cocoeval.md).  
CrowdHuman + MOT17 in one table: [`docs/benchmark_group_b_unified_two_domains.md`](docs/benchmark_group_b_unified_two_domains.md).  
Example tee: [`results/logs/yolov8n_crowdhuman_unified_cocoeval_2026-05-11T140530Z.log`](results/logs/yolov8n_crowdhuman_unified_cocoeval_2026-05-11T140530Z.log).

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

Regenerate Group B figures:

```bash
pip install matplotlib pyyaml
python3 scripts/plot_group_b_results.py
```

## Repository vs cloud disk

**Git** holds scripts and tiny configs. **Datasets, checkpoints, and large logs** stay on the instance (for example `/workspace`) per [`.gitignore`](.gitignore). Workflow notes: [`scripts/vast/README.md`](scripts/vast/README.md).
