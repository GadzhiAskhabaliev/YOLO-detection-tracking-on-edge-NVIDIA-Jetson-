# Pedestrian detection and tracking on edge hardware

This repository supports **quantitative comparison** of pedestrian detectors aimed ultimately at **NVIDIA Jetson-class edge deployment**. Benchmarks are run primarily on cloud GPUs (e.g. Vast.ai); throughput and energy targets remain board-local measurements elsewhere.

## What we benchmark

All detectors tracked here belong to **Group B** — crowded-scene pedestrian models enumerated in [`docs/group_b_pedestrian_detectors.yaml`](docs/group_b_pedestrian_detectors.yaml):

| ID | Short name | Description |
|----|------------|-------------|
| 4 | **CrowdDet** | RCNN EMD Refine, ResNet-50 + FPN |
| 5 | **Pedestron** | Cascade Mask R-CNN, HRNet-W32 |
| 6 | **YOLOv8n-CH** | YOLOv8 nano trained on CrowdHuman (`integration: ultralytics`) |
| 7 | **FreeYOLO** | YOLOX-family; MOT17-oriented checkpoints (`integration: manual`) |
| 8 | **PeopleNet** | NVIDIA TAO / NGC pipeline |

**Currently persisted runs** under [`results/runs/`](results/runs/) cover slots **6** and **7** only (YOLOv8n-CrowdHuman and two FreeYOLO CrowdHuman variants). Slots **4, 5, and 8** require **upstream codebases** (CrowdDet, Pedestron, TAO) that are **not vendored here**.

### Scientific scope (explicit boundary)

This repository **closes one reproducible slice** of the wider study:

- **We integrate**: Ultralytics for slot 6, plus tooling + reproducible scripts for FreeYOLO on CrowdHuman val (slot 7).
- **We do not ship**: MMDetection installs, CrowdDet/Pedestron forks, NVIDIA TAO containers, or a universal multi-framework launcher.

Comparable detection metrics across heterogeneous stacks should therefore rely on **shared ground truth** and **`scripts/eval_coco_predictions.py`** once upstream models emit COCO-style box lists (see [`docs/benchmark_metrics_schema.md`](docs/benchmark_metrics_schema.md)). FPS remains backend-specific and must be recorded under our canon (`fps_forward` vs `fps_predict`) with definitions in each run’s `notes`.

## Layout

| Path | Purpose |
|------|---------|
| [`scripts/vast/`](scripts/vast/) | Cloud bootstrap, datasets, CrowdHuman→YOLO layout |
| [`configs/datasets/`](configs/datasets/) | Ultralytics dataset YAML (CrowdHuman val) |
| [`docs/group_b_pedestrian_detectors.yaml`](docs/group_b_pedestrian_detectors.yaml) | Canonical detector list & URLs |
| [`docs/group_b_benchmarks.md`](docs/group_b_benchmarks.md) | Operational notes for Group B runs |
| [`docs/benchmark_metrics_schema.md`](docs/benchmark_metrics_schema.md) | JSON schema & metric definitions |
| [`results/runs/`](results/runs/) | One JSON file per benchmark run |
| [`scripts/bench_runner.py`](scripts/bench_runner.py) | Orchestration + README / summary refresh |
| [`scripts/eval_coco_predictions.py`](scripts/eval_coco_predictions.py) | Unified COCOeval on dumped predictions |

## Benchmark table (auto-generated)

After each `bench_runner.py` save/merge, the block below updates automatically.

<!-- TABLE_START -->

| Backend | Model | mAP50 | mAP50-95 | FPS (forward) | FPS (predict) | MOTA | TRT FP16 | Date |
|---------|-------|-------|----------|---------------|---------------|------|----------|------|
| ultralytics_yolo | yolov8n_crowdhuman | 0.7471 | 0.4642 | 117.368 | 127.104 |  | no | 2026-05-09T14:37:40Z |
| freeyolo | freeyolo_ch_tiny | 0.7166 | 0.3564 | 93.256 | 34.588 |  | no | 2026-05-09T14:33:28Z |
| freeyolo | freeyolo_yolox_mot17 | 0.6822 | 0.3204 | 57.935 | 23.988 |  | no | 2026-05-09T14:47:53Z |

<!-- TABLE_END -->

Detailed narratives live in [`results/benchmark_summary.md`](results/benchmark_summary.md). ASCII comparison: [`results/model_comparison.md`](results/model_comparison.md) (`scripts/generate_comparison_table.py`).

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
