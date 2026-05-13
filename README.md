# Pedestrian detection and tracking on edge hardware (YOLO)

Quantitative comparison of **Group B** pedestrian detectors for **Jetson-class** targets.
Timing runs are usually on a cloud GPU; on-board FPS/power stay out of this repo except as recorded numbers in run JSON.

## What we benchmark

**Group B** is the manifest in [`docs/group_b_pedestrian_detectors.yaml`](docs/group_b_pedestrian_detectors.yaml) (IDs **4–8**). Each row’s `integration` is either **`ultralytics`** (bench code in this repo) or **`manual`** (export + eval elsewhere; we only store unified JSON/logs when provided).

| ID | Short name | Description (from manifest + how we measure) |
|----|------------|------------------------------------------------|
| 4 | **CrowdDet** | *Detection in Crowded Scenes* (CVPR’20): RCNN **EMD Refine**, ResNet-50 + FPN. `integration: manual`. Persisted: unified COCOeval on CrowdHuman val. |
| 5 | **Pedestron** | Cascade **Mask** R-CNN, HRNet-W32. `integration: manual`. Listed for coverage; **no** `results/runs/` row yet — MMDet/Pedestron export is out-of-tree. |
| 6 | **YOLOv8n-CH** | YOLOv8 **nano**, CrowdHuman-trained (HF [`yakhyo/yolov8-crowdhuman`](https://huggingface.co/yakhyo/yolov8-crowdhuman)). `integration: ultralytics`. |
| 7 | **FreeYOLO** | [FreeYOLO](https://github.com/yjh0410/FreeYOLO) YOLOX-family heads. `integration: manual`. **CrowdHuman val** rows use author **`yolo_free_*_ch.pth`** checkpoints (e.g. tiny + nano); manifest `bench_slug` still points at the MOT17-oriented naming for one slot. |
| 8 | **PeopleNet** | [NVIDIA PeopleNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet), **ResNet-34**, TAO/NGC training. `integration: manual`. **Persisted run** = NGC **pruned INT8 ONNX** + **ONNX Runtime** (CUDA EP) on CrowdHuman val + unified `eval_coco_predictions.py` — not the full TAO `.etlt` / TensorRT-only path from the catalog blurb. |

**Also in the unified CrowdHuman val table (same GT + evaluator, not YAML IDs 4–8):** MMDetection **FCOS R50** and **SSD300** COCO-pretrained dumps — see [`docs/crowdhuman_val_full_metrics_table.md`](docs/crowdhuman_val_full_metrics_table.md).

**Persisted runs** under [`results/runs/`](results/runs/): slots **4**, **6**, **7**, **8** as above, plus the two **mmdet** comparators. Slot **5** (Pedestron) remains manifest-only until a run JSON is added.

### What lives here vs elsewhere

| Here (this git repo) | Outside (other repos / host disk) |
|----------------------|-----------------------------------|
| `bench_runner.py`, `eval_coco_predictions.py`, dump scripts, `scripts/vast/*`, Group B shell drivers | Ultralytics install, CUDA wheels, datasets under `data/`, weights under `models/` ([`.gitignore`](.gitignore)) |
| CrowdHuman val YAML, MOT17→COCO GT scripts, unified metric docs | FreeYOLO **upstream tree** cloned on the GPU box (`FREEYOLO_HOME`), its venv, `eval.py` for CrowdHuman-native runs |
| `results/runs/*.json`, committed `results/logs/*.log`, `results/crowdhuman/*.json` (metric mirrors) | CrowdDet eval fork ([CrowdDet-detection](https://github.com/GadzhiAskhabaliev/CrowdDet-detection)), MMDet export pipelines (e.g. [CV-MMdetect](https://github.com/GadzhiAskhabaliev/CV-MMdetect)), PeopleNet NGC ONNX + host export/bench tree |

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

## One-command pipeline (Vast/Jetson)

For a reproducible YOLO-only run from a fresh machine, use:

```bash
bash scripts/run_yolo_edge_pipeline.sh
```

The pipeline executes:

1. dependency install (`scripts/vast/install_deps.sh`)
2. dataset + weights bootstrap (`scripts/vast/run_cloud_bootstrap.sh`)
3. YOLO benchmark (`scripts/bench_runner.py --bench-mode all`)
4. MOT17 dump for tracker input (`scripts/dump_ultralytics_mot17.py`)
5. unified metrics + merge into latest run JSON (`scripts/eval_coco_predictions.py` + `--merge-json`)

Useful overrides:

```bash
SKIP_INSTALL=1 \
SKIP_BOOTSTRAP=1 \
WEIGHTS=/workspace/models/yolov8n_crowdhuman.pt \
DEVICE=cuda:0 \
bash scripts/run_yolo_edge_pipeline.sh
```

## Tracking pipeline: YOLOv8n + ByteTrack/BoxMOT trackers

Tracking utilities are isolated in `scripts/tracking/` and keep detection benchmarks intact.
Path defaults are auto-detected in this order: `EDGE_WORK_ROOT`, `/workspace`, `/root/workspace`, `/root`.
So scripts run on instances without `/workspace` too.

Quick start on a Vast instance:

```bash
bash scripts/vast/install_deps.sh
pip install -r requirements-tracking.txt
bash scripts/vast/download_mot17.sh
python3 scripts/tracking/check_dataset_layout.py --out-json results/tracking/dataset_layout_check.json
MOT17_SEQ=MOT17-02-FRCNN bash scripts/tracking/run_yolov8_bytetrack_mot17.sh
MOT17_SEQ=MOT17-02-FRCNN bash scripts/tracking/run_yolov8_strongsort_mot17.sh
MOT17_SEQ=MOT17-02-FRCNN bash scripts/tracking/run_yolov8_botsort_mot17.sh
MOT17_SEQ=MOT17-02-FRCNN bash scripts/tracking/run_yolov8_hybridsort_mot17.sh
MOT17_SEQ=MOT17-02-FRCNN bash scripts/tracking/run_yolov8_deepocsort_mot17.sh
```

Convert/export (if you have only raw tracking JSON):

```bash
python3 scripts/tracking/export_ultralytics_to_mot.py \
  --in-json results/tracking/<run_tag>_raw_tracks.json \
  --out-txt results/tracking/<run_tag>.txt \
  --strict
```

TrackEval (MOTA/IDF1/HOTA):

```bash
MOT17_SEQ=MOT17-02-FRCNN \
TRACKER_NAME=yolov8_bytetrack \
PRED_TXT=results/tracking/<run_tag>.txt \
bash scripts/tracking/eval_trackeval_mot17.sh

MOT17_SEQ=MOT17-02-FRCNN \
TRACKER_NAME=yolov8_strongsort \
PRED_TXT=results/tracking/<run_tag>.txt \
bash scripts/tracking/eval_trackeval_mot17.sh
```

Important: this TrackEval flow uses `MOT17 train` GT and should be treated as a **development benchmark**
(not an official MOTChallenge test-server submission).

Default benchmark sweep for three configs:

```bash
python3 scripts/tracking/run_tracking_benchmarks.py --mot17-seq MOT17-02-FRCNN
python3 scripts/tracking/run_tracking_benchmarks_strongsort.py --mot17-seq MOT17-02-FRCNN
python3 scripts/tracking/run_tracking_benchmarks_boxmot.py --tracker-type botsort --mot17-seq MOT17-02-FRCNN
python3 scripts/tracking/run_tracking_benchmarks_boxmot.py --tracker-type hybridsort --mot17-seq MOT17-02-FRCNN
python3 scripts/tracking/run_tracking_benchmarks_boxmot.py --tracker-type deepocsort --mot17-seq MOT17-02-FRCNN

# sequential automation for BoT-SORT + HybridSORT + DeepOCSORT
bash scripts/tracking/run_tracking_benchmarks_top_trackers.sh
```

Known limitations:
- Cloud GPU FPS is not equal to Jetson FPS; re-run with Jetson-specific torch/onnx/tensorrt stack.
- MOT17 evaluation quality depends on sequence choice and GT consistency (`gt/gt.txt`).
- Some instances do not expose `/workspace`; override `MOT17_ROOT` / data paths when needed.

## Benchmark table (auto-generated)

After each `bench_runner.py` save/merge, the block below updates automatically.

<!-- TABLE_START -->

| Backend | Model | AP25 | AP50 | AP75 | AP50-95 | AR_coco | grP50 | grR50 | grF50 | cAR50 | FPS (forward) | FPS (predict) | MOTA | TRT FP16 | Date |
|---------|-------|------|------|------|---------|---------|-------|-------|-------|-------|---------------|---------------|------|----------|------|
| crowddet | crowddet_rcnn_emd_refine_e30 | 0.913 | 0.8662 | 0.5586 | 0.5247 | 0.5814 | 0.7641 | 0.8533 | 0.2359 | 0.8993 |  |  |  | no | 2026-05-12T00:00:00Z |
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
