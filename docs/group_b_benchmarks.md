# Group B: runs and reporting

Model manifest: [`group_b_pedestrian_detectors.yaml`](group_b_pedestrian_detectors.yaml). Metric contract: [`benchmark_metrics_schema.md`](benchmark_metrics_schema.md).

Unified CrowdHuman val (pycocotools mAP / AR): [`benchmark_unified_cocoeval.md`](benchmark_unified_cocoeval.md).  
CrowdHuman + MOT17: [`benchmark_group_b_unified_two_domains.md`](benchmark_group_b_unified_two_domains.md).

## Where artifacts go

- **`results/runs/*.json`** → **`results/benchmark_summary.md`** and the README table (`bench_runner`).
- `results/logs/*.log` — keep CrowdHuman **and** MOT17 transcripts: YOLOv8 unified eval, FreeYOLO `eval.py` tees (`freeyolo_yolo_free_*`), MOT17 dump+eval tees (`*_mot17_train_unified_*.log`); see [`benchmark_group_b_unified_two_domains.md`](benchmark_group_b_unified_two_domains.md).
- Full shell transcript: `bash scripts/run_group_b_benchmarks.sh 2>&1 | tee results/logs/group_b_run_<UTC>.log`

## Automated runs from this repository

### One entrypoint (YOLOv8n-CH + FreeYOLO)

```bash
bash scripts/run_group_b_benchmarks.sh
```

- Slot **6 — YOLOv8n-CrowdHuman**: `bench_runner.py` (forward / predict / val).
- Slot **7 — FreeYOLO**: separate venv under `GROUP_B_ROOT`, full CrowdHuman val (slow).
- Slots **4, 5, 8** — only pointers; full benchmarks live in their upstream repos / Docker.

YOLOv8 only: `GROUP_B_EXTRA_MODELS=0 bash scripts/run_group_b_benchmarks.sh`  
Skip FreeYOLO: `GROUP_B_FREEYOLO=0 bash scripts/group_b/run_remaining_models.sh`

### FreeYOLO variants (tiny / large)

Tiny example:

```bash
FREEYOLO_VARIANT=yolo_free_tiny \
FREEYOLO_WEIGHT_URL=https://github.com/yjh0410/FreeYOLO/releases/download/weight/yolo_free_tiny_ch.pth \
FREEYOLO_WEIGHT_PATH="${MODEL_DIR:-/workspace/models}/yolo_free_tiny_ch.pth" \
bash scripts/group_b/run_freeyolo_crowdhuman.sh
```

Large: `-v yolo_free_large` and `yolo_free_large_ch.pth` (see FreeYOLO releases).

### YOLOv8-only via CLI

```bash
python3 scripts/bench_runner.py --model-name yolov8n_crowdhuman \
  --weights /path/to/yolov8n_crowdhuman.pt \
  --weights-hub yakhyo/yolov8-crowdhuman \
  --bench-mode all \
  --group B --detector-id 6
```

| Variable | Default |
|----------|---------|
| `MODEL_DIR` | `/workspace/models` |
| `DATA_YAML` | `configs/datasets/crowdhuman_val.yaml` |

## CrowdDet, Pedestron, PeopleNet

Different codebases (CrowdDet repo, Pedestron / MMDetection, NVIDIA TAO). Remote GPU flow (CrowdDet dump → DT JSON → `eval_coco_predictions.py`): see **[group_b_remote_mmdet_bridge.md](group_b_remote_mmdet_bridge.md)** and repo **[CV-MMdetect](https://github.com/GadzhiAskhabaliev/CV-MMdetect)**.

After inference and FPS in **their** stack:

1. Record metrics in `results/runs/<slug>.json` or merge patches.
2. Set **`group`: `"B"`** and **`detector_id`** `4`, `5`, or `8` for plots.

```bash
python3 scripts/bench_runner.py \
  --merge-json results/runs/peoplenet_resnet34_2026.json \
  --patch-json /tmp/group_b_meta.json
```

For comparable **mAP** across stacks, prefer **`eval_coco_predictions.py`** on dumped boxes vs the same GT `val.json`.

## FreeYOLO maintenance

- **NumPy 2.x**: venv pins `numpy<2`; `patch_freeyolo_numpy_aliases.py` patches sources if needed.
- **PyTorch 2.6+ `weights_only`**: `run_freeyolo_crowdhuman.sh` runs `patch_freeyolo_torch_load.py`; or run it manually after clone.

## Plots

```bash
pip install matplotlib pyyaml
python3 scripts/plot_group_b_results.py
```

Outputs:

- `results/figures/group_b_scatter_map_fps.png`
- `results/figures/group_b_map50_bars.png`
- `results/group_b_report.md`

## Closure checklist for missing slots

1. Pin framework version and reproduction command.
2. Use the **same** GT `val.json` / `image_id` set as other committed runs.
3. Produce COCO-style DT list → `eval_coco_predictions.py --strict`.
4. Merge into `results/runs/`; add FPS via second merge if needed.
5. Regenerate plots.

Until those steps complete, manifest slots stay empty in tables — expected.
