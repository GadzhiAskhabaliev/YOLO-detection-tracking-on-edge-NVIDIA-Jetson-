# CrowdHuman val — unified detection metrics (`eval_coco_predictions.py`)

Cross-dataset summary (CrowdHuman + MOT17, all three models): [`benchmark_group_b_unified_two_domains.md`](benchmark_group_b_unified_two_domains.md).

This table is **not** auto-synced with `bench_runner.py`. It exists so **AP / recall** come from **one** procedure: **`scripts/eval_coco_predictions.py`** (pycocotools `COCOeval`, bbox IoU).

| Backend | Model | AP50 | AP50-95 | Recall (COCO AR maxDets=100) | FPS forward | FPS predict | Source run JSON |
|---------|-------|-------|--------|-------------------------------|-------------|-------------|-----------------|
| ultralytics_yolo | yolov8n_crowdhuman | 0.570286 | 0.271584 | 0.402259 | 117.368 | 127.104 | [`yolov8n_crowdhuman_2026-05-09T143848Z.json`](../results/runs/yolov8n_crowdhuman_2026-05-09T143848Z.json) + log [`yolov8n_crowdhuman_unified_cocoeval_2026-05-11T140530Z.log`](../results/logs/yolov8n_crowdhuman_unified_cocoeval_2026-05-11T140530Z.log) |
| freeyolo | freeyolo_ch_tiny | 0.716557 | 0.356380 | 0.456 | 93.256 | 34.588 | [`freeyolo_ch_tiny_2026-05-09T143328Z.json`](../results/runs/freeyolo_ch_tiny_2026-05-09T143328Z.json) + tee [`freeyolo_yolo_free_tiny_20260509T141227Z.log`](../results/logs/freeyolo_yolo_free_tiny_20260509T141227Z.log) |
| freeyolo | freeyolo_yolox_mot17 | 0.682212 | 0.320365 | 0.424 | 57.935 | 23.988 | [`freeyolo_yolox_mot17_2026-05-09T144753Z.json`](../results/runs/freeyolo_yolox_mot17_2026-05-09T144753Z.json) + tee [`freeyolo_yolo_free_nano_20260509T143905Z.log`](../results/logs/freeyolo_yolo_free_nano_20260509T143905Z.log) |

FreeYOLO rows reuse **`metrics`** from each run JSON (upstream `eval.py` + pycocotools `COCOeval` on CrowdHuman val; full console tee in the `freeyolo_yolo_free_*.log` files above).

YOLOv8 row: `dump_ultralytics_coco_dt.py` then `eval_coco_predictions.py` on the same `val.json` as FreeYOLO. README `bench_runner` AP may differ (Ultralytics `model.val()` protocol). Tee: [`yolov8n_crowdhuman_unified_cocoeval_2026-05-11T140530Z.log`](../results/logs/yolov8n_crowdhuman_unified_cocoeval_2026-05-11T140530Z.log).

Reproduce:

1. Dump predictions (same `val.json` / bridge as FreeYOLO):

```bash
python3 scripts/dump_ultralytics_coco_dt.py \
  --gt-json "$VAL_JSON" \
  --images-dir "$CROWDHUMAN_ROOT/Images" \
  --weights /path/to/yolov8n_crowdhuman.pt \
  --out-json /tmp/yolov8n_ch_val_dt.json
```

2. Evaluate:

```bash
python3 scripts/eval_coco_predictions.py \
  --gt-json "$VAL_JSON" \
  --dt-json /tmp/yolov8n_ch_val_dt.json \
  --strict \
  --out-metrics-json /tmp/yolov8_unified_metrics.json \
  --out-patch-json /tmp/yolov8_unified_patch.json
```

3. Merge metrics into the existing run (keeps FPS fields):

```bash
python3 scripts/bench_runner.py \
  --merge-json results/runs/yolov8n_crowdhuman_2026-05-09T143848Z.json \
  --patch-json /tmp/yolov8_unified_patch.json
```

4. Update **this file** table + add a log under `results/logs/` if you re-run.

The README auto-table still comes from `bench_runner`; use the unified table above when you need one `COCOeval` definition for AP/recall.
