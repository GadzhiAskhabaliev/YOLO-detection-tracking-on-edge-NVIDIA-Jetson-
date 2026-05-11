# Group B — unified detection metrics (CrowdHuman val + MOT17 train)

Single reference for **cross-dataset** rows when quality is computed with **`scripts/eval_coco_predictions.py`** (pycocotools `COCOeval`, bbox IoU, same DT list shape). **FPS** always comes from the existing `results/runs/*.json` microbenches unless you re-measure and merge.

Detailed CrowdHuman-only doc (repro, merge): [`benchmark_unified_cocoeval.md`](benchmark_unified_cocoeval.md).

## CrowdHuman val

| Backend | Model | mAP50 | mAP50-95 | Recall (AR maxDets=100) | FPS forward | FPS predict | Source |
|---------|-------|-------|----------|---------------------------|-------------|-------------|--------|
| ultralytics_yolo | yolov8n_crowdhuman | 0.570286 | 0.271584 | 0.402259 | 117.368 | 127.104 | [`yolov8n_crowdhuman_2026-05-09T143848Z.json`](../results/runs/yolov8n_crowdhuman_2026-05-09T143848Z.json), log [`yolov8n_crowdhuman_unified_cocoeval_2026-05-11T140530Z.log`](../results/logs/yolov8n_crowdhuman_unified_cocoeval_2026-05-11T140530Z.log) |
| freeyolo | freeyolo_ch_tiny | 0.716557 | 0.356380 | 0.456 | 93.256 | 34.588 | [`freeyolo_ch_tiny_2026-05-09T143328Z.json`](../results/runs/freeyolo_ch_tiny_2026-05-09T143328Z.json) |
| freeyolo | freeyolo_yolox_mot17 | 0.682212 | 0.320365 | 0.424 | 57.935 | 23.988 | [`freeyolo_yolox_mot17_2026-05-09T144753Z.json`](../results/runs/freeyolo_yolox_mot17_2026-05-09T144753Z.json) |

**Protocol:** YOLOv8 — `dump_ultralytics_coco_dt.py` → `eval_coco_predictions.py --strict` on FreeYOLO-bridge `val.json`. FreeYOLO — same `COCOeval` path inside upstream `eval.py` on CrowdHuman (numbers copied from run JSON).

---

## MOT17 train (detector tag **FRCNN** only)

GT: [`scripts/mot17_gt_to_coco.py`](../scripts/mot17_gt_to_coco.py) → one COCO instances JSON (e.g. `mot17_train_frcnn_gt.json`), **5316** images / **112297** GT boxes for the standard FRCNN train split (seven sequences).

| Backend | Model | mAP50 | mAP50-95 | Recall (AR maxDets=100) | FPS forward | FPS predict | Source |
|---------|-------|-------|----------|---------------------------|-------------|-------------|--------|
| ultralytics_yolo | yolov8n_crowdhuman | 0.647584 | 0.334005 | 0.427085 | 117.368 | 127.104 | Dump → eval on instance; log [`yolov8n_crowdhuman_mot17_unified_cocoeval_2026-05-11T141600Z.log`](../results/logs/yolov8n_crowdhuman_mot17_unified_cocoeval_2026-05-11T141600Z.log) (IDE buffer had no full `summarize()` block) |
| freeyolo | freeyolo_ch_tiny | **TBD** | **TBD** | **TBD** | 93.256 | 34.588 | Run [`scripts/group_b/run_freeyolo_mot17_unified_eval.sh`](../scripts/group_b/run_freeyolo_mot17_unified_eval.sh) (full log under `results/logs/`); dump: [`dump_freeyolo_mot17.py`](../scripts/group_b/dump_freeyolo_mot17.py) |
| freeyolo | freeyolo_yolox_mot17 | **TBD** | **TBD** | **TBD** | 57.935 | 23.988 | Same script with `FREEYOLO_VARIANT=yolo_free_nano` + nano weights |

**YOLOv8 MOT17 repro** (weights = same `yolov8n_crowdhuman.pt` as CrowdHuman):

```bash
# data: scripts/vast/download_mot17.sh  →  MOT17_ROOT/MOT17/train
python3 scripts/mot17_gt_to_coco.py \
  --mot17-train-root "$MOT17_ROOT/MOT17/train" \
  --det-subdir-suffix FRCNN \
  --out-json "$MOT17_ROOT/annotations/mot17_train_frcnn_gt.json"

python3 scripts/dump_ultralytics_mot17.py \
  --gt-json "$MOT17_ROOT/annotations/mot17_train_frcnn_gt.json" \
  --mot17-train-root "$MOT17_ROOT/MOT17/train" \
  --weights /path/to/yolov8n_crowdhuman.pt \
  --out-coco-dt-json /tmp/yolov8_mot17_train_dt.json \
  --mot-det-root "$MOT17_ROOT/detections/yolov8n_ch_mot17_train"

python3 scripts/eval_coco_predictions.py \
  --gt-json "$MOT17_ROOT/annotations/mot17_train_frcnn_gt.json" \
  --dt-json /tmp/yolov8_mot17_train_dt.json \
  --strict \
  --out-metrics-json /tmp/yolov8_mot17_unified_metrics.json
```

**Full log (YOLOv8):** wrap dump + eval in `{ ...; } 2>&1 | tee ~/yolov8_mot17_full.log` or re-run eval only:

```bash
python3 scripts/eval_coco_predictions.py ... 2>&1 | tee ~/yolov8_mot17_eval_full.log
```

**FreeYOLO MOT17 (dump + eval, one tee’d log)** — use FreeYOLO venv; clone/patches/weights same idea as [`run_freeyolo_crowdhuman.sh`](../scripts/group_b/run_freeyolo_crowdhuman.sh):

```bash
export MOT17_ROOT=/root/data/mot17
export FREEYOLO_HOME=/root/group_b/FreeYOLO
export FREEYOLO_VENV=/root/group_b/venv_freeyolo
export FREEYOLO_WEIGHT_PATH=/root/models/yolo_free_tiny_ch.pth
export FREEYOLO_VARIANT=yolo_free_tiny
bash scripts/group_b/run_freeyolo_mot17_unified_eval.sh
```

Nano (`freeyolo_yolox_mot17`):

```bash
export FREEYOLO_VARIANT=yolo_free_nano
export FREEYOLO_WEIGHT_PATH=/root/models/yolo_free_nano_ch.pth
export FREEYOLO_DT_STEM=freeyolo_nano_mot17_train
bash scripts/group_b/run_freeyolo_mot17_unified_eval.sh
```

Inference matches **CrowdHumanEvaluator** rescale (`bboxes * max(orig_h, orig_w)`). Defaults: `conf_thresh=0.005`, `nms_thresh=0.6`, `topk=1000`, `img_size=640`.

---

## Outstanding (next steps)

1. Run **`run_freeyolo_mot17_unified_eval.sh`** for **tiny** and **nano**; paste metrics into the MOT17 table + commit new `results/logs/*.log`.
2. Keep **FPS** from CrowdHuman microbenches for table readability, or re-bench on MOT17-sized loops and document in `notes`.

---

## Model names (three slots)

| Bench slug | Typical checkpoint |
|------------|-------------------|
| `yolov8n_crowdhuman` | Ultralytics `yolov8n_crowdhuman.pt` |
| `freeyolo_ch_tiny` | FreeYOLO `yolo_free_tiny` + `*_ch.pth` |
| `freeyolo_yolox_mot17` | FreeYOLO `yolo_free_nano` (Group B label; MOT17-oriented weights) |
