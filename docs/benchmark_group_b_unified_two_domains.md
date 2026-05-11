# Group B — unified detection metrics (CrowdHuman val + MOT17 train)

**This repo:** scripts under `scripts/`, GT builders (`mot17_gt_to_coco.py`), dumpers
(`dump_ultralytics_*.py`, `dump_freeyolo_mot17.py`), and **`eval_coco_predictions.py`**
(pycocotools `COCOeval`, bbox IoU). **Elsewhere:** Ultralytics (pip), FreeYOLO
([yjh0410/FreeYOLO](https://github.com/yjh0410/FreeYOLO) clone + venv on the GPU host),
CrowdDet / MMDet workflows in other repos (see `docs/group_b_remote_mmdet_bridge.md`).

**FPS** in the tables below always comes from `results/runs/*.json` microbenches unless you re-bench and merge.

Detailed CrowdHuman-only doc (repro, merge): [`benchmark_unified_cocoeval.md`](benchmark_unified_cocoeval.md).

## CrowdHuman val

| Backend | Model | AP50 | AP50-95 | Recall (AR maxDets=100) | FPS forward | FPS predict | Source |
|---------|-------|-------|----------|---------------------------|-------------|-------------|--------|
| ultralytics_yolo | yolov8n_crowdhuman | 0.570286 | 0.271584 | 0.402259 | 117.368 | 127.104 | [`yolov8n_crowdhuman_2026-05-09T143848Z.json`](../results/runs/yolov8n_crowdhuman_2026-05-09T143848Z.json), log [`yolov8n_crowdhuman_unified_cocoeval_2026-05-11T140530Z.log`](../results/logs/yolov8n_crowdhuman_unified_cocoeval_2026-05-11T140530Z.log) |
| freeyolo | freeyolo_ch_tiny | 0.716557 | 0.356380 | 0.456 | 93.256 | 34.588 | [`freeyolo_ch_tiny_2026-05-09T143328Z.json`](../results/runs/freeyolo_ch_tiny_2026-05-09T143328Z.json), tee [`freeyolo_yolo_free_tiny_20260509T141227Z.log`](../results/logs/freeyolo_yolo_free_tiny_20260509T141227Z.log) |
| freeyolo | freeyolo_yolox_mot17 | 0.682212 | 0.320365 | 0.424 | 57.935 | 23.988 | [`freeyolo_yolox_mot17_2026-05-09T144753Z.json`](../results/runs/freeyolo_yolox_mot17_2026-05-09T144753Z.json), tee [`freeyolo_yolo_free_nano_20260509T143905Z.log`](../results/logs/freeyolo_yolo_free_nano_20260509T143905Z.log) |

**Protocol:** YOLOv8 — `dump_ultralytics_coco_dt.py` → `eval_coco_predictions.py --strict` on FreeYOLO-bridge `val.json`. FreeYOLO — upstream `eval.py` on CrowdHuman val (same `COCOeval`; numbers in run JSON).

**Six committed tee logs (`results/logs/`):** CrowdHuman val — [`yolov8n_crowdhuman_unified_cocoeval_2026-05-11T140530Z.log`](../results/logs/yolov8n_crowdhuman_unified_cocoeval_2026-05-11T140530Z.log), [`freeyolo_yolo_free_tiny_20260509T141227Z.log`](../results/logs/freeyolo_yolo_free_tiny_20260509T141227Z.log), [`freeyolo_yolo_free_nano_20260509T143905Z.log`](../results/logs/freeyolo_yolo_free_nano_20260509T143905Z.log). MOT17 train — [`yolov8n_crowdhuman_mot17_unified_cocoeval_2026-05-11T141600Z.log`](../results/logs/yolov8n_crowdhuman_mot17_unified_cocoeval_2026-05-11T141600Z.log), [`freeyolo_tiny_mot17_train_unified_20260511T144123Z.log`](../results/logs/freeyolo_tiny_mot17_train_unified_20260511T144123Z.log), [`freeyolo_nano_mot17_train_unified_20260511T144951Z.log`](../results/logs/freeyolo_nano_mot17_train_unified_20260511T144951Z.log).

---

## MOT17 train (detector tag **FRCNN** only)

GT: [`scripts/mot17_gt_to_coco.py`](../scripts/mot17_gt_to_coco.py) → one COCO instances JSON (e.g. `mot17_train_frcnn_gt.json`), **5316** images / **112297** GT boxes for the standard FRCNN train split (seven sequences).

| Backend | Model | AP50 | AP50-95 | Recall (AR maxDets=100) | FPS forward | FPS predict | Source |
|---------|-------|-------|----------|---------------------------|-------------|-------------|--------|
| ultralytics_yolo | yolov8n_crowdhuman | 0.647584 | 0.334005 | 0.427085 | 117.368 | 127.104 | [`yolov8n_crowdhuman_mot17_unified_cocoeval_2026-05-11T141600Z.log`](../results/logs/yolov8n_crowdhuman_mot17_unified_cocoeval_2026-05-11T141600Z.log) |
| freeyolo | freeyolo_ch_tiny | 0.649053 | 0.321077 | 0.406507 | 93.256 | 34.588 | [`run_freeyolo_mot17_unified_eval.sh`](../scripts/group_b/run_freeyolo_mot17_unified_eval.sh); log [`freeyolo_tiny_mot17_train_unified_20260511T144123Z.log`](../results/logs/freeyolo_tiny_mot17_train_unified_20260511T144123Z.log) |
| freeyolo | freeyolo_yolox_mot17 | 0.63572 | 0.316118 | 0.409424 | 57.935 | 23.988 | [`run_freeyolo_mot17_unified_eval.sh`](../scripts/group_b/run_freeyolo_mot17_unified_eval.sh); log [`freeyolo_nano_mot17_train_unified_20260511T144951Z.log`](../results/logs/freeyolo_nano_mot17_train_unified_20260511T144951Z.log) |

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

**FreeYOLO MOT17** — FreeYOLO venv on the host; clone + patches + weights as in [`run_freeyolo_crowdhuman.sh`](../scripts/group_b/run_freeyolo_crowdhuman.sh):

```bash
export MOT17_ROOT=/root/data/mot17
export FREEYOLO_HOME=/root/group_b/FreeYOLO
# venv must exist (create via run_freeyolo_crowdhuman.sh). If path differs:
export FREEYOLO_VENV=/root/group_b/venv_freeyolo
# or: export FREEYOLO_PYTHON=/root/group_b/venv_freeyolo/bin/python
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

**Where logs go:** `run_freeyolo_mot17_unified_eval.sh` tees **dump + eval** into **`<bench-repo>/results/logs/${FREEYOLO_DT_STEM}_unified_<UTC>.log`** on the host that runs the script. COCO DT and per-run metrics JSON are under **`/tmp/`** on that same host. The script prints **`--- Full log: <path> ---`** when finished; use that path with `scp` if you need to copy a new run into your checkout.

**Logs in this repo (full tee):** FreeYOLO tiny — [`freeyolo_tiny_mot17_train_unified_20260511T144123Z.log`](../results/logs/freeyolo_tiny_mot17_train_unified_20260511T144123Z.log); FreeYOLO nano — [`freeyolo_nano_mot17_train_unified_20260511T144951Z.log`](../results/logs/freeyolo_nano_mot17_train_unified_20260511T144951Z.log).

Inference matches **CrowdHumanEvaluator** rescale (`bboxes * max(orig_h, orig_w)`). Defaults: `conf_thresh=0.005`, `nms_thresh=0.6`, `topk=1000`, `img_size=640`.

---

## Model names (three slots)

| Bench slug | Typical checkpoint |
|------------|-------------------|
| `yolov8n_crowdhuman` | Ultralytics `yolov8n_crowdhuman.pt` |
| `freeyolo_ch_tiny` | FreeYOLO `yolo_free_tiny` + `yolo_free_tiny_ch.pth` (~49.5 MiB; **not** the ~16 MiB nano file) |
| `freeyolo_yolox_mot17` | FreeYOLO `yolo_free_nano` (Group B label; MOT17-oriented weights) |
