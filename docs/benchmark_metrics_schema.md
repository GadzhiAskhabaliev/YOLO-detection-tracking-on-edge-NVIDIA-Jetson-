# Benchmark JSON schema (`results/runs/*.json`)

Every detector row in **`results/benchmark_summary.md`** and the README table must use the **same field names** below. Set **`backend`** to the actual framework tag (for example `ultralytics_yolo`, `freeyolo`, `mmdet`). **`notes`** must record dataset, split, thresholds, and **how** precision, recall, and FPS were obtained whenever they differ from this canon.

## Top-level fields

| Field | Required | Description |
|-------|----------|-------------|
| `model` | yes | Logical row name (e.g. `yolov8n_crowdhuman`, `fcos_r50_caffe_fpn_gn_ch`). |
| `date` | yes | UTC ISO8601 at save time. |
| `weights` | yes | Path to checkpoint on the machine that ran the benchmark. |
| `weights_hub` | no | Registry / zoo id / release URL for reproduction. |
| `hardware` | yes | GPU or CPU string (e.g. first line of `nvidia-smi`). |
| `backend` | recommended | Short tag: `mmdet`, `fairmot`, `freeyolo`, `ultralytics_yolo`, `crowddet`, `tao_peoplenet`, `onnx_runtime`, … |
| `batch_size`, `imgsz` | yes | As in the run; fair FPS comparisons usually use `batch_size=1`. |
| `group`, `detector_id`, `detector_label` | no | Report slots (e.g. Group B manifest). |
| `metrics` | yes | Numeric metrics (see below). |
| `tracking` | no | MOTA, IDF1, HOTA, etc. |
| `tensorrt` | no | `engine_exists`, optional `fps_fp16`. |
| `notes` | no | Dataset, split, thresholds, **definitions** of precision/recall/FPS if non-standard. |

## `metrics` keys (fixed names)

| Key | Definition |
|-----|------------|
| `AP50` | AP at IoU=0.50 on a **fixed** validation split (often CrowdHuman val). For a **single class**, this is the same scalar COCO reports as “mAP50” (mean over categories has one term). |
| `AP50-95` | AP at IoU=0.50:0.95 on the same split. |
| `AP25`, `AP75` | Optional: from unified `eval_coco_predictions.py` — COCOeval AP at IoU=0.25 (extra eval pass) and IoU=0.75 (`stats[2]`). |
| `mAP50`, `mAP50-95` | **Legacy keys** in older `results/runs/*.json`; readers fall back to these if `AP50` / `AP50-95` are absent. |
| `coco_ar_iou25`, `coco_ar_iou50`, `coco_ar_iou75` | Optional: **official COCO** Average Recall (AR), maxDets=100, area=all, at that **single** IoU — from `pycocotools` `eval['recall']` (same matching as AP). |
| `coco_precision_r50_iou25` (and `…iou50`, `…iou75`) | Optional: mean over categories of **official COCO** precision on the PR curve at **recall grid 0.50** (override with `--coco-pr-recall`). Not fixed-score greedy matching. |
| `coco_fdr_r50_iou25` (…) | Optional: 1 − corresponding `coco_precision_r50_iou*`. |
| `precision` | Precision under the **protocol you fixed** (e.g. MMDet val report). Do not mix with AP@0.5 without stating so. |
| `precision_iou25`, `recall_iou25`, `fdr_iou25` (and `…50`, `…75`) | Optional **greedy micro** metrics from `eval_coco_predictions.py`: score ≥ `--precision-score-thr`, one-to-one match per image vs non-crowd GT at IoU ≥ threshold; `fdr_*` = 1 − `precision_*`. |
| `fdr` | Optional: 1 − `precision` when `precision` is the greedy scalar at `--precision-iou-thr` (default 0.5). |
| `recall` | In unified eval JSON this is **COCO AR** maxDets=100, IoU **0.50:0.95** (not single-IoU AR). Single-IoU AR → `coco_ar_iou*`. |
| `inference_time_ms` | Mean **end-to-end** milliseconds per frame: preprocess + network + decode/NMS, aligned with `fps_predict`. |
| `fps_forward` | Throughput of the **narrow network path**: tensor already preprocessed, **without** heavy CPU postprocess where separable (document rule in `notes`). |
| `fps_predict` | Full detector path: preprocess + inference + decode/NMS, batch=1, representative input. |
| `forward_time_ms_mean` | Optional `1000 / fps_forward`. |
| `inference_time_ms_predict` | Optional `1000 / fps_predict`; should match `inference_time_ms` within rounding. |

Extra keys (`eval_throughput_fps`, `eval_wall_seconds`, …) may store **script wall-clock**; never treat them as canonical detector FPS without explanation in `notes`.

## Model families

### MMDetection (FCOS, SSD, YOLOX, …)

1. Fixed config + checkpoint; `weights_hub` holds the zoo id.
2. CrowdHuman quality may need a dedicated config or COCO conversion — state in `notes`.
3. Prefer explicit warmup + timed loops with `torch.cuda.synchronize()` for **`fps_predict`** / **`inference_time_ms`**; define **`fps_forward`** in `notes` if split from predict.

### ONNX / MMDeploy / TensorRT

Document MMDeploy version, opset, precision; use `backend` tags such as `onnx_runtime` or `tensorrt`.

### FairMOT and joint detector–tracker

Detector metrics: same `metrics`. Tracking: **`tracking`**. If **`fps_predict`** includes tracking, say so explicitly — do not compare to bare detectors without a caveat.

### Unified evaluation from predictions

To avoid framework-specific val quirks, dump boxes as a **COCO-style detection list**:

```json
[
  {"image_id": 1, "category_id": 1, "bbox": [10.5, 20.0, 80.0, 160.0], "score": 0.91}
]
```

(`bbox` is xywh in pixels.)

Use one **GT** file (`val.json`, COCO instances) for all models. **`scripts/eval_coco_predictions.py`** computes **`AP25`** (second COCOeval with `iouThrs=[0.25]` only), **`AP50`**, **`AP75`**, **`AP50-95`**, and **`recall`** (= COCO AR, maxDets=100, IoU 0.50:0.95) via **pycocotools**. For **supervisor-style** numbers straight from the **same** COCO tensors as AP: **`coco_ar_iou25/50/75`** (AR at that IoU), **`coco_precision_r50_iou25/50/75`** and **`coco_fdr_r50_iou*`** (precision / FDR on the official PR grid at recall **0.50**, configurable via **`--coco-pr-recall`**). Separately it adds **greedy micro** **`precision_iou*`** / **`recall_iou*`** / **`fdr_iou*`** for IoUs in **`--greedy-iou-thrs`** (default `0.25,0.5,0.75`): fixed **score ≥ `--precision-score-thr`**, **TP/(TP+FP)**, **TP/N_gt**, **FDR = 1 − precision**. Legacy **`precision`** / **`fdr`** use **`--precision-iou-thr`** (default 0.5). **`--strict`** fails if DT references unknown `image_id`. **`--quiet-summarize`** still fills `stats` (table output is suppressed).

```bash
python3 scripts/eval_coco_predictions.py \
  --gt-json path/to/annotations/val.json \
  --dt-json path/to/predictions.json \
  --strict \
  --out-patch-json /tmp/metrics_patch.json

python3 scripts/bench_runner.py --merge-json results/runs/your_model.json \
  --patch-json /tmp/metrics_patch.json
```

FPS is **not** computed by that script; measure separately and merge again.

### External stacks (CrowdDet, Pedestron, PeopleNet, MMDet zoo, …)

This repository does **not** vendor those codebases. Produce predictions (or full metrics + FPS) **in their environment**, then normalize via **`eval_coco_predictions.py`** (quality) and **`bench_runner.py --merge-json`** (table + summary).

### Minimal merge patch examples

Patch fragment for metrics placeholder:

```json
{
  "backend": "mmdet",
  "metrics": {
    "AP50": null,
    "AP50-95": null,
    "fps_forward": null,
    "fps_predict": null
  },
  "notes": ["Fill after CrowdHuman val; document fps_forward vs fps_predict in notes."]
}
```

Group B slot metadata:

```json
{
  "group": "B",
  "detector_id": 8,
  "detector_label": "PeopleNet"
}
```

Regenerating **`results/benchmark_summary.md`** and the README table is handled inside `save_result` / `merge_run_json`.
