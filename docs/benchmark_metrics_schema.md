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
| `backend` | recommended | Short tag: `mmdet`, `fairmot`, `freeyolo`, `ultralytics_yolo`, `tao_peoplenet`, `onnx_runtime`, … |
| `batch_size`, `imgsz` | yes | As in the run; fair FPS comparisons usually use `batch_size=1`. |
| `group`, `detector_id`, `detector_label` | no | Report slots (e.g. Group B manifest). |
| `metrics` | yes | Numeric metrics (see below). |
| `tracking` | no | MOTA, IDF1, HOTA, etc. |
| `tensorrt` | no | `engine_exists`, optional `fps_fp16`. |
| `notes` | no | Dataset, split, thresholds, **definitions** of precision/recall/FPS if non-standard. |

## `metrics` keys (fixed names)

| Key | Definition |
|-----|------------|
| `mAP50` | AP at IoU=0.50 on a **fixed** validation split (often CrowdHuman val in this repo). Other data → document in `notes`. |
| `mAP50-95` | AP at IoU=0.50:0.95 on the same split. |
| `precision` | Precision under the **protocol you fixed** (e.g. MMDet val report). Do not mix with AP@0.5 without stating so. |
| `recall` | Recall under the same protocol, **or** COCO Average Recall (AR) — then state IoU range and maxDets in `notes`. |
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

Use one **GT** file (`val.json`, COCO instances) for all models. **`scripts/eval_coco_predictions.py`** computes **`mAP50`**, **`mAP50-95`**, and **`recall`** (= COCO AR, maxDets=100) via **pycocotools**. **`--strict`** fails if DT references unknown `image_id`.

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
    "mAP50": null,
    "mAP50-95": null,
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
