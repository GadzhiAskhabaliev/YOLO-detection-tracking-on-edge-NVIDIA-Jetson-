# Benchmark JSON schema (`results/runs/*.json`)

This schema documents the normalized fields used for the 3 supported detectors:

- `yolov8n_crowdhuman`
- `freeyolo_ch_tiny`
- `freeyolo_yolox_mot17`

## Top-level fields

| Field | Required | Description |
| --- | --- | --- |
| `model` | yes | Detector id used in reports |
| `date` | yes | UTC ISO8601 save time |
| `weights` | yes | Checkpoint path used in run |
| `weights_hub` | no | Reproducible upstream source |
| `hardware` | yes | Hardware string from benchmark host |
| `backend` | yes | `ultralytics_yolo` or `freeyolo` |
| `batch_size`, `imgsz` | yes | Inference config |
| `metrics` | yes | Numeric metrics map |
| `tracking` | no | Optional tracking metrics container |
| `tensorrt` | no | Optional TensorRT info |
| `notes` | no | Protocol and caveats |

## Canonical `metrics` keys

| Key | Meaning |
| --- | --- |
| `AP25`, `AP50`, `AP75`, `AP50-95` | COCO-style AP metrics |
| `mAP50`, `mAP50-95` | Legacy aliases (read-only compatibility) |
| `recall` | COCO AR maxDets=100 (IoU 0.50:0.95) |
| `coco_ar_iou25/50/75` | COCO AR at single IoU |
| `precision_iou25/50/75` | Greedy micro precision |
| `recall_iou25/50/75` | Greedy micro recall |
| `fdr_iou25/50/75` | Greedy micro false discovery rate |
| `fps_forward` | Pure model forward throughput |
| `fps_predict` | End-to-end detector throughput |
| `inference_time_ms` | End-to-end latency |

## Unified quality evaluation

Use:

```bash
python3 scripts/eval_coco_predictions.py \
  --gt-json path/to/val.json \
  --dt-json path/to/predictions.json \
  --strict \
  --out-patch-json /tmp/metrics_patch.json
```

Then merge:

```bash
python3 scripts/bench_runner.py --merge-json results/runs/<run>.json \
  --patch-json /tmp/metrics_patch.json
```
