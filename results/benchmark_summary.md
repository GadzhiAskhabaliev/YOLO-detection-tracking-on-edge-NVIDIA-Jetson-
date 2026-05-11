# Benchmark runs (auto-generated)
Sources: `results/runs/*.json`. Updated by `scripts/bench_runner.py` and callers of `save_result` / `merge_run_json`.

## [freeyolo_yolox_mot17] — 2026-05-09T14:47:53Z
- **File**: `results/runs/freeyolo_yolox_mot17_2026-05-09T144753Z.json`
- **Weights (path)**: `/workspace/models/yolo_free_nano_ch.pth`
- **Weights (id / hub)**: `https://github.com/yjh0410/FreeYOLO/releases/download/weight/yolo_free_nano_ch.pth`
- **Hardware**: NVIDIA GeForce RTX 4090
- **Backend**: `freeyolo`
- **mAP50**: 0.6822
- **FPS forward**: 57.935
- **FPS predict**: 23.988
- **Notes**: recall: COCO Average Recall (AR) IoU=0.50:0.95, maxDets=100 from pycocotools summarize; see docs/benchmark_metrics_schema.md — do not mix with other val protocols.; FreeYOLO: forward=no_decode tensor output; predict=ValTransforms+full decode/NMS (docs/benchmark_metrics_schema.md).; eval_throughput_fps = num_validation_frames / wall_time(eval.py) (includes COCOeval on CPU); compare to YOLOv8 using fps_predict from the microbench.; FreeYOLO eval.py -d crowdhuman, variant=yolo_free_nano, bench model=freeyolo_yolox_mot17; CrowdHuman val split.

## [yolov8n_crowdhuman] — 2026-05-09T14:37:40Z
- **File**: `results/runs/yolov8n_crowdhuman_2026-05-09T143848Z.json`
- **Weights (path)**: `/workspace/models/yolov8n_crowdhuman.pt`
- **Weights (id / hub)**: `yakhyo/yolov8-crowdhuman`
- **Hardware**: NVIDIA GeForce RTX 4090
- **Backend**: `ultralytics_yolo`
- **mAP50**: 0.5703
- **FPS forward**: 117.368
- **FPS predict**: 127.104
- **Notes**: mAP50 / mAP50-95 / recall: scripts/eval_coco_predictions.py (pycocotools COCOeval bbox) on DT from scripts/dump_ultralytics_coco_dt.py; CrowdHuman val GT aligned with FreeYOLO bridge val.json; conf=0.001, imgsz=640, NMS iou=0.7, max_det=300. recall = COCO AR maxDets=100 IoU=0.50:0.95 (see docs/benchmark_metrics_schema.md).; Log: results/logs/yolov8n_crowdhuman_unified_cocoeval_2026-05-11T140530Z.log; fps_forward / fps_predict / inference_time_ms: from bench_runner Ultralytics microbench (2026-05-09); not recomputed in unified quality run.; Prior model.val() mAP/recall/precision removed from metrics to avoid mixing protocols; compare README historical row only with that caveat.; MOT17 train FRCNN (separate split): eval_coco_predictions.py on dump_ultralytics_mot17.py DT; mAP50=0.647584, mAP50-95=0.334005, recall=0.427085 (AR maxDets=100). Not merged into metrics{} — CrowdHuman val remains canonical there; see docs/benchmark_group_b_unified_two_domains.md + results/logs/yolov8n_crowdhuman_mot17_unified_cocoeval_2026-05-11T141600Z.log.

## [freeyolo_ch_tiny] — 2026-05-09T14:33:28Z
- **File**: `results/runs/freeyolo_ch_tiny_2026-05-09T143328Z.json`
- **Weights (path)**: `/workspace/models/yolo_free_tiny_ch.pth`
- **Weights (id / hub)**: `https://github.com/yjh0410/FreeYOLO/releases/download/weight/yolo_free_tiny_ch.pth`
- **Hardware**: NVIDIA GeForce RTX 4090
- **Backend**: `freeyolo`
- **mAP50**: 0.7166
- **FPS forward**: 93.256
- **FPS predict**: 34.588
- **Notes**: recall: COCO Average Recall (AR) IoU=0.50:0.95, maxDets=100 from pycocotools summarize; see docs/benchmark_metrics_schema.md — do not mix with other val protocols.; FreeYOLO: forward=no_decode tensor output; predict=ValTransforms+full decode/NMS (docs/benchmark_metrics_schema.md).; FreeYOLO eval.py -d crowdhuman, variant=yolo_free_tiny, bench model=freeyolo_ch_tiny; CrowdHuman val split.

---

## Summary table (all runs)

| Backend | Model | Date | mAP50 | mAP50-95 | Precision | Recall | Infer (ms) | FPS fwd | FPS pred | MOTA | TRT |
|---------|--------|------|-------|----------|-----------|--------|------------|---------|----------|------|-----|
| freeyolo | freeyolo_yolox_mot17 | 2026-05-09T14:47:53Z | 0.6822 | 0.3204 |  | 0.424 | 41.6876 | 57.935 | 23.988 |  | no |
| ultralytics_yolo | yolov8n_crowdhuman | 2026-05-09T14:37:40Z | 0.5703 | 0.2716 |  | 0.4023 | 6.9666 | 117.368 | 127.104 |  | no |
| freeyolo | freeyolo_ch_tiny | 2026-05-09T14:33:28Z | 0.7166 | 0.3564 |  | 0.456 | 28.9121 | 93.256 | 34.588 |  | no |
