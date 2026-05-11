# Benchmark runs (auto-generated)
Sources: `results/runs/*.json`. Updated by `scripts/bench_runner.py` and callers of `save_result` / `merge_run_json`.

## [fcos_r50_crowdhuman] — 2026-05-11T00:00:00Z
- **File**: `results/runs/fcos_r50_crowdhuman.json`
- **Weights (path)**: `/workspace/repos/mmdetection/fcos_r50_caffe_fpn_gn-head_1x_coco-821213aa.pth`
- **Hardware**: TBD
- **Backend**: `mmdet`
- **AP50**: 0.3284
- **AP25**: 0.5425
- **AP75**: 0.1108
- **COCO AR (recall, IoU=0.50:0.95, maxDets=100)**: 0.2938
- **coco AR @IoU0.25**: 0.8024
- **coco AR @IoU0.50**: 0.5899
- **coco AR @IoU0.75**: 0.2567
- **Greedy micro @IoU0.25** (score≥thr in eval): P=0.8762 R=0.096 FDR=0.1238
- **Greedy micro @IoU0.50** (score≥thr in eval): P=0.7714 R=0.0845 FDR=0.2286
- **Greedy micro @IoU0.75** (score≥thr in eval): P=0.5579 R=0.0611 FDR=0.4421
- **Greedy legacy** (`precision` / `fdr` @ `--precision-iou-thr`): 0.7714 / 0.2286
- **FPS forward**: 
- **FPS predict**: 
- **Notes**: Quality: scripts/eval_coco_predictions.py — COCOeval bbox on --gt-json/--dt-json; recall=COCO AR maxDets=100 IoU=0.50:0.95; coco_ar_iou25/50/75 + coco_precision_r*_iou* + coco_fdr_* from official pycocotools tensors (PR recall grid=0.5); AP25 extra eval IoU=[0.25]; greedy P/R/FDR score>=0.5, IoUs [0.25, 0.5, 0.75]; legacy precision/fdr greedy @ IoU=0.5.; backend=mmdet model=FCOS split=CrowdHuman_val; config=/workspace/repos/mmdetection/configs/fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py; checkpoint=/workspace/repos/mmdetection/fcos_r50_caffe_fpn_gn-head_1x_coco-821213aa.pth; GT_ann=/workspace/data/crowdhuman_bridge/CrowdHuman/annotations/val.json; img_prefix=/workspace/data/crowdhuman_bridge/CrowdHuman/CrowdHuman_val/Images/; dt_json=/workspace/artifacts/fcos_crowdhuman_val_dt.json; mmdet_dump_outfile_prefix=/workspace/artifacts/mmdet_dump_prefix_fcos work_subdir=fcos_ch_val; --- versions ---; torch 2.5.1+cu121; mmdet 3.3.0; mmcv 2.2.0; pycocotools 2.0.11

## [ssd300_crowdhuman] — 2026-05-11T00:00:00Z
- **File**: `results/runs/ssd300_crowdhuman.json`
- **Weights (path)**: `/workspace/repos/mmdetection/ssd300_coco_20210803_015428-d231a06e.pth`
- **Hardware**: TBD
- **Backend**: `mmdet`
- **AP50**: 0.2874
- **AP25**: 0.5976
- **AP75**: 0.0473
- **COCO AR (recall, IoU=0.50:0.95, maxDets=100)**: 0.181
- **coco AR @IoU0.25**: 0.7801
- **coco AR @IoU0.50**: 0.4634
- **coco AR @IoU0.75**: 0.1176
- **Greedy micro @IoU0.25** (score≥thr in eval): P=0.893 R=0.218 FDR=0.107
- **Greedy micro @IoU0.50** (score≥thr in eval): P=0.7132 R=0.1741 FDR=0.2868
- **Greedy micro @IoU0.75** (score≥thr in eval): P=0.3042 R=0.0743 FDR=0.6958
- **Greedy legacy** (`precision` / `fdr` @ `--precision-iou-thr`): 0.7132 / 0.2868
- **FPS forward**: 
- **FPS predict**: 
- **Notes**: Quality: scripts/eval_coco_predictions.py — COCOeval bbox on --gt-json/--dt-json; recall=COCO AR maxDets=100 IoU=0.50:0.95; coco_ar_iou25/50/75 + coco_precision_r*_iou* + coco_fdr_* from official pycocotools tensors (PR recall grid=0.5); AP25 extra eval IoU=[0.25]; greedy P/R/FDR score>=0.5, IoUs [0.25, 0.5, 0.75]; legacy precision/fdr greedy @ IoU=0.5.; backend=mmdet model=SSD300 split=CrowdHuman_val; config=/workspace/repos/mmdetection/configs/ssd/ssd300_coco.py; checkpoint=/workspace/repos/mmdetection/ssd300_coco_20210803_015428-d231a06e.pth; GT_ann=/workspace/data/crowdhuman_bridge/CrowdHuman/annotations/val.json; img_prefix=/workspace/data/crowdhuman_bridge/CrowdHuman/CrowdHuman_val/Images/; dt_json=/workspace/artifacts/ssd_crowdhuman_val_dt.json; mmdet_dump_outfile_prefix=/workspace/artifacts/mmdet_dump_prefix_ssd work_subdir=ssd_ch_val; --- versions ---; torch 2.5.1+cu121; mmdet 3.3.0; mmcv 2.2.0; pycocotools 2.0.11

## [freeyolo_yolox_mot17] — 2026-05-09T14:47:53Z
- **File**: `results/runs/freeyolo_yolox_mot17_2026-05-09T144753Z.json`
- **Weights (path)**: `/workspace/models/yolo_free_nano_ch.pth`
- **Weights (id / hub)**: `https://github.com/yjh0410/FreeYOLO/releases/download/weight/yolo_free_nano_ch.pth`
- **Hardware**: NVIDIA GeForce RTX 4090
- **Backend**: `freeyolo`
- **AP50**: 0.6822
- **FPS forward**: 57.935
- **FPS predict**: 23.988
- **Notes**: recall: COCO Average Recall (AR) IoU=0.50:0.95, maxDets=100 from pycocotools summarize; see docs/benchmark_metrics_schema.md — do not mix with other val protocols.; FreeYOLO: forward=no_decode tensor output; predict=ValTransforms+full decode/NMS (docs/benchmark_metrics_schema.md).; eval_throughput_fps = num_validation_frames / wall_time(eval.py) (includes COCOeval on CPU); compare to YOLOv8 using fps_predict from the microbench.; FreeYOLO eval.py -d crowdhuman, variant=yolo_free_nano, bench model=freeyolo_yolox_mot17; CrowdHuman val split.

## [yolov8n_crowdhuman] — 2026-05-09T14:37:40Z
- **File**: `results/runs/yolov8n_crowdhuman_2026-05-09T143848Z.json`
- **Weights (path)**: `/workspace/models/yolov8n_crowdhuman.pt`
- **Weights (id / hub)**: `yakhyo/yolov8-crowdhuman`
- **Hardware**: NVIDIA GeForce RTX 4090
- **Backend**: `ultralytics_yolo`
- **AP50**: 0.5703
- **FPS forward**: 117.368
- **FPS predict**: 127.104
- **Notes**: mAP50 / mAP50-95 / recall: scripts/eval_coco_predictions.py (pycocotools COCOeval bbox) on DT from scripts/dump_ultralytics_coco_dt.py; CrowdHuman val GT aligned with FreeYOLO bridge val.json; conf=0.001, imgsz=640, NMS iou=0.7, max_det=300. recall = COCO AR maxDets=100 IoU=0.50:0.95 (see docs/benchmark_metrics_schema.md).; Log (canonical unified eval): results/logs/yolov8n_crowdhuman_unified_cocoeval_2026-05-11T203804Z.log; fps_forward / fps_predict / inference_time_ms: from bench_runner Ultralytics microbench (2026-05-09); not recomputed in unified quality run.; Prior model.val() mAP/recall/precision removed from metrics to avoid mixing protocols; compare README historical row only with that caveat.; MOT17 train FRCNN (separate split): eval_coco_predictions.py on dump_ultralytics_mot17.py DT; mAP50=0.647584, mAP50-95=0.334005, recall=0.427085 (AR maxDets=100). Not merged into metrics{} — CrowdHuman val remains canonical there; see docs/crowdhuman_val_full_metrics_table.md + results/logs/yolov8n_crowdhuman_mot17_unified_cocoeval_2026-05-11T141600Z.log.

## [freeyolo_ch_tiny] — 2026-05-09T14:33:28Z
- **File**: `results/runs/freeyolo_ch_tiny_2026-05-09T143328Z.json`
- **Weights (path)**: `/workspace/models/yolo_free_tiny_ch.pth`
- **Weights (id / hub)**: `https://github.com/yjh0410/FreeYOLO/releases/download/weight/yolo_free_tiny_ch.pth`
- **Hardware**: NVIDIA GeForce RTX 4090
- **Backend**: `freeyolo`
- **AP50**: 0.7166
- **FPS forward**: 93.256
- **FPS predict**: 34.588
- **Notes**: recall: COCO Average Recall (AR) IoU=0.50:0.95, maxDets=100 from pycocotools summarize; see docs/benchmark_metrics_schema.md — do not mix with other val protocols.; FreeYOLO: forward=no_decode tensor output; predict=ValTransforms+full decode/NMS (docs/benchmark_metrics_schema.md).; FreeYOLO eval.py -d crowdhuman, variant=yolo_free_tiny, bench model=freeyolo_ch_tiny; CrowdHuman val split.

---

## Summary table (all runs)

Legend: **AR_coco** = `metrics.recall` (COCO AR maxDets=100, IoU 0.50:0.95). **grP50/grR50/grF50** = greedy micro P/R/FDR at IoU 0.50 (`precision_iou50` / `recall_iou50` / `fdr_iou50`), score threshold as in eval notes. **cAR50** = `coco_ar_iou50` (COCO AR @ IoU 0.50). Extra columns are blank if the run JSON predates unified eval.

| Backend | Model | Date | AP25 | AP50 | AP75 | AP50-95 | AR_coco | grP50 | grR50 | grF50 | cAR50 | Infer (ms) | FPS fwd | FPS pred | MOTA | TRT |
|---------|--------|------|------|------|------|---------|---------|-------|-------|-------|-------|------------|---------|----------|------|-----|
| mmdet | fcos_r50_crowdhuman | 2026-05-11T00:00:00Z | 0.5425 | 0.3284 | 0.1108 | 0.144 | 0.2938 | 0.7714 | 0.0845 | 0.2286 | 0.5899 |  |  |  |  | no |
| mmdet | ssd300_crowdhuman | 2026-05-11T00:00:00Z | 0.5976 | 0.2874 | 0.0473 | 0.0965 | 0.181 | 0.7132 | 0.1741 | 0.2868 | 0.4634 |  |  |  |  | no |
| freeyolo | freeyolo_yolox_mot17 | 2026-05-09T14:47:53Z |  | 0.6822 |  | 0.3204 | 0.424 |  |  |  |  | 41.6876 | 57.935 | 23.988 |  | no |
| ultralytics_yolo | yolov8n_crowdhuman | 2026-05-09T14:37:40Z |  | 0.5703 |  | 0.2716 | 0.4023 |  |  |  |  | 6.9666 | 117.368 | 127.104 |  | no |
| freeyolo | freeyolo_ch_tiny | 2026-05-09T14:33:28Z |  | 0.7166 |  | 0.3564 | 0.456 |  |  |  |  | 28.9121 | 93.256 | 34.588 |  | no |
