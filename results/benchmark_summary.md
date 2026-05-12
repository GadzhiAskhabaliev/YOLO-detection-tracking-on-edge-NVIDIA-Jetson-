# Benchmark runs (auto-generated)
Sources: `results/runs/*.json`. Updated by `scripts/bench_runner.py` and callers of `save_result` / `merge_run_json`.

## [peoplenet_crowdhuman] — 2026-05-12T15:09:19Z
- **File**: `results/runs/peoplenet_crowdhuman_2026-05-12T150919Z.json`
- **Weights (path)**: `/root/workspace/models/peoplenet/resnet34_peoplenet_int8.onnx`
- **Weights (id / hub)**: `ngc:nvidia/tao/peoplenet:pruned_quantized_decrypted_v2.3.4`
- **Hardware**: NVIDIA GeForce RTX 3090
- **Backend**: `onnx_runtime`
- **AP50**: 0.2076
- **AP25**: 0.3327
- **AP75**: 0.0363
- **COCO AR (recall, IoU=0.50:0.95, maxDets=100)**: 0.1021
- **coco AR @IoU0.25**: 0.3316
- **coco AR @IoU0.50**: 0.2401
- **coco AR @IoU0.75**: 0.0747
- **Greedy micro @IoU0.25** (score≥thr in eval): P=0.9646 R=0.2363 FDR=0.0354
- **Greedy micro @IoU0.50** (score≥thr in eval): P=0.7642 R=0.1872 FDR=0.2358
- **Greedy micro @IoU0.75** (score≥thr in eval): P=0.2656 R=0.0651 FDR=0.7344
- **Greedy legacy** (`precision` / `fdr` @ `--precision-iou-thr`): 0.7642 / 0.2358
- **FPS forward**: 214.7649
- **FPS predict**: 147.8996
- **Notes**: Quality from eval_coco_predictions.py on CrowdHuman val + peoplenet_dt_v2.json (--strict).; fps_forward=tensor-only; fps_predict=preprocess+run+light decode scan (ORT CUDA EP).; Quality: scripts/eval_coco_predictions.py — COCOeval bbox on --gt-json/--dt-json; recall=COCO AR maxDets=100 IoU=0.50:0.95; coco_ar_iou25/50/75 + coco_precision_r*_iou* + coco_fdr_* from official pycocotools tensors (PR recall grid=0.5); AP25 extra eval IoU=[0.25]; greedy P/R/FDR score>=0.5, IoUs [0.25, 0.5, 0.75]; legacy precision/fdr greedy @ IoU=0.5.; FPS measured on ONNXRuntime CUDAExecutionProvider; warmup=100 iters=500.; fps_forward: sess.run on preprocessed tensor only.; fps_predict: preprocess + sess.run + lightweight decode scan.; PeopleNet ONNX (NGC pruned_quantized_decrypted_v2.3.4), CrowdHuman val.; Quality: scripts/eval_coco_predictions.py --strict on peoplenet_dt_v2.json.; FPS: ORT CUDA EP; fps_forward=tensor-only, fps_predict=preprocess+run+light decode scan.

## [peoplenet_crowdhuman] — 2026-05-12T15:08:56Z
- **File**: `results/runs/peoplenet_crowdhuman_2026-05-12T150856Z.json`
- **Weights (path)**: `/root/workspace/models/peoplenet/resnet34_peoplenet_int8.onnx`
- **Weights (id / hub)**: `ngc:nvidia/tao/peoplenet:pruned_quantized_decrypted_v2.3.4`
- **Hardware**: NVIDIA GeForce RTX 3090
- **Backend**: `onnx_runtime`
- **AP50**: 0.2076
- **AP25**: 0.3327
- **AP75**: 0.0363
- **COCO AR (recall, IoU=0.50:0.95, maxDets=100)**: 0.1021
- **coco AR @IoU0.25**: 0.3316
- **coco AR @IoU0.50**: 0.2401
- **coco AR @IoU0.75**: 0.0747
- **Greedy micro @IoU0.25** (score≥thr in eval): P=0.9646 R=0.2363 FDR=0.0354
- **Greedy micro @IoU0.50** (score≥thr in eval): P=0.7642 R=0.1872 FDR=0.2358
- **Greedy micro @IoU0.75** (score≥thr in eval): P=0.2656 R=0.0651 FDR=0.7344
- **Greedy legacy** (`precision` / `fdr` @ `--precision-iou-thr`): 0.7642 / 0.2358
- **FPS forward**: 214.7649
- **FPS predict**: 147.8996
- **Notes**: Quality from eval_coco_predictions.py on CrowdHuman val + peoplenet_dt_v2.json (--strict).; fps_forward=tensor-only; fps_predict=preprocess+run+light decode scan (ORT CUDA EP).; Quality: scripts/eval_coco_predictions.py — COCOeval bbox on --gt-json/--dt-json; recall=COCO AR maxDets=100 IoU=0.50:0.95; coco_ar_iou25/50/75 + coco_precision_r*_iou* + coco_fdr_* from official pycocotools tensors (PR recall grid=0.5); AP25 extra eval IoU=[0.25]; greedy P/R/FDR score>=0.5, IoUs [0.25, 0.5, 0.75]; legacy precision/fdr greedy @ IoU=0.5.; FPS measured on ONNXRuntime CUDAExecutionProvider; warmup=100 iters=500.; fps_forward: sess.run on preprocessed tensor only.; fps_predict: preprocess + sess.run + lightweight decode scan.

## [freeyolo_yolox_mot17] — 2026-05-11T21:16:29Z
- **File**: `results/runs/freeyolo_yolox_mot17_2026-05-11T211629Z.json`
- **Weights (path)**: `/workspace/models/yolo_free_nano_ch.pth`
- **Weights (id / hub)**: `https://github.com/yjh0410/FreeYOLO/releases/download/weight/yolo_free_nano_ch.pth`
- **Hardware**: NVIDIA GeForce RTX 5090
- **Backend**: `freeyolo`
- **AP50**: 0.6819
- **AP25**: 0.8414
- **AP75**: 0.2595
- **COCO AR (recall, IoU=0.50:0.95, maxDets=100)**: 0.4239
- **coco AR @IoU0.25**: 0.8935
- **coco AR @IoU0.50**: 0.7807
- **coco AR @IoU0.75**: 0.3998
- **Greedy micro @IoU0.25** (score≥thr in eval): P=0.7509 R=0.7893 FDR=0.2491
- **Greedy micro @IoU0.50** (score≥thr in eval): P=0.661 R=0.6948 FDR=0.339
- **Greedy micro @IoU0.75** (score≥thr in eval): P=0.3567 R=0.375 FDR=0.6433
- **Greedy legacy** (`precision` / `fdr` @ `--precision-iou-thr`): 0.661 / 0.339
- **FPS forward**: 72.277
- **FPS predict**: 31.659
- **Notes**: Unified CrowdHuman val: dump + scripts/eval_coco_predictions.py (strict); full tee: results/logs/freeyolo_yolo_free_nano_unified_cocoeval_2026-05-11T212522Z.log; Quality: scripts/eval_coco_predictions.py — COCOeval bbox on --gt-json/--dt-json; recall=COCO AR maxDets=100 IoU=0.50:0.95; coco_ar_iou25/50/75 + coco_precision_r*_iou* + coco_fdr_* from official pycocotools tensors (PR recall grid=0.5); AP25 extra eval IoU=[0.25]; greedy P/R/FDR score>=0.5, IoUs [0.25, 0.5, 0.75]; legacy precision/fdr greedy @ IoU=0.5.

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

## [yolov8n_crowdhuman] — 2026-05-09T14:37:40Z
- **File**: `results/runs/yolov8n_crowdhuman_2026-05-09T143848Z.json`
- **Weights (path)**: `/workspace/models/yolov8n_crowdhuman.pt`
- **Weights (id / hub)**: `yakhyo/yolov8-crowdhuman`
- **Hardware**: NVIDIA GeForce RTX 4090
- **Backend**: `ultralytics_yolo`
- **AP50**: 0.5703
- **AP25**: 0.8102
- **AP75**: 0.2312
- **COCO AR (recall, IoU=0.50:0.95, maxDets=100)**: 0.4022
- **coco AR @IoU0.25**: 0.8829
- **coco AR @IoU0.50**: 0.7226
- **coco AR @IoU0.75**: 0.3848
- **Greedy micro @IoU0.25** (score≥thr in eval): P=0.9384 R=0.5129 FDR=0.0616
- **Greedy micro @IoU0.50** (score≥thr in eval): P=0.793 R=0.4334 FDR=0.207
- **Greedy micro @IoU0.75** (score≥thr in eval): P=0.5066 R=0.2769 FDR=0.4934
- **Greedy legacy** (`precision` / `fdr` @ `--precision-iou-thr`): 0.793 / 0.207
- **FPS forward**: 117.368
- **FPS predict**: 127.104
- **Notes**: Unified CrowdHuman val: dump + scripts/eval_coco_predictions.py (strict); full tee: results/logs/yolov8n_crowdhuman_unified_cocoeval_2026-05-11T203804Z.log; Quality: scripts/eval_coco_predictions.py — COCOeval bbox on --gt-json/--dt-json; recall=COCO AR maxDets=100 IoU=0.50:0.95; coco_ar_iou25/50/75 + coco_precision_r*_iou* + coco_fdr_* from official pycocotools tensors (PR recall grid=0.5); AP25 extra eval IoU=[0.25]; greedy P/R/FDR score>=0.5, IoUs [0.25, 0.5, 0.75]; legacy precision/fdr greedy @ IoU=0.5.

## [freeyolo_ch_tiny] — 2026-05-09T14:33:28Z
- **File**: `results/runs/freeyolo_ch_tiny_2026-05-09T143328Z.json`
- **Weights (path)**: `/workspace/models/yolo_free_tiny_ch.pth`
- **Weights (id / hub)**: `https://github.com/yjh0410/FreeYOLO/releases/download/weight/yolo_free_tiny_ch.pth`
- **Hardware**: NVIDIA GeForce RTX 4090
- **Backend**: `freeyolo`
- **AP50**: 0.7164
- **AP25**: 0.8608
- **AP75**: 0.3084
- **COCO AR (recall, IoU=0.50:0.95, maxDets=100)**: 0.456
- **coco AR @IoU0.25**: 0.9033
- **coco AR @IoU0.50**: 0.8029
- **coco AR @IoU0.75**: 0.444
- **Greedy micro @IoU0.25** (score≥thr in eval): P=0.7657 R=0.8135 FDR=0.2343
- **Greedy micro @IoU0.50** (score≥thr in eval): P=0.6894 R=0.7324 FDR=0.3106
- **Greedy micro @IoU0.75** (score≥thr in eval): P=0.3993 R=0.4242 FDR=0.6007
- **Greedy legacy** (`precision` / `fdr` @ `--precision-iou-thr`): 0.6894 / 0.3106
- **FPS forward**: 93.256
- **FPS predict**: 34.588
- **Notes**: Unified CrowdHuman val: dump + scripts/eval_coco_predictions.py (strict); full tee: results/logs/freeyolo_yolo_free_tiny_unified_cocoeval_2026-05-11T211739Z.log; Quality: scripts/eval_coco_predictions.py — COCOeval bbox on --gt-json/--dt-json; recall=COCO AR maxDets=100 IoU=0.50:0.95; coco_ar_iou25/50/75 + coco_precision_r*_iou* + coco_fdr_* from official pycocotools tensors (PR recall grid=0.5); AP25 extra eval IoU=[0.25]; greedy P/R/FDR score>=0.5, IoUs [0.25, 0.5, 0.75]; legacy precision/fdr greedy @ IoU=0.5.

---

## Summary table (all runs)

Legend: **AR_coco** = `metrics.recall` (COCO AR maxDets=100, IoU 0.50:0.95). **grP50/grR50/grF50** = greedy micro P/R/FDR at IoU 0.50 (`precision_iou50` / `recall_iou50` / `fdr_iou50`), score threshold as in eval notes. **cAR50** = `coco_ar_iou50` (COCO AR @ IoU 0.50). Extra columns are blank if the run JSON predates unified eval.

| Backend | Model | Date | AP25 | AP50 | AP75 | AP50-95 | AR_coco | grP50 | grR50 | grF50 | cAR50 | Infer (ms) | FPS fwd | FPS pred | MOTA | TRT |
|---------|--------|------|------|------|------|---------|---------|-------|-------|-------|-------|------------|---------|----------|------|-----|
| onnx_runtime | peoplenet_crowdhuman | 2026-05-12T15:09:19Z | 0.3327 | 0.2076 | 0.0363 | 0.0717 | 0.1021 | 0.7642 | 0.1872 | 0.2358 | 0.2401 | 6.7613 | 214.7649 | 147.8996 |  | no |
| onnx_runtime | peoplenet_crowdhuman | 2026-05-12T15:08:56Z | 0.3327 | 0.2076 | 0.0363 | 0.0717 | 0.1021 | 0.7642 | 0.1872 | 0.2358 | 0.2401 | 6.7613 | 214.7649 | 147.8996 |  | no |
| freeyolo | freeyolo_yolox_mot17 | 2026-05-11T21:16:29Z | 0.8414 | 0.6819 | 0.2595 | 0.3202 | 0.4239 | 0.661 | 0.6948 | 0.339 | 0.7807 | 31.5871 | 72.277 | 31.659 |  | no |
| mmdet | fcos_r50_crowdhuman | 2026-05-11T00:00:00Z | 0.5425 | 0.3284 | 0.1108 | 0.144 | 0.2938 | 0.7714 | 0.0845 | 0.2286 | 0.5899 |  |  |  |  | no |
| mmdet | ssd300_crowdhuman | 2026-05-11T00:00:00Z | 0.5976 | 0.2874 | 0.0473 | 0.0965 | 0.181 | 0.7132 | 0.1741 | 0.2868 | 0.4634 |  |  |  |  | no |
| ultralytics_yolo | yolov8n_crowdhuman | 2026-05-09T14:37:40Z | 0.8102 | 0.5703 | 0.2312 | 0.2716 | 0.4022 | 0.793 | 0.4334 | 0.207 | 0.7226 | 6.9666 | 117.368 | 127.104 |  | no |
| freeyolo | freeyolo_ch_tiny | 2026-05-09T14:33:28Z | 0.8608 | 0.7164 | 0.3084 | 0.3563 | 0.456 | 0.6894 | 0.7324 | 0.3106 | 0.8029 | 28.9121 | 93.256 | 34.588 |  | no |
