# Model comparison (auto-generated)

Source: latest run per model from `results/runs/*.json`.

## Table

| Backend | Model | AP50 | AP25 | grR50 | AR_coco | FPS forward | FPS predict | MOTA | Date |
|---------|-------|------|------|-------|---------|-------------|-------------|------|------|
| crowddet | crowddet_rcnn_emd_refine_e30 | 0.8662 | 0.913 | 0.8533 | 0.5814 |  |  |  | 2026-05-12T00:00:00Z |
| freeyolo | freeyolo_ch_tiny | 0.7164 | 0.8608 | 0.7324 | 0.456 | 93.256 | 34.588 |  | 2026-05-09T14:33:28Z |
| freeyolo | freeyolo_yolox_mot17 | 0.6819 | 0.8414 | 0.6948 | 0.4239 | 72.277 | 31.659 |  | 2026-05-11T21:16:29Z |
| ultralytics_yolo | yolov8n_crowdhuman | 0.5703 | 0.8102 | 0.4334 | 0.4022 | 117.368 | 127.104 |  | 2026-05-09T14:37:40Z |
| mmdet | fcos_r50_crowdhuman | 0.3284 | 0.5425 | 0.0845 | 0.2938 |  |  |  | 2026-05-11T00:00:00Z |
| mmdet | ssd300_crowdhuman | 0.2874 | 0.5976 | 0.1741 | 0.181 |  |  |  | 2026-05-11T00:00:00Z |
| onnx_runtime | peoplenet_crowdhuman | 0.2076 | 0.3327 | 0.1872 | 0.1021 | 214.7649 | 147.8996 |  | 2026-05-12T15:09:19Z |

## AP50 vs FPS (ASCII)

FPS forward: 72.277 … 214.7649  |  AP50: 0.2076 … 0.7164

```
|        A                                               |
|B                                                       |
|                                                        |
|                                                        |
|                 C                                      |
|                                                        |
|                                                        |
|                                                        |
|                                                        |
|                                                        |
|                                                        |
|                                                        |
|                                                        |
|                                                       D|
```

Legend:
  A: freeyolo_ch_tiny — AP50 0.7164, FPS fwd 93.256
  B: freeyolo_yolox_mot17 — AP50 0.6819, FPS fwd 72.277
  C: yolov8n_crowdhuman — AP50 0.5703, FPS fwd 117.368
  D: peoplenet_crowdhuman — AP50 0.2076, FPS fwd 214.7649
