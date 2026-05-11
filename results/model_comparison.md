# Model comparison (auto-generated)

Source: latest run per model from `results/runs/*.json`.

## Table

| Backend | Model | AP50 | AP25 | grR50 | AR_coco | FPS forward | FPS predict | MOTA | Date |
|---------|-------|------|------|-------|---------|-------------|-------------|------|------|
| freeyolo | freeyolo_ch_tiny | 0.7166 |  |  | 0.456 | 93.256 | 34.588 |  | 2026-05-09T14:33:28Z |
| freeyolo | freeyolo_yolox_mot17 | 0.6822 |  |  | 0.424 | 57.935 | 23.988 |  | 2026-05-09T14:47:53Z |
| ultralytics_yolo | yolov8n_crowdhuman | 0.5703 |  |  | 0.4023 | 117.368 | 127.104 |  | 2026-05-09T14:37:40Z |
| mmdet | fcos_r50_crowdhuman | 0.3284 | 0.5425 | 0.0845 | 0.2938 |  |  |  | 2026-05-11T00:00:00Z |
| mmdet | ssd300_crowdhuman | 0.2874 | 0.5976 | 0.1741 | 0.181 |  |  |  | 2026-05-11T00:00:00Z |

## AP50 vs FPS (ASCII)

FPS forward: 57.935 … 117.368  |  AP50: 0.5703 … 0.7166

```
|                                 A                      |
|                                                        |
|                                                        |
|B                                                       |
|                                                        |
|                                                        |
|                                                        |
|                                                        |
|                                                        |
|                                                        |
|                                                        |
|                                                        |
|                                                        |
|                                                       C|
```

Legend:
  A: freeyolo_ch_tiny — AP50 0.7166, FPS fwd 93.256
  B: freeyolo_yolox_mot17 — AP50 0.6822, FPS fwd 57.935
  C: yolov8n_crowdhuman — AP50 0.5703, FPS fwd 117.368
