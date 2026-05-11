# Model comparison (auto-generated)

Source: latest run per model from `results/runs/*.json`.

## Table

| Backend | Model | AP50 | FPS forward | FPS predict | MOTA | Date |
|---------|-------|-------|-------------|-------------|------|------|
| freeyolo | freeyolo_ch_tiny | 0.7166 | 93.256 | 34.588 |  | 2026-05-09T14:33:28Z |
| freeyolo | freeyolo_yolox_mot17 | 0.6822 | 57.935 | 23.988 |  | 2026-05-09T14:47:53Z |
| ultralytics_yolo | yolov8n_crowdhuman | 0.5703 | 117.368 | 127.104 |  | 2026-05-09T14:37:40Z |

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
