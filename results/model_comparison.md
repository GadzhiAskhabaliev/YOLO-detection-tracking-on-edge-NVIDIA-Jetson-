# Model comparison (auto-generated)

Источник: последний прогон на модель из `results/runs/*.json`.

## Таблица

| Backend | Модель | mAP50 | FPS forward | FPS predict | MOTA | Дата |
|---------|--------|-------|-------------|-------------|------|------|
| ultralytics_yolo | yolov8n_crowdhuman | 0.7471 | 117.368 | 127.104 |  | 2026-05-09T14:37:40Z |
| freeyolo | freeyolo_ch_tiny | 0.7166 | 93.256 | 34.588 |  | 2026-05-09T14:33:28Z |
| freeyolo | freeyolo_yolox_mot17 | 0.6822 | 57.935 | 23.988 |  | 2026-05-09T14:47:53Z |

## mAP50 vs FPS (ASCII)

FPS forward: 57.935 … 117.368  |  mAP50: 0.6822 … 0.7471

```
|                                                       A|
|                                                        |
|                                                        |
|                                                        |
|                                                        |
|                                                        |
|                                 B                      |
|                                                        |
|                                                        |
|                                                        |
|                                                        |
|                                                        |
|                                                        |
|C                                                       |
```

Легенда:
  A: yolov8n_crowdhuman — mAP50 0.7471, FPS fwd 117.368
  B: freeyolo_ch_tiny — mAP50 0.7166, FPS fwd 93.256
  C: freeyolo_yolox_mot17 — mAP50 0.6822, FPS fwd 57.935
