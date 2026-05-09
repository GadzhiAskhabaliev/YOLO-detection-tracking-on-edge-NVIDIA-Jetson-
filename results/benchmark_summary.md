# Benchmark runs (auto-generated)
Файлы-источники: `results/runs/*.json`. Обновляется `scripts/bench_runner.py` и скрипты, которые вызывают `save_result` / `merge_run_json`.

## [freeyolo_yolox_mot17] — 2026-05-09T14:47:53Z
- **Файл**: `results/runs/freeyolo_yolox_mot17_2026-05-09T144753Z.json`
- **Weights (path)**: `/workspace/models/yolo_free_nano_ch.pth`
- **Weights (id / hub)**: `https://github.com/yjh0410/FreeYOLO/releases/download/weight/yolo_free_nano_ch.pth`
- **Hardware**: NVIDIA GeForce RTX 4090
- **Backend**: `freeyolo`
- **mAP50**: 0.6822
- **FPS forward**: 57.935
- **FPS predict**: 23.988
- **Примечания**: recall: COCO Average Recall (AR) IoU=0.50:0.95, maxDets=100 из summarize pycocotools; см. определение recall в docs/BENCHMARK_METRICS_SCHEMA.md — не смешивать с другими протоколами val.; FreeYOLO: forward=no_decode tensor output; predict=ValTransforms+full decode/NMS (см. docs/BENCHMARK_METRICS_SCHEMA.md).; eval_throughput_fps = число кадров валидации / wall-clock всего eval.py (включает COCOeval на CPU); для сравнения с YOLOv8 используйте fps_predict из микробенча.; FreeYOLO eval.py -d crowdhuman, variant=yolo_free_nano, bench model=freeyolo_yolox_mot17. Веса — сплит CrowdHuman; имя freeyolo_yolox_mot17 — исторический слот группы B для nano.

## [yolov8n_crowdhuman] — 2026-05-09T14:37:40Z
- **Файл**: `results/runs/yolov8n_crowdhuman_2026-05-09T143848Z.json`
- **Weights (path)**: `/workspace/models/yolov8n_crowdhuman.pt`
- **Weights (id / hub)**: `yakhyo/yolov8-crowdhuman`
- **Hardware**: NVIDIA GeForce RTX 4090
- **Backend**: `ultralytics_yolo`
- **mAP50**: 0.7471
- **FPS forward**: 117.368
- **FPS predict**: 127.104

## [freeyolo_ch_tiny] — 2026-05-09T14:33:28Z
- **Файл**: `results/runs/freeyolo_ch_tiny_2026-05-09T143328Z.json`
- **Weights (path)**: `/workspace/models/yolo_free_tiny_ch.pth`
- **Weights (id / hub)**: `https://github.com/yjh0410/FreeYOLO/releases/download/weight/yolo_free_tiny_ch.pth`
- **Hardware**: NVIDIA GeForce RTX 4090
- **Backend**: `freeyolo`
- **mAP50**: 0.7166
- **FPS forward**: 93.256
- **FPS predict**: 34.588
- **Примечания**: recall: COCO Average Recall (AR) IoU=0.50:0.95, maxDets=100 из summarize pycocotools; см. определение recall в docs/BENCHMARK_METRICS_SCHEMA.md — не смешивать с другими протоколами val.; FreeYOLO: forward=no_decode tensor output; predict=ValTransforms+full decode/NMS (см. docs/BENCHMARK_METRICS_SCHEMA.md).; FreeYOLO eval.py -d crowdhuman, variant=yolo_free_tiny, bench model=freeyolo_ch_tiny. Веса — сплит CrowdHuman; имя freeyolo_yolox_mot17 — исторический слот группы B для nano.

---

## Сводная таблица (все прогоны)

| Backend | Модель | Дата | mAP50 | mAP50-95 | Precision | Recall | Infer (ms) | FPS fwd | FPS pred | MOTA | TRT |
|---------|--------|------|-------|----------|-----------|--------|------------|---------|----------|------|-----|
| freeyolo | freeyolo_yolox_mot17 | 2026-05-09T14:47:53Z | 0.6822 | 0.3204 |  | 0.424 | 41.6876 | 57.935 | 23.988 |  | no |
| ultralytics_yolo | yolov8n_crowdhuman | 2026-05-09T14:37:40Z | 0.7471 | 0.4642 | 0.8113 | 0.6501 | 6.9666 | 117.368 | 127.104 |  | no |
| freeyolo | freeyolo_ch_tiny | 2026-05-09T14:33:28Z | 0.7166 | 0.3564 |  | 0.456 | 28.9121 | 93.256 | 34.588 |  | no |
