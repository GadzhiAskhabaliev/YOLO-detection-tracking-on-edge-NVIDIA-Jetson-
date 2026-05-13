# Unified CrowdHuman detection table (3 YOLO models)

Scope in this repository is intentionally limited to:

1. `yolov8n_crowdhuman`
2. `freeyolo_ch_tiny`
3. `freeyolo_yolox_mot17` (FreeYOLO nano)

Metrics source: `results/runs/*.json` (`metrics` field).

| Backend | Model | Run JSON | AP25 | AP50 | AP75 | AP50-95 | AR_coco | FPS forward | FPS predict | Log |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ultralytics_yolo | yolov8n_crowdhuman | [`yolov8n_crowdhuman_2026-05-09T143848Z.json`](../results/runs/yolov8n_crowdhuman_2026-05-09T143848Z.json) | 0.810214 | 0.570265 | 0.2312 | 0.271562 | 0.40217 | 117.368 | 127.104 | [`yolov8n_crowdhuman_unified_cocoeval_2026-05-11T203804Z.log`](../results/logs/yolov8n_crowdhuman_unified_cocoeval_2026-05-11T203804Z.log) |
| freeyolo | freeyolo_ch_tiny | [`freeyolo_ch_tiny_2026-05-09T143328Z.json`](../results/runs/freeyolo_ch_tiny_2026-05-09T143328Z.json) | 0.860757 | 0.716403 | 0.308386 | 0.356308 | 0.456033 | 93.256 | 34.588 | [`freeyolo_yolo_free_tiny_unified_cocoeval_2026-05-11T211739Z.log`](../results/logs/freeyolo_yolo_free_tiny_unified_cocoeval_2026-05-11T211739Z.log) |
| freeyolo | freeyolo_yolox_mot17 | [`freeyolo_yolox_mot17_2026-05-11T211629Z.json`](../results/runs/freeyolo_yolox_mot17_2026-05-11T211629Z.json) | 0.841419 | 0.681859 | 0.2595 | 0.320213 | 0.423903 | 72.277 | 31.659 | [`freeyolo_yolo_free_nano_unified_cocoeval_2026-05-11T212522Z.log`](../results/logs/freeyolo_yolo_free_nano_unified_cocoeval_2026-05-11T212522Z.log) |

## Notes

- All rows evaluate on CrowdHuman val with a unified COCO protocol.
- `AR_coco` is COCO AR (IoU 0.50:0.95, maxDets=100).
- Tracking benchmarks are stored separately in `results/tracking/`.
