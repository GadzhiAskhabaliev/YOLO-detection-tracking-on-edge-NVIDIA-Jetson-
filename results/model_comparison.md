# YOLO detector comparison

Source: latest 3 supported detector runs from `results/runs/*.json`.

## Table

| Backend | Model | AP50 | AP25 | AR_coco | FPS forward | FPS predict | Date |
| --- | --- | --- | --- | --- | --- | --- | --- |
| freeyolo | freeyolo_ch_tiny | 0.716403 | 0.860757 | 0.456033 | 93.256 | 34.588 | 2026-05-09T14:33:28Z |
| freeyolo | freeyolo_yolox_mot17 | 0.681859 | 0.841419 | 0.423903 | 72.277 | 31.659 | 2026-05-11T21:16:29Z |
| ultralytics_yolo | yolov8n_crowdhuman | 0.570265 | 0.810214 | 0.40217 | 117.368 | 127.104 | 2026-05-09T14:37:40Z |

## Quick take

- Best quality by AP50: `freeyolo_ch_tiny`
- Best realtime throughput: `yolov8n_crowdhuman`
- Balanced option (quality/speed): `freeyolo_yolox_mot17`
