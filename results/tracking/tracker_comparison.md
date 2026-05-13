# Tracker comparison (all models, MOT17)

Score formula: `norm(HOTA)+norm(IDF1)+norm(MOTA)+norm(FPS)`.

## yolov8n_crowdhuman

| Rank | Tracker | conf | iou | HOTA | IDF1 | MOTA | FPS | Score |
|------|---------|------|-----|------|------|------|-----|-------|
| 1 | yolov8_bytetrack | 0.35 | 0.7 | 0.314034 | 0.382897 | 0.269576 | 46.8053 | 3.6930 |
| 2 | yolov8_deepocsort | 0.35 | 0.7 | 0.306109 | 0.340972 | 0.227598 | 19.2282 | 2.6961 |
| 3 | yolov8_hybridsort | 0.35 | 0.7 | 0.291017 | 0.307507 | 0.264356 | 15.4780 | 2.2762 |
| 4 | yolov8_strongsort | 0.25 | 0.7 | 0.290052 | 0.313026 | 0.213121 | 16.9263 | 2.2662 |
| 5 | yolov8_botsort | 0.25 | 0.7 | 0.267456 | 0.298341 | 0.218557 | 17.0925 | 1.9478 |

Winner (yolov8n_crowdhuman): `yolov8_bytetrack` with conf=0.35, iou=0.7.

## freeyolo_tiny

| Rank | Tracker | conf | iou | HOTA | IDF1 | MOTA | FPS | Score |
|------|---------|------|-----|------|------|------|-----|-------|
| 1 | bytetrack | - | - | 0.282934 | 0.352998 | 0.161940 | 53.5395 | 3.1540 |
| 2 | botsort | - | - | 0.316050 | 0.393904 | 0.210269 | 16.3749 | 3.1056 |
| 3 | strongsort | - | - | 0.313470 | 0.400919 | 0.283408 | 7.2321 | 3.0510 |
| 4 | deepocsort | - | - | 0.300165 | 0.380862 | 0.275066 | 12.3156 | 2.8623 |
| 5 | hybridsort | - | - | 0.213333 | 0.275491 | -0.345299 | 3.5518 | 0.2038 |

Winner (freeyolo_tiny): `bytetrack` with conf=-, iou=-.

## freeyolo_nano

| Rank | Tracker | conf | iou | HOTA | IDF1 | MOTA | FPS | Score |
|------|---------|------|-----|------|------|------|-----|-------|
| 1 | bytetrack | - | - | 0.265459 | 0.327274 | 0.159841 | 52.0511 | 2.7600 |
| 2 | strongsort | - | - | 0.290702 | 0.352572 | 0.237339 | 7.6084 | 2.4139 |
| 3 | botsort | - | - | 0.280045 | 0.320172 | 0.189871 | 16.2141 | 2.1756 |
| 4 | deepocsort | - | - | 0.270947 | 0.318032 | 0.211775 | 12.7996 | 2.0332 |
| 5 | hybridsort | - | - | 0.224144 | 0.266396 | -0.442226 | 3.4188 | 0.1052 |

Winner (freeyolo_nano): `bytetrack` with conf=-, iou=-.

## Overall Summary

| Rank | Detector | Best tracker | conf | iou | HOTA | IDF1 | MOTA | FPS | Score |
|------|----------|--------------|------|-----|------|------|------|-----|-------|
| 1 | yolov8n_crowdhuman | yolov8_bytetrack | 0.35 | 0.7 | 0.314034 | 0.382897 | 0.269576 | 46.8053 | 3.6930 |
| 2 | freeyolo_tiny | bytetrack | - | - | 0.282934 | 0.352998 | 0.161940 | 53.5395 | 3.1540 |
| 3 | freeyolo_nano | bytetrack | - | - | 0.265459 | 0.327274 | 0.159841 | 52.0511 | 2.7600 |

Best detector baseline: `yolov8n_crowdhuman + yolov8_bytetrack`.

Overall winner: `yolov8n_crowdhuman + yolov8_bytetrack` with conf=0.35, iou=0.7.
