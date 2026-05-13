# Tracker comparison (MOT17)

Score formula: `norm(HOTA)+norm(IDF1)+norm(MOTA)+norm(FPS)`.

| Rank | Tracker | conf | iou | HOTA | IDF1 | MOTA | FPS | Score |
|------|---------|------|-----|------|------|------|-----|-------|
| 1 | yolov8_bytetrack | 0.35 | 0.7 | 0.314034 | 0.382897 | 0.269576 | 46.8053 | 3.7640 |
| 2 | yolov8_hybridsort | 0.2 | 0.6 | 0.306657 | 0.350699 | 0.287014 | 14.7048 | 2.5056 |
| 3 | yolov8_deepocsort | 0.35 | 0.7 | 0.306109 | 0.340972 | 0.227598 | 19.2282 | 1.7254 |
| 4 | yolov8_strongsort | 0.25 | 0.7 | 0.290052 | 0.313026 | 0.213121 | 16.9263 | 0.8429 |
| 5 | yolov8_botsort | 0.2 | 0.6 | 0.268003 | 0.299909 | 0.220117 | 17.0857 | 0.3747 |

Winner: `yolov8_bytetrack` with conf=0.35, iou=0.7.
