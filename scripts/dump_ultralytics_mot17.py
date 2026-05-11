#!/usr/bin/env python3
"""
Run Ultralytics YOLO on MOT17 train frames and write:

1. **COCO detection JSON** (list) for `scripts/eval_coco_predictions.py`
2. **MOT challenge detection files** per sequence: `<mot-det-root>/<SEQUENCE>/det.txt`
   Lines: `frame, -1, x, y, w, h, score, -1, -1, -1` (bb top-left, pixel coords).

Requires a COCO-style GT JSON produced by `scripts/mot17_gt_to_coco.py` (images must
include `mot_sequence` and `mot_frame`; `file_name` relative to MOT17 train root).

Example:

  python3 scripts/mot17_gt_to_coco.py \\
    --mot17-train-root /workspace/data/mot17/MOT17/train \\
    --det-subdir-suffix FRCNN \\
    --out-json /workspace/data/mot17/annotations/mot17_train_frcnn_gt.json

  python3 scripts/dump_ultralytics_mot17.py \\
    --gt-json /workspace/data/mot17/annotations/mot17_train_frcnn_gt.json \\
    --mot17-train-root /workspace/data/mot17/MOT17/train \\
    --weights /workspace/models/yolov8n_crowdhuman.pt \\
    --out-coco-dt-json /tmp/yolov8_mot17_train_dt.json \\
    --mot-det-root /workspace/data/mot17/detections/yolov8n_ch_mot17_train

  python3 scripts/eval_coco_predictions.py \\
    --gt-json /workspace/data/mot17/annotations/mot17_train_frcnn_gt.json \\
    --dt-json /tmp/yolov8_mot17_train_dt.json \\
    --strict
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def _bbox_xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float) -> list[float]:
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return [float(x1), float(y1), float(w), float(h)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gt-json", type=Path, required=True, help="MOT17 COCO GT from mot17_gt_to_coco.py")
    p.add_argument("--mot17-train-root", type=Path, required=True, help="…/MOT17/train")
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--out-coco-dt-json", type=Path, required=True)
    p.add_argument(
        "--mot-det-root",
        type=Path,
        required=True,
        help="Per-sequence folders created here with det.txt (MOT tracker input)",
    )
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--half", action="store_true")
    p.add_argument("--coco-category-offset", type=int, default=1)
    args = p.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit("Requires ultralytics\n" + str(e)) from e

    gt_path = args.gt_json.expanduser().resolve()
    train_root = args.mot17_train_root.expanduser().resolve()
    if not gt_path.is_file():
        raise SystemExit(f"GT not found: {gt_path}")
    if not train_root.is_dir():
        raise SystemExit(f"MOT17 train root not found: {train_root}")

    coco = json.loads(gt_path.read_text(encoding="utf-8"))
    images = coco.get("images")
    if not isinstance(images, list):
        raise SystemExit("GT JSON missing images[]")

    weights = args.weights.expanduser().resolve()
    if not weights.is_file():
        raise SystemExit(f"Weights not found: {weights}")

    model = YOLO(str(weights))
    dt: list[dict[str, Any]] = []
    mot_lines: dict[str, list[tuple[int, str]]] = defaultdict(list)

    missing = 0
    for im in images:
        if not isinstance(im, dict):
            continue
        fname = im.get("file_name")
        image_id = im.get("id")
        seq = im.get("mot_sequence")
        frame = im.get("mot_frame")
        if not isinstance(fname, str) or image_id is None:
            continue
        if not isinstance(seq, str) or not isinstance(frame, int):
            raise SystemExit(
                "GT images must include mot_sequence (str) and mot_frame (int). "
                "Regenerate GT with scripts/mot17_gt_to_coco.py"
            )

        ipath = train_root / fname
        if not ipath.is_file():
            missing += 1
            continue

        results = model.predict(
            source=str(ipath),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            verbose=False,
            device=args.device,
            half=bool(args.half),
        )

        ih, iw = im.get("height"), im.get("width")
        if not isinstance(ih, int) or not isinstance(iw, int):
            ih, iw = results[0].orig_shape

        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            scores = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = (float(xyxy[i, j]) for j in range(4))
                x1 = max(0.0, min(x1, float(iw)))
                x2 = max(0.0, min(x2, float(iw)))
                y1 = max(0.0, min(y1, float(ih)))
                y2 = max(0.0, min(y2, float(ih)))
                bbox = _bbox_xyxy_to_xywh(x1, y1, x2, y2)
                sc = float(scores[i])
                cid = int(clss[i]) + int(args.coco_category_offset)
                dt.append(
                    {
                        "image_id": int(image_id),
                        "category_id": cid,
                        "bbox": bbox,
                        "score": sc,
                    }
                )
                mot_lines[seq].append(
                    (
                        frame,
                        f"{frame}, -1, {bbox[0]:.2f}, {bbox[1]:.2f}, {bbox[2]:.2f}, {bbox[3]:.2f}, "
                        f"{sc:.6f}, -1, -1, -1",
                    )
                )

    if missing:
        print(f"WARNING: {missing} GT images missing on disk under {train_root}", file=sys.stderr)

    args.out_coco_dt_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_coco_dt_json.write_text(json.dumps(dt), encoding="utf-8")

    args.mot_det_root.mkdir(parents=True, exist_ok=True)
    for seq, rows in mot_lines.items():
        seq_dir = args.mot_det_root / seq
        seq_dir.mkdir(parents=True, exist_ok=True)
        rows.sort(key=lambda t: t[0])
        det_path = seq_dir / "det.txt"
        det_path.write_text("\n".join(line for _, line in rows) + ("\n" if rows else ""), encoding="utf-8")

    print(
        json.dumps(
            {
                "coco_dt": str(args.out_coco_dt_json),
                "detections": len(dt),
                "mot_sequences": len(mot_lines),
                "mot_det_root": str(args.mot_det_root),
                "missing_images": missing,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
