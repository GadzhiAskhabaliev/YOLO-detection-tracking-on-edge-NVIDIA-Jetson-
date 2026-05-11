#!/usr/bin/env python3
"""
Dump Ultralytics YOLO predictions as a COCO-style detection list for
`scripts/eval_coco_predictions.py`.

Use the **same** CrowdHuman val GT as FreeYOLO (`freeyolo_prepare_crowdhuman.py` →
`.../CrowdHuman/annotations/val.json`): `images[].id` are sequential integers and
`file_name` matches `{crowdhuman_id}.jpg` under `--images-dir`.

Example (after preparing GT + images):

  python3 scripts/dump_ultralytics_coco_dt.py \\
    --gt-json /workspace/group_b/freeyolo_crowdhuman_bridge/CrowdHuman/annotations/val.json \\
    --images-dir /workspace/data/crowdhuman/Images \\
    --weights /workspace/models/yolov8n_crowdhuman.pt \\
    --out-json /tmp/yolov8n_ch_val_dt.json

  python3 scripts/eval_coco_predictions.py \\
    --gt-json .../val.json \\
    --dt-json /tmp/yolov8n_ch_val_dt.json \\
    --strict \\
    --out-patch-json /tmp/yolov8_unified_metrics.json

FreeYOLO CrowdHuman: for the same COCO DT file fed to `eval_coco_predictions.py`,
use `scripts/group_b/dump_freeyolo_crowdhuman_coco_dt.py` (FreeYOLO venv). Upstream
`eval.py` also runs `COCOeval` internally; numbers can differ slightly from a dump
if decode/NMS paths differ.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _bbox_xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float) -> list[float]:
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return [float(x1), float(y1), float(w), float(h)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gt-json", type=Path, required=True, help="COCO instances GT (CrowdHuman val)")
    p.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Directory containing GT file_name entries (e.g. .../crowdhuman/Images)",
    )
    p.add_argument("--weights", type=Path, required=True, help="Ultralytics .pt checkpoint")
    p.add_argument("--out-json", type=Path, required=True, help="Output COCO DT JSON (list)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.001, help="Min confidence (default 0.001; FreeYOLO eval uses 0.005)")
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU (Ultralytics predict)")
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--half", action="store_true", help="FP16 inference on CUDA")
    p.add_argument(
        "--coco-category-offset",
        type=int,
        default=1,
        help="DT category_id = model_class_id + offset (single-class CrowdHuman → offset 1 maps cls 0 → id 1)",
    )
    args = p.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit("Requires ultralytics: pip install ultralytics\n" + str(e)) from e

    gt_path = args.gt_json.expanduser().resolve()
    img_root = args.images_dir.expanduser().resolve()
    if not gt_path.is_file():
        raise SystemExit(f"GT not found: {gt_path}")
    if not img_root.is_dir():
        raise SystemExit(f"images-dir not found: {img_root}")

    coco_gt = json.loads(gt_path.read_text(encoding="utf-8"))
    images = coco_gt.get("images")
    if not isinstance(images, list):
        raise SystemExit("GT JSON missing images[]")

    weights = args.weights.expanduser().resolve()
    if not weights.is_file():
        raise SystemExit(f"Weights not found: {weights}")

    model = YOLO(str(weights))
    device = args.device
    half = bool(args.half)

    dt: list[dict[str, Any]] = []
    missing_files: list[str] = []

    for im in images:
        if not isinstance(im, dict):
            continue
        fname = im.get("file_name")
        image_id = im.get("id")
        if not isinstance(fname, str) or image_id is None:
            continue
        ipath = img_root / fname
        if not ipath.is_file():
            missing_files.append(fname)
            continue

        results = model.predict(
            source=str(ipath),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            verbose=False,
            device=device,
            half=half,
        )
        for r in results:
            ih, iw = r.orig_shape
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
                cid = int(clss[i]) + int(args.coco_category_offset)
                dt.append(
                    {
                        "image_id": int(image_id),
                        "category_id": cid,
                        "bbox": bbox,
                        "score": float(scores[i]),
                    }
                )

    if missing_files:
        print(
            f"WARNING: {len(missing_files)} GT images missing under {img_root} "
            f"(first: {missing_files[:5]}{'...' if len(missing_files) > 5 else ''})",
            file=sys.stderr,
        )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(dt), encoding="utf-8")
    print(
        json.dumps(
            {
                "out_json": str(args.out_json),
                "detections": len(dt),
                "gt_images": len(images),
                "missing_files": len(missing_files),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
