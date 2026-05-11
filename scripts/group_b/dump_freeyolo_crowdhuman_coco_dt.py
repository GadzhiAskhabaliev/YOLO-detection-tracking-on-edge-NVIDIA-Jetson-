#!/usr/bin/env python3
"""
Dump FreeYOLO predictions on CrowdHuman val as a COCO detection list for
`scripts/eval_coco_predictions.py` (same bridge `val.json` as Ultralytics).

Use the FreeYOLO venv interpreter. Boxes are rescaled like upstream
CrowdHumanEvaluator / MOT17 dumper: normalized coords * max(orig_h, orig_w).

Example:

  FREEYOLO_HOME=/path/to/FreeYOLO /path/to/venv/bin/python \\
    scripts/group_b/dump_freeyolo_crowdhuman_coco_dt.py \\
    --freeyolo-home "$FREEYOLO_HOME" \\
    --variant yolo_free_tiny \\
    --weights /path/to/yolo_free_tiny_ch.pth \\
    --gt-json "$BRIDGE/CrowdHuman/annotations/val.json" \\
    --images-dir /path/to/crowdhuman/Images \\
    --out-coco-dt-json /tmp/freeyolo_tiny_ch_val_dt.json
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import torch


def _bbox_xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float) -> list[float]:
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return [float(x1), float(y1), float(w), float(h)]


def _empty_target() -> dict[str, Any]:
    return {
        "boxes": np.zeros((0, 4), dtype=np.float32),
        "labels": np.zeros((0,), dtype=np.int64),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--freeyolo-home", type=Path, required=True)
    p.add_argument("--variant", type=str, required=True, help="e.g. yolo_free_tiny, yolo_free_nano")
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--gt-json", type=Path, required=True, help="Bridge CrowdHuman val.json")
    p.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Directory with {id}.jpg (same as CrowdHuman Images/)",
    )
    p.add_argument("--out-coco-dt-json", type=Path, required=True)
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--device", default="", help="cuda | cpu; default: cuda if available")
    p.add_argument("--conf-thresh", type=float, default=0.005, help="Match FreeYOLO eval defaults")
    p.add_argument("--nms-thresh", type=float, default=0.6)
    p.add_argument("--topk", type=int, default=1000)
    p.add_argument(
        "--coco-category-id",
        type=int,
        default=1,
        help="category_id in DT (single-class pedestrian → 1)",
    )
    args = p.parse_args()

    fy = args.freeyolo_home.expanduser().resolve()
    wpath = args.weights.expanduser().resolve()
    gt_path = args.gt_json.expanduser().resolve()
    img_root = args.images_dir.expanduser().resolve()

    if not fy.is_dir():
        raise SystemExit(f"FreeYOLO home not found: {fy}")
    if not wpath.is_file():
        raise SystemExit(f"Weights not found: {wpath}")
    if not gt_path.is_file():
        raise SystemExit(f"GT not found: {gt_path}")
    if not img_root.is_dir():
        raise SystemExit(f"images-dir not found: {img_root}")

    device_s = (args.device or "").strip().lower()
    if device_s == "cpu":
        device = torch.device("cpu")
    elif device_s == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sys.path.insert(0, str(fy))
    os.chdir(str(fy))

    from config import build_config
    from dataset.transforms import ValTransforms
    from models import build_model
    from utils.misc import load_weight

    ns = SimpleNamespace(
        dataset="crowdhuman",
        version=args.variant,
        img_size=args.img_size,
        mosaic=None,
        mixup=None,
        cuda=device.type == "cuda",
        weight=str(wpath),
        conf_thresh=args.conf_thresh,
        nms_thresh=args.nms_thresh,
        topk=args.topk,
        no_decode=False,
        root=str(fy),
    )

    cfg = copy.deepcopy(build_config(ns))

    model = build_model(args=ns, cfg=cfg, device=device, num_classes=1, trainable=False)
    model = load_weight(model=model, path_to_ckpt=str(wpath))
    model.to(device).eval()
    model.no_decode = False

    transform = ValTransforms(img_size=args.img_size)

    coco = json.loads(gt_path.read_text(encoding="utf-8"))
    images = coco.get("images")
    if not isinstance(images, list):
        raise SystemExit("GT JSON missing images[]")

    dt: list[dict[str, Any]] = []
    missing = 0

    with torch.no_grad():
        for im in images:
            if not isinstance(im, dict):
                continue
            fname = im.get("file_name")
            image_id = im.get("id")
            if not isinstance(fname, str) or image_id is None:
                continue

            ipath = img_root / fname
            if not ipath.is_file():
                missing += 1
                continue

            img_bgr = cv2.imread(str(ipath))
            if img_bgr is None:
                missing += 1
                continue

            orig_h, orig_w = img_bgr.shape[:2]
            x_tensor, _ = transform(img_bgr, _empty_target())
            x = x_tensor.unsqueeze(0).to(device)

            outputs = model(x)
            bboxes, scores, _cls_inds = outputs
            if isinstance(bboxes, torch.Tensor):
                bboxes = bboxes.detach().float().cpu().numpy()
            if isinstance(scores, torch.Tensor):
                scores = scores.detach().float().cpu().numpy()
            bboxes = np.asarray(bboxes, dtype=np.float64) * float(max(orig_h, orig_w))
            scores = np.asarray(scores, dtype=np.float64)

            ih, iw = orig_h, orig_w
            for i in range(len(bboxes)):
                x1, y1, x2, y2 = (float(bboxes[i, j]) for j in range(4))
                x1 = max(0.0, min(x1, float(iw)))
                x2 = max(0.0, min(x2, float(iw)))
                y1 = max(0.0, min(y1, float(ih)))
                y2 = max(0.0, min(y2, float(ih)))
                bbox = _bbox_xyxy_to_xywh(x1, y1, x2, y2)
                sc = float(scores[i])
                cid = args.coco_category_id
                dt.append(
                    {
                        "image_id": int(image_id),
                        "category_id": int(cid),
                        "bbox": bbox,
                        "score": sc,
                    }
                )

    if missing:
        print(
            f"WARNING: {missing} GT images missing or unreadable under {img_root}",
            file=sys.stderr,
        )

    args.out_coco_dt_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_coco_dt_json.write_text(json.dumps(dt), encoding="utf-8")

    print(
        json.dumps(
            {
                "coco_dt": str(args.out_coco_dt_json),
                "detections": len(dt),
                "variant": args.variant,
                "weights": str(wpath),
                "missing_images": missing,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
