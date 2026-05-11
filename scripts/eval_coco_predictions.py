#!/usr/bin/env python3
"""
Evaluate bbox detection: COCO-instances **GT** JSON + list of **DT** dicts.

GT is any single split (e.g. CrowdHuman val) with `images[].id`. DT rows must use
those `image_id` values. Each DT dict: `image_id`, `category_id`, `bbox` [x,y,w,h]
pixels xywh, `score`.

Outputs: mAP50, mAP50-95, recall (COCO AR maxDets=100, IoU 0.50:0.95), precision
(greedy TP/(TP+FP) at fixed score / IoU thresholds; see --precision-score-thr). No FPS.

  python3 scripts/eval_coco_predictions.py --gt-json .../val.json --dt-json .../dt.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def _xywh_to_xyxy(b: list[float] | tuple[float, ...]) -> tuple[float, float, float, float]:
    x, y, w, h = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
    return x, y, x + w, y + h


def _iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _greedy_precision(
    coco_gt: Any,
    raw_dt: list[dict[str, Any]],
    *,
    score_thr: float,
    iou_thr: float,
) -> float:
    """
    Global TP/(TP+FP): per image, sort DT by score desc, greedy match to GT (same
    category_id) at IoU >= iou_thr; each GT matches at most one DT.
    """
    by_img: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for d in raw_dt:
        if float(d["score"]) < score_thr:
            continue
        by_img[int(d["image_id"])].append(d)
    for img_id in by_img:
        by_img[img_id].sort(key=lambda x: float(x["score"]), reverse=True)

    tp = 0
    fp = 0
    for img_id, dets in by_img.items():
        ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
        anns = coco_gt.loadAnns(ann_ids)
        gts: list[dict[str, Any]] = []
        for a in anns:
            if a.get("iscrowd", 0) == 1:
                continue
            gts.append(
                {
                    "bbox": a["bbox"],
                    "cat": int(a["category_id"]),
                    "xyxy": _xywh_to_xyxy(a["bbox"]),
                }
            )
        matched = [False] * len(gts)

        for det in dets:
            dxy = _xywh_to_xyxy(det["bbox"])
            dcat = int(det["category_id"])
            best_j = -1
            best_iou = 0.0
            for j, g in enumerate(gts):
                if matched[j] or g["cat"] != dcat:
                    continue
                iou = _iou_xyxy(dxy, g["xyxy"])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_j >= 0 and best_iou >= iou_thr:
                matched[best_j] = True
                tp += 1
            else:
                fp += 1

    denom = tp + fp
    return float(tp / denom) if denom > 0 else 0.0


def _validate_entries(raw_dt: list[Any]) -> None:
    need = ("image_id", "category_id", "bbox", "score")
    for i, d in enumerate(raw_dt):
        if not isinstance(d, dict):
            raise SystemExit(f"dt[{i}] must be a dict object")
        for k in need:
            if k not in d:
                raise SystemExit(f"dt[{i}]: missing key {k!r}")
        b = d["bbox"]
        if not isinstance(b, (list, tuple)) or len(b) != 4:
            raise SystemExit(f"dt[{i}]: bbox must be [x,y,w,h] with four numbers")


def _check_image_ids(coco_gt: Any, raw_dt: list[dict[str, Any]], *, strict: bool) -> None:
    gt_ids = set(coco_gt.getImgIds())
    dt_ids = {int(d["image_id"]) for d in raw_dt}
    unknown = sorted(dt_ids - gt_ids)
    if not unknown:
        return
    msg = (
        f"Predictions contain image_id not in GT ({len(unknown)} ids), "
        f"first: {unknown[:10]}{'...' if len(unknown) > 10 else ''}. "
        "GT and DT refer to different splits or annotation files."
    )
    if strict:
        raise SystemExit(msg)
    print(f"WARNING: {msg}", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gt-json", type=Path, required=True, help="COCO GT (instances)")
    p.add_argument(
        "--dt-json",
        type=Path,
        required=True,
        help="COCO-style detection list or mmcv/json dump of a list",
    )
    p.add_argument(
        "--out-metrics-json",
        type=Path,
        default=None,
        help="Write metrics object (mAP50, mAP50-95, recall=AR, precision)",
    )
    p.add_argument(
        "--precision-score-thr",
        type=float,
        default=0.5,
        help="Score threshold for greedy precision (default 0.5)",
    )
    p.add_argument(
        "--precision-iou-thr",
        type=float,
        default=0.5,
        help="IoU threshold for greedy match to GT (default 0.5)",
    )
    p.add_argument(
        "--out-patch-json",
        type=Path,
        default=None,
        help="Patch JSON for bench_runner --merge-json: {metrics, notes}",
    )
    p.add_argument(
        "--quiet-summarize",
        action="store_true",
        help="Do not print default pycocotools summarize() block",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error if DT references unknown image_id vs GT",
    )
    args = p.parse_args()

    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as e:
        raise SystemExit(
            "Requires pycocotools: pip install pycocotools\n" + str(e)
        ) from e

    gt_path = args.gt_json.expanduser().resolve()
    dt_path = args.dt_json.expanduser().resolve()
    if not gt_path.is_file():
        raise SystemExit(f"GT not found: {gt_path}")
    if not dt_path.is_file():
        raise SystemExit(f"DT not found: {dt_path}")

    raw_dt = json.loads(dt_path.read_text(encoding="utf-8"))
    if isinstance(raw_dt, dict) and "annotations" in raw_dt:
        raw_dt = raw_dt["annotations"]
    if not isinstance(raw_dt, list):
        raise SystemExit("dt-json must be a list of detections or dict with 'annotations'")

    _validate_entries(raw_dt)

    coco_gt = COCO(str(gt_path))
    _check_image_ids(coco_gt, raw_dt, strict=args.strict)

    n_gt_img = len(coco_gt.getImgIds())
    dt_img_hit = len({int(d["image_id"]) for d in raw_dt})
    print(
        f"[eval_coco_predictions] GT images: {n_gt_img}, "
        f"images with ≥1 detection in DT: {dt_img_hit}, detections: {len(raw_dt)}",
        file=sys.stderr,
    )

    coco_dt = coco_gt.loadRes(raw_dt)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    if not args.quiet_summarize:
        coco_eval.summarize()

    stats = coco_eval.stats
    ap5095 = float(stats[0])
    ap50 = float(stats[1])
    ar100 = float(stats[8])

    prec = _greedy_precision(
        coco_gt,
        raw_dt,
        score_thr=args.precision_score_thr,
        iou_thr=args.precision_iou_thr,
    )

    metrics: dict[str, Any] = {
        "mAP50": round(ap50, 6),
        "mAP50-95": round(ap5095, 6),
        "recall": round(ar100, 6),
        "precision": round(prec, 6),
    }

    print(json.dumps(metrics, indent=2))
    print(
        "\n(recall = COCO AR maxDets=100 IoU=0.50:0.95; "
        f"precision = greedy TP/(TP+FP) at score>={args.precision_score_thr}, "
        f"IoU>={args.precision_iou_thr} vs GT; see docs/benchmark_metrics_schema.md)",
        file=sys.stderr,
    )

    if args.out_metrics_json:
        args.out_metrics_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Wrote {args.out_metrics_json}", file=sys.stderr)

    if args.out_patch_json:
        patch = {
            "metrics": metrics,
            "notes": [
                "Quality: scripts/eval_coco_predictions.py — unified COCOeval bbox on "
                "--gt-json and --dt-json; recall = COCO AR maxDets=100; "
                f"precision = greedy TP/(TP+FP), score>={args.precision_score_thr}, "
                f"IoU>={args.precision_iou_thr}."
            ],
        }
        args.out_patch_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_patch_json.write_text(json.dumps(patch, indent=2), encoding="utf-8")
        print(f"Wrote {args.out_patch_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
