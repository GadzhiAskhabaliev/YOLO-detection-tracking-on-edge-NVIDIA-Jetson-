#!/usr/bin/env python3
"""
Evaluate detection quality from a fixed split: **GT** (COCO instances JSON) + **DT** (predictions).

Consistency (not “must be MS COCO train2017”):
  • **One split** — e.g. CrowdHuman **val**: one GT file lists all images and annotations.
  • **Predictions** must reference **`image_id`** values present in GT (`images[].id`).
  • GT uses **COCO instances** as a convenient container for pycocotools.

DT format — list of dicts (or a dict with key `"annotations"`):
  {"image_id": int, "category_id": int, "bbox": [x, y, w, h], "score": float}
  bbox in pixels, xywh, COCO-style.

Outputs: mAP50, mAP50-95, recall (= COCO AR maxDets=100). Does **not** compute FPS.

Example:
  python3 scripts/eval_coco_predictions.py \\
    --gt-json .../CrowdHuman/annotations/val.json \\
    --dt-json .../my_model_val_predictions.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


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
        help="Write metrics object only (mAP50, mAP50-95, recall=AR)",
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

    metrics: dict[str, Any] = {
        "mAP50": round(ap50, 6),
        "mAP50-95": round(ap5095, 6),
        "recall": round(ar100, 6),
    }

    print(json.dumps(metrics, indent=2))
    print(
        "\n(recall here = COCO AR maxDets=100 IoU=0.50:0.95; see docs/benchmark_metrics_schema.md)",
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
                "--gt-json and --dt-json; recall = COCO AR maxDets=100."
            ],
        }
        args.out_patch_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_patch_json.write_text(json.dumps(patch, indent=2), encoding="utf-8")
        print(f"Wrote {args.out_patch_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
