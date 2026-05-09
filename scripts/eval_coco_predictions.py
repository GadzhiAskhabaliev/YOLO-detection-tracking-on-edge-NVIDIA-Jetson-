#!/usr/bin/env python3
"""
Единая оценка детекции по уже сохранённым предиктам (COCO bbox JSON).

Идея: любая модель (MMDet, Ultralytics export, FreeYOLO dump, …) переводится в
формат списка детекций COCO — тот же GT (`instances`-style), что и у eval.
Тогда mAP50 / mAP50-95 / AR считаются **одним** кодом (pycocotools), без расхождений
между фреймворками.

Формат dt (минимально нужное поле на объект):
  [{"image_id": int, "category_id": int, "bbox": [x, y, w, h], "score": float}, ...]

Пример:
  python3 scripts/eval_coco_predictions.py \\
    --gt-json /path/to/CrowdHuman/annotations/val.json \\
    --dt-json /path/to/my_model_val_bbox.json \\
    --out-metrics-json /tmp/metrics_subset.json

FPS этим скриптом не меряется — только качество по предиктам.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gt-json", type=Path, required=True, help="COCO GT (instances)")
    p.add_argument(
        "--dt-json",
        type=Path,
        required=True,
        help="COCO-style detection list or результат mmcv/json.dump списка",
    )
    p.add_argument(
        "--out-metrics-json",
        type=Path,
        default=None,
        help="Записать только объект metrics (mAP50, mAP50-95, recall=AR)",
    )
    p.add_argument(
        "--out-patch-json",
        type=Path,
        default=None,
        help="Готовый patch для bench_runner --merge-json: {metrics, notes}",
    )
    p.add_argument(
        "--quiet-summarize",
        action="store_true",
        help="Не печатать стандартный блок summarize() в stdout",
    )
    args = p.parse_args()

    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as e:
        raise SystemExit(
            "Нужен пакет pycocotools: pip install pycocotools\n" + str(e)
        ) from e

    gt_path = args.gt_json.expanduser().resolve()
    dt_path = args.dt_json.expanduser().resolve()
    if not gt_path.is_file():
        raise SystemExit(f"GT не найден: {gt_path}")
    if not dt_path.is_file():
        raise SystemExit(f"DT не найден: {dt_path}")

    raw_dt = json.loads(dt_path.read_text(encoding="utf-8"))
    if isinstance(raw_dt, dict) and "annotations" in raw_dt:
        raw_dt = raw_dt["annotations"]
    if not isinstance(raw_dt, list):
        raise SystemExit("dt-json должен быть списком детекций или dict с ключом 'annotations'")

    coco_gt = COCO(str(gt_path))
    coco_dt = coco_gt.loadRes(raw_dt)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    if not args.quiet_summarize:
        coco_eval.summarize()

    # Совпадает с порядком stats после summarize() в pycocotools
    stats = coco_eval.stats
    ap5095 = float(stats[0])
    ap50 = float(stats[1])
    ar100 = float(stats[8])  # AR IoU=0.50:0.95 area=all maxDets=100

    metrics: dict[str, Any] = {
        "mAP50": round(ap50, 6),
        "mAP50-95": round(ap5095, 6),
        "recall": round(ar100, 6),
    }

    print(json.dumps(metrics, indent=2))
    print(
        "\n(recall здесь = COCO AR maxDets=100 IoU=0.50:0.95; см. docs/BENCHMARK_METRICS_SCHEMA.md)",
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
                "Качество: scripts/eval_coco_predictions.py — единый COCOeval bbox по "
                "--gt-json и --dt-json; recall = COCO AR maxDets=100."
            ],
        }
        args.out_patch_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_patch_json.write_text(json.dumps(patch, indent=2), encoding="utf-8")
        print(f"Wrote {args.out_patch_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
