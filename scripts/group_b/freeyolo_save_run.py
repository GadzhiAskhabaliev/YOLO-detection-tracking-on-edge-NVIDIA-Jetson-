#!/usr/bin/env python3
"""Пишет unified JSON прогона FreeYOLO через bench_runner.save_result."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from bench_runner import default_payload, save_result


def parse_eval_log(text: str) -> tuple[float | None, float | None]:
    ap50_95 = ap50 = None
    m = re.search(r"ap50_95\s*:\s*([\d.eE+-]+)", text)
    if m:
        ap50_95 = float(m.group(1))
    m = re.search(r"ap50\s*:\s*([\d.eE+-]+)", text)
    if m:
        ap50 = float(m.group(1))
    return ap50_95, ap50


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-log", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--variant", default="yolo_free_nano")
    p.add_argument("--weights-uri", default="https://github.com/yjh0410/FreeYOLO/releases/download/weight/yolo_free_nano_ch.pth")
    p.add_argument("--wall-seconds", type=float, default=None)
    p.add_argument("--num-images", type=int, default=None)
    args = p.parse_args()

    text = args.eval_log.read_text(encoding="utf-8", errors="replace")
    ap5095, ap50 = parse_eval_log(text)
    if ap50 is None:
        raise SystemExit("Не удалось распарсить ap50 из лога eval.py")

    fps_infer = None
    if args.wall_seconds and args.num_images and args.wall_seconds > 0:
        fps_infer = round(args.num_images / args.wall_seconds, 4)

    payload = default_payload(
        model_name="freeyolo_yolox_mot17",
        weights_path=args.weights.resolve(),
        weights_hub=args.weights_uri,
        batch_size=1,
        imgsz=640,
        group="B",
        detector_id=7,
        detector_label="FreeYOLO (CrowdHuman ckpt)",
    )
    payload["metrics"]["mAP50"] = round(ap50, 6)
    if ap5095 is not None:
        payload["metrics"]["mAP50-95"] = round(ap5095, 6)
    if fps_infer:
        payload["metrics"]["fps_predict"] = fps_infer
        payload["metrics"]["eval_wall_seconds"] = round(args.wall_seconds, 3)
        payload["metrics"]["eval_images"] = args.num_images
    payload["notes"].append(
        f"FreeYOLO eval.py -d crowdhuman, variant={args.variant}. "
        "Веса — CrowdHuman (nano); официальные MOT17 релизы в README часто пустые — слот группы B сохранён как freeyolo_yolox_mot17."
    )

    out = save_result(payload)
    print(json.dumps({"saved": str(out.relative_to(REPO_ROOT)), "mAP50": ap50, "mAP50-95": ap5095}, indent=2))


if __name__ == "__main__":
    main()
