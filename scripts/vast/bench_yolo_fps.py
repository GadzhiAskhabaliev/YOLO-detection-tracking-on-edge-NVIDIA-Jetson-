#!/usr/bin/env python3
"""Rough FPS / latency for Ultralytics YOLO on GPU (predict pipeline)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-json", default="")
    p.add_argument(
        "--record-model-name",
        default="",
        help="If set, also writes unified results/runs/*.json via scripts/bench_runner.py",
    )
    p.add_argument(
        "--weights-hub",
        default="",
        help="Stored in run JSON when --record-model-name is used (e.g. yakhyo/yolov8-crowdhuman)",
    )
    args = p.parse_args()

    wp = Path(args.weights)
    if not wp.is_file():
        raise SystemExit(
            f"Weights not found: {wp}\n"
            "Run: bash scripts/vast/download_yolov8n_crowdhuman.sh\n"
            "Or pass --weights /full/path/to.pt"
        )

    from ultralytics import YOLO

    model = YOLO(str(wp)).to(args.device)
    dummy = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)

    for _ in range(args.warmup):
        model.predict(source=dummy, imgsz=args.imgsz, verbose=False, half=False)

    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        model.predict(source=dummy, imgsz=args.imgsz, verbose=False, half=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    ms = (dt / args.iters) * 1000.0
    fps = args.iters / dt
    payload = {
        "weights": args.weights,
        "imgsz": args.imgsz,
        "warmup": args.warmup,
        "iters": args.iters,
        "latency_ms_mean": ms,
        "fps": fps,
    }
    print(json.dumps(payload, indent=2))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.record_model_name.strip():
        repo_root = Path(__file__).resolve().parents[2]
        scripts_dir = repo_root / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from bench_runner import default_payload, save_result

        full = default_payload(
            model_name=args.record_model_name.strip(),
            weights_path=wp.resolve(),
            weights_hub=args.weights_hub.strip(),
            batch_size=1,
            imgsz=args.imgsz,
            backend="ultralytics_yolo",
        )
        full["metrics"]["fps_predict"] = round(fps, 3)
        full["metrics"]["inference_time_ms_predict"] = round(ms, 4)
        saved = save_result(full)
        print(json.dumps({"bench_runner_saved": str(saved.relative_to(repo_root))}, indent=2))


if __name__ == "__main__":
    main()
