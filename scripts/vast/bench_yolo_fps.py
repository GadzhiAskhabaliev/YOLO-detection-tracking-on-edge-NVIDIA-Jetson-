#!/usr/bin/env python3
"""Rough FPS / latency for Ultralytics YOLO on GPU (predict pipeline)."""
from __future__ import annotations

import argparse
import json
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
    args = p.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights).to(args.device)
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


if __name__ == "__main__":
    main()
