#!/usr/bin/env python3
"""
FreeYOLO speed microbench (ValTransforms + GPU).

Output JSON keys follow docs/benchmark_metrics_schema.md:
fps_forward / forward_time_ms_mean — model(x), no_decode=True;
fps_predict / inference_time_ms — ValTransforms + full decode/NMS (no_decode=False).

Prints one JSON line to stdout; run under FreeYOLO venv.

  FREEYOLO_HOME=/path/to/FreeYOLO /path/to/venv/bin/python scripts/yolo_detectors/freeyolo_speed_bench.py \\
    --freeyolo-home "$FREEYOLO_HOME" --weights model.pth --variant yolo_free_nano
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--freeyolo-home", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--variant", default="yolo_free_nano")
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--device", default="", help="cuda | cpu; default cuda if available")
    args = p.parse_args()

    fy = args.freeyolo_home.resolve()
    if not fy.is_dir():
        print(json.dumps({"error": f"missing FreeYOLO dir: {fy}"}), file=sys.stderr)
        raise SystemExit(2)

    wpath = args.weights.resolve()
    if not wpath.is_file():
        print(json.dumps({"error": f"missing weights: {wpath}"}), file=sys.stderr)
        raise SystemExit(2)

    device_s = (args.device or "").strip().lower()
    if device_s == "cpu":
        device = torch.device("cpu")
    elif device_s == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sys.path.insert(0, str(fy))
    os.chdir(str(fy))

    ns = SimpleNamespace(
        dataset="crowdhuman",
        version=args.variant,
        img_size=args.img_size,
        mosaic=None,
        mixup=None,
        cuda=device.type == "cuda",
        weight=str(wpath),
        conf_thresh=0.005,
        nms_thresh=0.6,
        topk=1000,
        no_decode=False,
        root=str(fy),
    )

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        from config import build_config
        from dataset.transforms import ValTransforms
        from models import build_model
        from utils.misc import load_weight

        cfg = build_config(ns)
        model = build_model(args=ns, cfg=cfg, device=device, num_classes=1, trainable=False)
        model = load_weight(model=model, path_to_ckpt=str(wpath))
        model.to(device).eval()

    transform = ValTransforms(img_size=args.img_size)
    dummy_bgr = np.zeros((args.img_size, args.img_size, 3), dtype=np.uint8)

    def input_tensor():
        x_img, _ = transform(dummy_bgr)
        return x_img.unsqueeze(0).to(device)

    x0 = input_tensor()

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    # --- forward-only path (no_decode): no NMS / no numpy cpu-heavy decode ---
    model.no_decode = True
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(x0)
        sync()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            _ = model(x0)
        sync()
        dt_fwd = time.perf_counter() - t0

    ms_fwd = (dt_fwd / args.iters) * 1000.0
    fps_fwd = args.iters / dt_fwd

    # --- predict-like path: preprocess + full decode (as in eval) ---
    model.no_decode = False
    with torch.no_grad():
        for _ in range(args.warmup):
            xi = input_tensor()
            _ = model(xi)
        sync()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            xi = input_tensor()
            _ = model(xi)
        sync()
        dt_pred = time.perf_counter() - t0

    ms_pred = (dt_pred / args.iters) * 1000.0
    fps_pred = args.iters / dt_pred

    out = {
        "fps_forward": round(fps_fwd, 3),
        "forward_time_ms_mean": round(ms_fwd, 4),
        "fps_predict": round(fps_pred, 3),
        "inference_time_ms_predict": round(ms_pred, 4),
        "inference_time_ms": round(ms_pred, 4),
        "speed_device": str(device),
        "speed_warmup": args.warmup,
        "speed_iters": args.iters,
        "speed_note": (
            "FreeYOLO: forward=no_decode tensor output; predict=ValTransforms+full decode/NMS "
            "(docs/benchmark_metrics_schema.md)."
        ),
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
