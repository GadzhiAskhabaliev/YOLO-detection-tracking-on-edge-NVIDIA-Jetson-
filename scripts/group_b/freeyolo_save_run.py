#!/usr/bin/env python3
"""Persist FreeYOLO benchmark rows via bench_runner.save_result."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from bench_runner import default_payload, save_result


def parse_eval_log(text: str) -> tuple[float | None, float | None, float | None]:
    """Parse ap50_95, ap50, recall_ar (COCO AR IoU=0.50:0.95, area=all, maxDets=100 from summarize)."""
    ap50_95 = ap50 = recall_ar = None
    m = re.search(r"ap50_95\s*:\s*([\d.eE+-]+)", text)
    if m:
        ap50_95 = float(m.group(1))
    m = re.search(r"ap50\s*:\s*([\d.eE+-]+)", text)
    if m:
        ap50 = float(m.group(1))
    # summarize tolerates spaces: | area=   all | maxDets= 100 ] or maxDets=100
    m_ar = re.search(
        r"Average Recall\s+\(AR\)\s+@\[\s*IoU=0\.50:0\.95\s*\|\s*area=\s*all\s*\|\s*"
        r"maxDets\s*=\s*100\s*\]\s*=\s*([\d.eE+-]+)",
        text,
    )
    if m_ar:
        recall_ar = float(m_ar.group(1))
    return ap50_95, ap50, recall_ar


def run_speed_bench(
    *,
    freeyolo_home: Path,
    weights: Path,
    variant: str,
    imgsz: int,
    warmup: int,
    iters: int,
) -> dict | None:
    script = REPO_ROOT / "scripts" / "group_b" / "freeyolo_speed_bench.py"
    cmd = [
        sys.executable,
        str(script),
        "--freeyolo-home",
        str(freeyolo_home.resolve()),
        "--weights",
        str(weights.resolve()),
        "--variant",
        variant,
        "--img-size",
        str(imgsz),
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        print(f"freeyolo_speed_bench: not started ({e})", file=sys.stderr)
        return None
    if proc.returncode != 0:
        print(
            f"freeyolo_speed_bench exit {proc.returncode}\n{proc.stderr}",
            file=sys.stderr,
        )
        return None
    line = (proc.stdout or "").strip().splitlines()
    raw = line[-1] if line else ""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print(f"freeyolo_speed_bench: stdout not JSON: {raw[:200]!r}", file=sys.stderr)
        return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-log", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--variant", default="yolo_free_nano")
    p.add_argument("--weights-uri", default="https://github.com/yjh0410/FreeYOLO/releases/download/weight/yolo_free_nano_ch.pth")
    p.add_argument("--wall-seconds", type=float, default=None)
    p.add_argument("--num-images", type=int, default=None)
    p.add_argument(
        "--freeyolo-home",
        type=Path,
        default=None,
        help="FreeYOLO clone path; if set, runs speed microbench after eval (same Python/venv).",
    )
    p.add_argument("--skip-speed-bench", action="store_true")
    p.add_argument("--speed-warmup", type=int, default=20)
    p.add_argument("--speed-iters", type=int, default=100)
    p.add_argument(
        "--model-name",
        default="freeyolo_yolox_mot17",
        help="Row name in results/runs and README (variants should differ)",
    )
    p.add_argument("--detector-label", default="", help="Optional display label for reports")
    p.add_argument("--imgsz", type=int, default=640)
    args = p.parse_args()

    text = args.eval_log.read_text(encoding="utf-8", errors="replace")
    ap5095, ap50, recall_ar = parse_eval_log(text)
    if ap50 is None:
        raise SystemExit("Could not parse ap50 from eval.py log")

    fy_home = args.freeyolo_home
    if fy_home is None:
        env_fy = os.environ.get("FREEYOLO_HOME", "").strip()
        if env_fy:
            fy_home = Path(env_fy)

    speed: dict | None = None
    if fy_home and not args.skip_speed_bench:
        speed = run_speed_bench(
            freeyolo_home=fy_home,
            weights=args.weights,
            variant=args.variant,
            imgsz=args.imgsz,
            warmup=args.speed_warmup,
            iters=args.speed_iters,
        )

    fps_eval_e2e = None
    if args.wall_seconds and args.num_images and args.wall_seconds > 0:
        fps_eval_e2e = round(args.num_images / args.wall_seconds, 4)

    label = args.detector_label.strip() or f"FreeYOLO {args.variant} CrowdHuman"
    payload = default_payload(
        model_name=args.model_name.strip(),
        weights_path=args.weights.resolve(),
        weights_hub=args.weights_uri,
        batch_size=1,
        imgsz=args.imgsz,
        backend="freeyolo",
        group="B",
        detector_id=7,
        detector_label=label,
    )

    payload["metrics"]["mAP50"] = round(ap50, 6)
    if ap5095 is not None:
        payload["metrics"]["mAP50-95"] = round(ap5095, 6)
    if recall_ar is not None:
        payload["metrics"]["recall"] = round(recall_ar, 6)
        payload["notes"].append(
            "recall: COCO Average Recall (AR) IoU=0.50:0.95, maxDets=100 from pycocotools summarize; "
            "see docs/benchmark_metrics_schema.md — do not mix with other val protocols."
        )
    if speed:
        for k in (
            "fps_forward",
            "forward_time_ms_mean",
            "fps_predict",
            "inference_time_ms_predict",
            "inference_time_ms",
        ):
            if k in speed:
                payload["metrics"][k] = speed[k]
        if speed.get("speed_note"):
            payload["notes"].append(str(speed["speed_note"]))
    elif fy_home and not args.skip_speed_bench:
        payload["notes"].append(
            "Speed microbench failed (see stderr); FPS fields unset."
        )
    elif not fy_home and not args.skip_speed_bench:
        payload["notes"].append(
            "Speed microbench skipped: pass --freeyolo-home or set FREEYOLO_HOME."
        )

    if fps_eval_e2e is not None:
        payload["metrics"]["eval_throughput_fps"] = fps_eval_e2e
        payload["metrics"]["eval_wall_seconds"] = round(args.wall_seconds, 3)
        payload["metrics"]["eval_images"] = args.num_images
        payload["notes"].append(
            "eval_throughput_fps = num_validation_frames / wall_time(eval.py) "
            "(includes COCOeval on CPU); compare to YOLOv8 using fps_predict from the microbench."
        )
    payload["notes"].append(
        f"FreeYOLO eval.py -d crowdhuman, variant={args.variant}, bench model={args.model_name}; "
        "CrowdHuman val split."
    )

    out = save_result(payload)
    print(json.dumps({"saved": str(out.relative_to(REPO_ROOT)), "mAP50": ap50, "mAP50-95": ap5095}, indent=2))


if __name__ == "__main__":
    main()
