#!/usr/bin/env python3
"""Run YOLOv8 + ByteTrack on video or MOT17 img1 folder."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
from path_defaults import default_models_dir


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_classes(text: str) -> list[int] | None:
    if not text.strip():
        return None
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def infer_frame_id(path_value: str, index: int) -> int:
    stem = Path(path_value).stem
    if stem.isdigit():
        return int(stem)
    return index


def detections_from_result(result: Any, frame_idx: int) -> dict[str, Any]:
    rows = []
    boxes = getattr(result, "boxes", None)
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        if getattr(boxes, "id", None) is not None:
            ids = boxes.id.cpu().numpy().astype(int)
        else:
            ids = [-1] * len(xyxy)
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = [float(xyxy[i, j]) for j in range(4)]
            rows.append(
                {
                    "track_id": int(ids[i]),
                    "class_id": int(cls[i]),
                    "conf": float(conf[i]),
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "bbox_xywh": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                }
            )
    return {"frame": int(frame_idx), "path": str(getattr(result, "path", "")), "detections": rows}


def main() -> None:
    default_weights = default_models_dir() / "yolov8n_crowdhuman.pt"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=str(default_weights))
    parser.add_argument("--source", required=True, help="Video path or MOT17 img1 directory")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--classes", default="0", help="Comma-separated classes, default 0 (person)")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--save-mot-txt", action="store_true")
    parser.add_argument("--project", default="results/tracking/runs")
    parser.add_argument("--name", default="yolov8_bytetrack")
    parser.add_argument("--report-dir", default="results/tracking")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("ultralytics is required: pip install ultralytics") from exc

    weights = Path(args.weights).expanduser().resolve()
    source = Path(args.source).expanduser().resolve()
    if not weights.is_file():
        raise SystemExit(f"Weights not found: {weights}")
    if not source.exists():
        raise SystemExit(f"Source not found: {source}")

    report_dir = Path(args.report_dir).expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    run_tag = f"{args.name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    raw_json_path = report_dir / f"{run_tag}_raw_tracks.json"
    report_json_path = report_dir / f"{run_tag}_run_report.json"
    mot_txt_path = report_dir / f"{run_tag}.txt"

    classes = parse_classes(args.classes)
    model = YOLO(str(weights))

    start = time.perf_counter()
    frames: list[dict[str, Any]] = []
    for frame_idx, result in enumerate(
        model.track(
            source=str(source),
            tracker=args.tracker,
            persist=True,
            stream=True,
            save=args.save_video,
            project=args.project,
            name=args.name,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            classes=classes,
            verbose=False,
        ),
        start=1,
    ):
        path_value = str(getattr(result, "path", ""))
        frame_id = infer_frame_id(path_value, frame_idx)
        frames.append(detections_from_result(result, frame_id))

    wall_seconds = max(1e-9, time.perf_counter() - start)
    frame_count = len(frames)
    fps_e2e = frame_count / wall_seconds
    latency_ms = (wall_seconds / max(1, frame_count)) * 1000.0

    raw_payload = {
        "created_at": utc_now(),
        "weights": str(weights),
        "source": str(source),
        "tracker": args.tracker,
        "frames": frames,
    }
    raw_json_path.write_text(json.dumps(raw_payload, indent=2), encoding="utf-8")

    if args.save_mot_txt:
        rows = []
        for frame in frames:
            fidx = int(frame["frame"])
            for det in frame["detections"]:
                if int(det["track_id"]) < 0:
                    continue
                x, y, w, h = det["bbox_xywh"]
                rows.append(
                    f"{fidx},{int(det['track_id'])},{x:.3f},{y:.3f},{w:.3f},{h:.3f},{float(det['conf']):.6f},-1,-1,-1"
                )
        mot_txt_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")

    report_payload = {
        "created_at": utc_now(),
        "run_tag": run_tag,
        "weights": str(weights),
        "source": str(source),
        "fps_e2e": round(fps_e2e, 4),
        "latency_ms_e2e": round(latency_ms, 4),
        "wall_seconds": round(wall_seconds, 4),
        "frames": frame_count,
        "params": {
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "device": args.device,
            "tracker": args.tracker,
            "classes": classes,
        },
        "artifacts": {
            "raw_tracks_json": str(raw_json_path),
            "mot_txt": str(mot_txt_path) if args.save_mot_txt else "",
        },
    }
    report_json_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    print(json.dumps(report_payload, indent=2))


if __name__ == "__main__":
    main()
