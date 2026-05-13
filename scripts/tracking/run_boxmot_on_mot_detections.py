#!/usr/bin/env python3
"""Run BoxMOT tracker on MOT-format detector outputs (det.txt)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_det_map(det_txt: Path, min_conf: float = 0.0) -> dict[int, list[list[float]]]:
    dets: dict[int, list[list[float]]] = {}
    for raw in det_txt.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        frame = int(float(parts[0]))
        x = float(parts[2])
        y = float(parts[3])
        w = float(parts[4])
        h = float(parts[5])
        conf = float(parts[6])
        if conf < min_conf:
            continue
        x2 = x + w
        y2 = y + h
        dets.setdefault(frame, []).append([x, y, x2, y2, conf, 0.0])  # class=person
    return dets


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--img1-dir", required=True, help="MOT sequence img1 directory")
    p.add_argument("--det-txt", required=True, help="MOT detector file (det.txt)")
    p.add_argument("--tracker-type", required=True, help="boxmot tracker type")
    p.add_argument("--tracker-label", default="", help="output label; default tracker-type")
    p.add_argument("--detector-label", default="detector")
    p.add_argument("--reid-weights", default="osnet_x0_25_msmt17.pt")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--min-det-conf", type=float, default=0.0)
    p.add_argument("--report-dir", default="results/tracking")
    p.add_argument("--project", default="results/tracking/runs")
    args = p.parse_args()

    try:
        from boxmot.trackers.tracker_zoo import create_tracker
    except ImportError as exc:
        raise SystemExit("boxmot is required: pip install boxmot") from exc

    img1_dir = Path(args.img1_dir).expanduser().resolve()
    det_txt = Path(args.det_txt).expanduser().resolve()
    if not img1_dir.is_dir():
        raise SystemExit(f"img1 dir not found: {img1_dir}")
    if not det_txt.is_file():
        raise SystemExit(f"det txt not found: {det_txt}")

    tracker_type = args.tracker_type.strip().lower()
    tracker_label = args.tracker_label.strip().lower() or tracker_type
    detector_label = args.detector_label.strip().lower().replace(" ", "_")
    seq_name = img1_dir.parent.name

    report_dir = Path(args.report_dir).expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    run_name = f"{detector_label}_{tracker_label}_{seq_name}"
    run_tag = f"{run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    raw_json_path = report_dir / f"{run_tag}_raw_tracks.json"
    report_json_path = report_dir / f"{run_tag}_run_report.json"
    mot_txt_path = report_dir / f"{run_tag}.txt"

    det_map = load_det_map(det_txt, min_conf=args.min_det_conf)
    images = sorted(
        [p for p in img1_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}],
        key=lambda p: int(p.stem) if p.stem.isdigit() else p.name,
    )
    if not images:
        raise SystemExit(f"No frames in {img1_dir}")

    tracker = create_tracker(
        tracker_type,
        reid_weights=args.reid_weights,
        device=args.device,
        half=False,
        per_class=False,
    )

    start = time.perf_counter()
    frames: list[dict[str, Any]] = []
    mot_rows: list[str] = []

    for idx, image_path in enumerate(images, start=1):
        frame_id = int(image_path.stem) if image_path.stem.isdigit() else idx
        img = cv2.imread(str(image_path))
        if img is None:
            raise SystemExit(f"Failed reading frame: {image_path}")
        dets = np.asarray(det_map.get(frame_id, []), dtype=np.float32)
        if dets.size == 0:
            dets = np.empty((0, 6), dtype=np.float32)
        tracks = np.asarray(tracker.update(dets, img))

        det_rows = []
        for d in dets:
            x1, y1, x2, y2, conf, cls_id = d.tolist()
            det_rows.append(
                {
                    "class_id": int(cls_id),
                    "conf": float(conf),
                    "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                    "bbox_xywh": [float(x1), float(y1), max(0.0, float(x2 - x1)), max(0.0, float(y2 - y1))],
                }
            )

        track_rows = []
        for tr in tracks:
            if len(tr) < 8:
                continue
            x1, y1, x2, y2, track_id, conf, cls_id, _det_ind = tr[:8].tolist()
            row = {
                "track_id": int(track_id),
                "class_id": int(cls_id),
                "conf": float(conf),
                "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "bbox_xywh": [float(x1), float(y1), max(0.0, float(x2 - x1)), max(0.0, float(y2 - y1))],
            }
            track_rows.append(row)
            mot_rows.append(
                f"{frame_id},{int(track_id)},{float(x1):.3f},{float(y1):.3f},{float(x2-x1):.3f},"
                f"{float(y2-y1):.3f},{float(conf):.6f},-1,-1,-1"
            )

        frames.append(
            {
                "frame": frame_id,
                "path": str(image_path),
                "detections": det_rows,
                "tracks": track_rows,
            }
        )

    wall_seconds = max(1e-9, time.perf_counter() - start)
    frame_count = len(frames)
    fps_e2e = frame_count / wall_seconds
    latency_ms = (wall_seconds / max(1, frame_count)) * 1000.0

    raw_payload = {
        "created_at": utc_now(),
        "detector_label": detector_label,
        "tracker_type": tracker_type,
        "tracker_label": tracker_label,
        "sequence": seq_name,
        "det_txt": str(det_txt),
        "frames": frames,
    }
    raw_json_path.write_text(json.dumps(raw_payload, indent=2), encoding="utf-8")
    mot_txt_path.write_text("\n".join(mot_rows) + ("\n" if mot_rows else ""), encoding="utf-8")

    report_payload = {
        "created_at": utc_now(),
        "run_tag": run_tag,
        "detector_label": detector_label,
        "tracker_type": tracker_type,
        "tracker_label": tracker_label,
        "sequence": seq_name,
        "det_txt": str(det_txt),
        "fps_e2e": round(fps_e2e, 4),
        "latency_ms_e2e": round(latency_ms, 4),
        "wall_seconds": round(wall_seconds, 4),
        "frames": frame_count,
        "artifacts": {
            "raw_tracks_json": str(raw_json_path),
            "mot_txt": str(mot_txt_path),
        },
    }
    report_json_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    print(json.dumps(report_payload, indent=2))


if __name__ == "__main__":
    main()
