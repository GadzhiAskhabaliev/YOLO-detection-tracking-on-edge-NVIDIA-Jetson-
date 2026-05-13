#!/usr/bin/env python3
"""Convert raw tracking JSON from run_yolov8_bytetrack_mot17.py to MOTChallenge txt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-json", required=True, help="Raw JSON from run_yolov8_bytetrack_mot17.py")
    parser.add_argument("--out-txt", required=True, help="Output MOT txt file")
    parser.add_argument("--min-conf", type=float, default=0.0)
    parser.add_argument("--strict", action="store_true", help="Fail if there are missing frame ids")
    args = parser.parse_args()

    in_path = Path(args.in_json).expanduser().resolve()
    out_path = Path(args.out_txt).expanduser().resolve()
    if not in_path.is_file():
        raise SystemExit(f"Input JSON not found: {in_path}")

    payload = json.loads(in_path.read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise SystemExit("Input JSON must contain frames[]")

    rows = []
    frame_ids = []
    kept = 0
    skipped = 0
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        frame_id = int(frame.get("frame", 0))
        frame_ids.append(frame_id)
        detections = frame.get("detections", [])
        if not isinstance(detections, list):
            continue
        for det in detections:
            if not isinstance(det, dict):
                continue
            track_id = int(det.get("track_id", -1))
            if track_id < 0:
                skipped += 1
                continue
            conf = float(det.get("conf", 0.0))
            if conf < args.min_conf:
                skipped += 1
                continue
            bbox = det.get("bbox_xywh")
            if not isinstance(bbox, list) or len(bbox) != 4:
                skipped += 1
                continue
            x, y, w, h = [float(v) for v in bbox]
            rows.append(f"{frame_id},{track_id},{x:.3f},{y:.3f},{w:.3f},{h:.3f},{conf:.6f},-1,-1,-1")
            kept += 1

    if not frame_ids:
        raise SystemExit("No frames found in input JSON")

    unique_ids = sorted(set(frame_ids))
    missing_frames = []
    for idx in range(unique_ids[0], unique_ids[-1] + 1):
        if idx not in unique_ids:
            missing_frames.append(idx)

    if args.strict and missing_frames:
        raise SystemExit(f"Missing frames in sequence: {missing_frames[:20]}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    print(
        json.dumps(
            {
                "in_json": str(in_path),
                "out_txt": str(out_path),
                "rows_written": kept,
                "rows_skipped": skipped,
                "frame_count": len(unique_ids),
                "missing_frames": len(missing_frames),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
