#!/usr/bin/env python3
"""Run YOLOv8 detections + BoxMOT tracker on video or MOT17 img1 folder."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

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


def list_image_files(source_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    images.sort(key=lambda p: (not p.stem.isdigit(), int(p.stem) if p.stem.isdigit() else p.name))
    return images


def iter_frames(source: Path) -> Iterator[tuple[int, int, str, np.ndarray]]:
    if source.is_dir():
        images = list_image_files(source)
        if not images:
            raise SystemExit(f"No images found in source directory: {source}")
        for idx, image_path in enumerate(images, start=1):
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise SystemExit(f"Failed to read frame: {image_path}")
            yield idx, infer_frame_id(image_path.name, idx), str(image_path), frame
        return

    if not source.is_file():
        raise SystemExit(f"Source does not exist: {source}")

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video source: {source}")
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        yield idx, idx, str(source), frame
    cap.release()
    if idx == 0:
        raise SystemExit(f"No frames decoded from video source: {source}")


def parse_yolo_boxes(result: Any) -> np.ndarray:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return np.empty((0, 6), dtype=np.float32)
    xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    conf = boxes.conf.detach().cpu().numpy().reshape(-1, 1).astype(np.float32)
    cls = boxes.cls.detach().cpu().numpy().reshape(-1, 1).astype(np.float32)
    return np.concatenate([xyxy, conf, cls], axis=1)


def draw_tracks(frame: np.ndarray, tracks: list[dict[str, Any]]) -> np.ndarray:
    canvas = frame.copy()
    for tr in tracks:
        x1, y1, x2, y2 = [int(v) for v in tr["bbox_xyxy"]]
        tid = tr["track_id"]
        conf = tr["conf"]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (80, 220, 80), 2)
        cv2.putText(
            canvas,
            f"id={tid} conf={conf:.2f}",
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (80, 220, 80),
            1,
            cv2.LINE_AA,
        )
    return canvas


def main() -> None:
    default_weights = default_models_dir() / "yolov8n_crowdhuman.pt"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=str(default_weights))
    parser.add_argument("--source", required=True, help="Video path or MOT17 img1 directory")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--classes", default="0", help="Comma-separated classes, default 0 (person)")
    parser.add_argument("--tracker-type", default="strongsort", help="boxmot tracker key (e.g. botsort, hybridsort)")
    parser.add_argument("--tracker-label", default="", help="Label used in run_tag/report; defaults to tracker_type")
    parser.add_argument("--reid-weights", default="osnet_x0_25_msmt17.pt")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--save-mot-txt", action="store_true")
    parser.add_argument("--project", default="results/tracking/runs")
    parser.add_argument("--name", default="", help="If empty, auto-generated from tracker label")
    parser.add_argument("--report-dir", default="results/tracking")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("ultralytics is required: pip install ultralytics") from exc
    try:
        from boxmot.trackers.tracker_zoo import create_tracker
    except ImportError as exc:
        raise SystemExit("boxmot is required: pip install boxmot") from exc

    weights = Path(args.weights).expanduser().resolve()
    source = Path(args.source).expanduser().resolve()
    if not weights.is_file():
        raise SystemExit(f"Weights not found: {weights}")
    if not source.exists():
        raise SystemExit(f"Source not found: {source}")

    tracker_type = args.tracker_type.strip().lower()
    tracker_label = args.tracker_label.strip().lower() or tracker_type
    run_name = args.name.strip() or f"yolov8_{tracker_label}"

    report_dir = Path(args.report_dir).expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    run_tag = f"{run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    raw_json_path = report_dir / f"{run_tag}_raw_tracks.json"
    report_json_path = report_dir / f"{run_tag}_run_report.json"
    mot_txt_path = report_dir / f"{run_tag}.txt"

    run_dir = Path(args.project).expanduser().resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    video_path = run_dir / "tracks.mp4"
    video_writer: cv2.VideoWriter | None = None

    classes = parse_classes(args.classes)
    model = YOLO(str(weights))
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

    for frame_idx, frame_id, frame_path, frame in iter_frames(source):
        yolo_result = model.predict(
            source=frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            classes=classes,
            verbose=False,
        )[0]
        dets = parse_yolo_boxes(yolo_result)
        tracks = tracker.update(dets, frame)

        det_rows = []
        for det in dets:
            x1, y1, x2, y2, conf, cls_id = det.tolist()
            det_rows.append(
                {
                    "class_id": int(cls_id),
                    "conf": float(conf),
                    "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                    "bbox_xywh": [float(x1), float(y1), max(0.0, float(x2 - x1)), max(0.0, float(y2 - y1))],
                }
            )

        track_rows = []
        for tr in np.asarray(tracks):
            if len(tr) < 8:
                continue
            x1, y1, x2, y2, track_id, conf, cls_id, _det_ind = tr[:8].tolist()
            rec = {
                "track_id": int(track_id),
                "class_id": int(cls_id),
                "conf": float(conf),
                "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "bbox_xywh": [float(x1), float(y1), max(0.0, float(x2 - x1)), max(0.0, float(y2 - y1))],
            }
            track_rows.append(rec)
            if args.save_mot_txt:
                mot_rows.append(
                    f"{frame_id},{int(track_id)},{float(x1):.3f},{float(y1):.3f},{float(x2 - x1):.3f},"
                    f"{float(y2 - y1):.3f},{float(conf):.6f},-1,-1,-1"
                )

        if args.save_video:
            if video_writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(str(video_path), fourcc, 25.0, (w, h))
            video_writer.write(draw_tracks(frame, track_rows))

        frames.append(
            {
                "frame": int(frame_id),
                "index": int(frame_idx),
                "path": frame_path,
                "detections": det_rows,
                "tracks": track_rows,
            }
        )

    if video_writer is not None:
        video_writer.release()

    wall_seconds = max(1e-9, time.perf_counter() - start)
    frame_count = len(frames)
    fps_e2e = frame_count / wall_seconds
    latency_ms = (wall_seconds / max(1, frame_count)) * 1000.0

    raw_payload = {
        "created_at": utc_now(),
        "weights": str(weights),
        "source": str(source),
        "tracker": tracker_type,
        "tracker_label": tracker_label,
        "reid_weights": args.reid_weights,
        "frames": frames,
    }
    raw_json_path.write_text(json.dumps(raw_payload, indent=2), encoding="utf-8")

    if args.save_mot_txt:
        mot_txt_path.write_text("\n".join(mot_rows) + ("\n" if mot_rows else ""), encoding="utf-8")

    report_payload = {
        "created_at": utc_now(),
        "run_tag": run_tag,
        "weights": str(weights),
        "source": str(source),
        "tracker": tracker_type,
        "tracker_label": tracker_label,
        "reid_weights": args.reid_weights,
        "fps_e2e": round(fps_e2e, 4),
        "latency_ms_e2e": round(latency_ms, 4),
        "wall_seconds": round(wall_seconds, 4),
        "frames": frame_count,
        "params": {
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "device": args.device,
            "classes": classes,
        },
        "artifacts": {
            "raw_tracks_json": str(raw_json_path),
            "mot_txt": str(mot_txt_path) if args.save_mot_txt else "",
            "video": str(video_path) if args.save_video else "",
        },
    }
    report_json_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    print(json.dumps(report_payload, indent=2))


if __name__ == "__main__":
    main()
