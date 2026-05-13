#!/usr/bin/env python3
"""Validate CrowdHuman and MOT17 layout and print a concise summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
from path_defaults import default_data_dir


def count_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore"))


def count_images(path: Path) -> int:
    if not path.is_dir():
        return 0
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    total = 0
    for ext in exts:
        total += len(list(path.glob(ext)))
    return total


def inspect_mot17_train(mot17_train_root: Path) -> dict:
    out: dict[str, object] = {"root": str(mot17_train_root), "exists": mot17_train_root.is_dir()}
    sequences = []
    if mot17_train_root.is_dir():
        for seq_dir in sorted(p for p in mot17_train_root.iterdir() if p.is_dir()):
            img1 = seq_dir / "img1"
            gt = seq_dir / "gt" / "gt.txt"
            seqinfo = seq_dir / "seqinfo.ini"
            sequences.append(
                {
                    "name": seq_dir.name,
                    "img1_exists": img1.is_dir(),
                    "frames": count_images(img1),
                    "gt_exists": gt.is_file(),
                    "gt_rows": count_lines(gt),
                    "seqinfo_exists": seqinfo.is_file(),
                }
            )
    out["sequences"] = sequences
    out["sequence_count"] = len(sequences)
    out["total_frames"] = sum(int(s["frames"]) for s in sequences)
    return out


def inspect_crowdhuman(crowdhuman_root: Path) -> dict:
    images = crowdhuman_root / "Images"
    odgt = crowdhuman_root / "annotation_val.odgt"
    yolo_images = crowdhuman_root / "yolo" / "images" / "val"
    yolo_labels = crowdhuman_root / "yolo" / "labels" / "val"
    return {
        "root": str(crowdhuman_root),
        "exists": crowdhuman_root.is_dir(),
        "images_exists": images.is_dir(),
        "images_count": count_images(images),
        "annotation_val_odgt_exists": odgt.is_file(),
        "annotation_val_odgt_rows": count_lines(odgt),
        "yolo_images_val_exists": yolo_images.is_dir(),
        "yolo_images_val_count": count_images(yolo_images),
        "yolo_labels_val_exists": yolo_labels.is_dir(),
        "yolo_labels_val_count": len(list(yolo_labels.glob("*.txt"))) if yolo_labels.is_dir() else 0,
    }


def main() -> None:
    data_root = default_data_dir()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crowdhuman-root", default=str(data_root / "crowdhuman"))
    parser.add_argument("--mot17-root", default=str(data_root / "mot17"))
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()

    crowdhuman_root = Path(args.crowdhuman_root).expanduser().resolve()
    mot17_root = Path(args.mot17_root).expanduser().resolve()
    mot17_train_root = mot17_root / "MOT17" / "train"

    payload = {
        "crowdhuman": inspect_crowdhuman(crowdhuman_root),
        "mot17": inspect_mot17_train(mot17_train_root),
    }

    print(json.dumps(payload, indent=2))
    if args.out_json:
        out_path = Path(args.out_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
