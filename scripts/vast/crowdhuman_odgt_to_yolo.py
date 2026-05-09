#!/usr/bin/env python3
"""
Convert CrowdHuman .odgt (one JSON object per line) to YOLO label .txt files.

Skips gtboxes with tag != person, extra.ignore, or missing box.
Coordinates are normalized using actual image width/height from disk.

Example:
  cd /workspace/data/crowdhuman
  python3 /path/to/repo/scripts/vast/crowdhuman_odgt_to_yolo.py \\
    --odgt annotation_val.odgt --images-dir Images --out-dir labels_val
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

try:
    import cv2
except ImportError as e:
    raise SystemExit("Need OpenCV: pip install opencv-python-headless") from e


def parse_args():
    p = argparse.ArgumentParser(description="CrowdHuman ODGT → YOLO txt")
    p.add_argument("--odgt", type=Path, required=True)
    p.add_argument("--images-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--box-priority",
        choices=("vbox_first", "fbox_first"),
        default="vbox_first",
        help="Prefer visible vs full body box",
    )
    return p.parse_args()


def image_key_and_path(raw: dict, images_dir: Path) -> tuple[str, Path] | None:
    """CrowdHuman IDs look like '273271,deadbeef.jpg' — basename MUST keep the numeric prefix."""
    ref = raw.get("ID") or raw.get("img_path") or raw.get("name")
    if not ref:
        return None
    ref_str = str(ref).strip().replace("\\", "/")
    fname = Path(ref_str).name
    stem = Path(fname).stem
    img_path = images_dir / fname
    if img_path.is_file():
        return stem, img_path
    # Rare dumps use nested paths inside img_path; try basename-only anywhere (slow fallback).
    matches = list(images_dir.glob(fname))
    if len(matches) == 1:
        p = matches[0]
        return p.stem, p
    return None


def pick_box(gt: dict, priority: str) -> list[float] | None:
    if priority == "vbox_first":
        box = gt.get("vbox") or gt.get("fbox")
    else:
        box = gt.get("fbox") or gt.get("vbox")
    if box is None or len(box) < 4:
        return None
    return [float(x) for x in box[:4]]


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    skipped_no_img = 0
    with args.odgt.open(encoding="utf-8") as f:
        odgt_lines = f.readlines()
    for line in tqdm(odgt_lines, desc=args.odgt.name):
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)

        parsed = image_key_and_path(raw, args.images_dir)
        if parsed is None:
            skipped_no_img += 1
            continue
        stem, img_path = parsed

        im = cv2.imread(str(img_path))
        if im is None:
            skipped_no_img += 1
            continue
        ih, iw = im.shape[:2]
        if iw < 1 or ih < 1:
            continue

        rows: list[str] = []
        for g in raw.get("gtboxes", []):
            if g.get("tag") != "person":
                continue
            extra = g.get("extra")
            if isinstance(extra, dict) and extra.get("ignore"):
                continue
            box = pick_box(g, args.box_priority)
            if box is None:
                continue
            x, y, w, h = box
            xc = (x + w / 2.0) / iw
            yc = (y + h / 2.0) / ih
            nw = w / iw
            nh = h / ih
            rows.append(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")

        out_file = args.out_dir / f"{stem}.txt"
        out_file.write_text("".join(rows), encoding="utf-8")

    if skipped_no_img:
        print(f"Note: skipped/missing images count ~ {skipped_no_img} (check paths vs ID field)")
    print(f"Labels written under {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
