#!/usr/bin/env python3
"""
Build a single COCO instances JSON from MOT17 **train** (GT `gt/gt.txt` + `img1/`).

MOT17 train repeats each sequence for three detector variants (**-DPM**, **-FRCNN**, **-SDP**).
Unless you want triple copies of every frame, pass **`--det-subdir-suffix FRCNN`** (recommended).

Image `file_name` is relative to the MOT17 **train** root, e.g.
  `MOT17-02-FRCNN/img1/000001.jpg`

Each image record adds optional keys **`mot_sequence`**, **`mot_frame`** for MOT-format dumps (ignored by COCOeval).

Example:

  python3 scripts/mot17_gt_to_coco.py \\
    --mot17-train-root /workspace/data/mot17/MOT17/train \\
    --det-subdir-suffix FRCNN \\
    --out-json /workspace/data/mot17/annotations/mot17_train_frcnn_gt.json
"""
from __future__ import annotations

import argparse
import configparser
import json
import re
from pathlib import Path


def _parse_seqinfo(seq_dir: Path) -> tuple[int, int] | None:
    ini = seq_dir / "seqinfo.ini"
    if not ini.is_file():
        return None
    cfg = configparser.ConfigParser()
    cfg.read(ini, encoding="utf-8")
    sec = "Sequence" if cfg.has_section("Sequence") else None
    if sec is None:
        return None
    try:
        w = int(cfg.get(sec, "imWidth"))
        h = int(cfg.get(sec, "imHeight"))
        return w, h
    except (configparser.NoOptionError, ValueError):
        return None


def _frame_from_filename(stem: str) -> int | None:
    if stem.isdigit():
        return int(stem)
    m = re.match(r"^0*(\d+)$", stem)
    return int(m.group(1)) if m else None


def _list_sequences(train_root: Path, suffix: str | None) -> list[Path]:
    dirs = sorted([p for p in train_root.iterdir() if p.is_dir()])
    out: list[Path] = []
    for p in dirs:
        if not (p / "img1").is_dir() or not (p / "gt" / "gt.txt").is_file():
            continue
        if suffix and not p.name.endswith(f"-{suffix}"):
            continue
        out.append(p)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mot17-train-root",
        type=Path,
        required=True,
        help="Directory containing MOT17-xx-* sequence folders (…/MOT17/train)",
    )
    p.add_argument(
        "--det-subdir-suffix",
        default="FRCNN",
        help="Keep only folders ending with '-SUFFIX' (default FRCNN). "
        "Use empty string '' to include all variants (duplicates triple the dataset).",
    )
    p.add_argument(
        "--sequences",
        default="",
        help="Comma-separated folder names to include only (e.g. MOT17-02-FRCNN,MOT17-04-FRCNN)",
    )
    p.add_argument(
        "--mot-classes",
        default="1",
        help="Comma-separated MOT gt class ids to keep as positives (default: 1 = pedestrian)",
    )
    p.add_argument(
        "--min-visibility",
        type=float,
        default=0.0,
        help="Skip GT boxes with visibility below this (0 keeps all)",
    )
    p.add_argument("--out-json", type=Path, required=True)
    args = p.parse_args()

    train_root = args.mot17_train_root.expanduser().resolve()
    if not train_root.is_dir():
        raise SystemExit(f"Not a directory: {train_root}")

    suffix = args.det_subdir_suffix.strip()
    suffix_opt = suffix if suffix else None
    sequences = _list_sequences(train_root, suffix_opt)
    if args.sequences.strip():
        allow = {s.strip() for s in args.sequences.split(",") if s.strip()}
        sequences = [s for s in sequences if s.name in allow]
        missing = allow - {s.name for s in sequences}
        if missing:
            raise SystemExit(f"--sequences not found under train root: {sorted(missing)}")

    if not sequences:
        raise SystemExit(
            f"No MOT17 sequence dirs under {train_root} "
            f"(need img1/ + gt/gt.txt; suffix filter={suffix_opt!r})"
        )

    classes_allow = {int(x.strip()) for x in args.mot_classes.split(",") if x.strip()}

    if suffix_opt is None and len(sequences) > 12:
        print(
            "WARNING: including all DPM/FRCNN/SDP folders triples frames vs one detector tag. "
            "Prefer --det-subdir-suffix FRCNN.",
            flush=True,
        )

    categories = [{"id": 1, "name": "pedestrian"}]
    images: list[dict] = []
    annotations: list[dict] = []
    ann_id = 0
    global_img_id = 0

    # Map (sequence_name, frame) -> image id for annotations
    frame_to_imgid: dict[tuple[str, int], int] = {}

    for seq_dir in sequences:
        seq_name = seq_dir.name
        gt_path = seq_dir / "gt" / "gt.txt"
        img_dir = seq_dir / "img1"

        wh = _parse_seqinfo(seq_dir)

        jpgs = sorted(img_dir.glob("*.jpg"))
        if not jpgs:
            jpgs = sorted(img_dir.glob("*.png"))
        if not jpgs:
            raise SystemExit(f"No images under {img_dir}")

        for img_path in jpgs:
            frame = _frame_from_filename(img_path.stem)
            if frame is None:
                raise SystemExit(f"Bad frame filename {img_path.name}")
            global_img_id += 1
            if wh is None:
                try:
                    from PIL import Image

                    with Image.open(img_path) as im:
                        w, h = im.size
                except ImportError as e:
                    raise SystemExit(
                        "Need Pillow to read image sizes when seqinfo.ini is missing: pip install Pillow"
                    ) from e
            else:
                w, h = wh
            frame_sizes[frame] = (w, h)
            rel_name = f"{seq_name}/img1/{img_path.name}"
            images.append(
                {
                    "id": global_img_id,
                    "file_name": rel_name,
                    "width": w,
                    "height": h,
                    "mot_sequence": seq_name,
                    "mot_frame": frame,
                }
            )
            frame_to_imgid[(seq_name, frame)] = global_img_id

        # GT lines
        for raw in gt_path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            parts = [x.strip() for x in raw.split(",")]
            if len(parts) < 9:
                continue
            frame = int(parts[0])
            # track_id = int(parts[1])
            bb_left = float(parts[2])
            bb_top = float(parts[3])
            bb_w = float(parts[4])
            bb_h = float(parts[5])
            # conf_col = float(parts[6])
            cls = int(parts[7])
            vis = float(parts[8])
            if cls not in classes_allow:
                continue
            if vis < args.min_visibility:
                continue
            key = (seq_name, frame)
            iid = frame_to_imgid.get(key)
            if iid is None:
                continue
            ann_id += 1
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": iid,
                    "category_id": 1,
                    "bbox": [bb_left, bb_top, bb_w, bb_h],
                    "area": float(bb_w * bb_h),
                    "iscrowd": 0,
                }
            )

    coco = {
        "info": {
            "description": "MOT17 train converted to COCO bbox",
            "mot17_train_root": str(train_root),
            "det_subdir_suffix": suffix_opt,
        },
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(coco), encoding="utf-8")
    print(
        json.dumps(
            {
                "out_json": str(args.out_json),
                "sequences": len(sequences),
                "images": len(images),
                "annotations": len(annotations),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
