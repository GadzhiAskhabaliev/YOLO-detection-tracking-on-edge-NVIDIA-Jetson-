#!/usr/bin/env python3
"""
Convert CrowdHuman .odgt (one JSON object per line) to YOLO label .txt files.

Skips gtboxes with tag != person, extra.ignore, or missing box.
Coordinates use actual image size from disk.

Example:
  cd /workspace/data/crowdhuman
  python3 scripts/vast/crowdhuman_odgt_to_yolo.py \\
    --odgt annotation_val.odgt --images-dir Images --out-dir labels_val

Debug mismatch HF mirror vs filenames:
  python3 scripts/vast/crowdhuman_odgt_to_yolo.py ... --peek
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

_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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
    p.add_argument(
        "--peek",
        action="store_true",
        help="Print first ODGT record + sample image names, then exit",
    )
    return p.parse_args()


def collect_images(images_dir: Path) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    """Lower-name -> path; suffix-after-comma -> list of paths (HF naming quirks)."""
    by_lower: dict[str, Path] = {}
    by_tail: dict[str, list[Path]] = {}
    for p in images_dir.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in _IMG_EXT:
            continue
        low = p.name.lower()
        by_lower.setdefault(low, p)
        if "," in low:
            tail = low.split(",", 1)[1]
            by_tail.setdefault(tail, []).append(p)
        elif "_" in low:
            tail_u = low.split("_", 1)[1]
            by_tail.setdefault(tail_u, []).append(p)
    return by_lower, by_tail


def basename_variants(fname: str) -> list[str]:
    """Try several spellings seen across CrowdHuman / HF mirrors."""
    fname = fname.strip()
    out: list[str] = []
    base = Path(fname.replace("\\", "/")).name
    if not base:
        return out

    def add(s: str):
        if s and s not in out:
            out.append(s)

    add(base)
    add(base.lower())
    stem = Path(base).stem
    ext = Path(base).suffix if Path(base).suffix else ".jpg"

    add(stem + ext)
    add(stem.lower() + ext.lower())

    add(base.replace(",", "_"))
    add(base.replace("_", ",", 1))

    if "," in base:
        add(base.split(",", 1)[-1])
    if "_" in base and "," not in base:
        parts = base.split("_", 1)
        if len(parts) == 2:
            add(f"{parts[0]},{parts[1]}")

    return out


def resolve_image(
    raw: dict,
    images_dir: Path,
    by_lower: dict[str, Path],
    by_tail: dict[str, list[Path]],
) -> Path | None:
    refs: list[str] = []
    for key in ("ID", "img_path", "name", "file_name"):
        v = raw.get(key)
        if v is not None and str(v).strip():
            refs.append(str(v).strip())

    candidates: list[str] = []
    for ref in refs:
        ref_norm = ref.replace("\\", "/")
        candidates.extend(basename_variants(Path(ref_norm).name))

    for c in candidates:
        hit = by_lower.get(c.lower())
        if hit is not None:
            return hit
        direct = images_dir / c
        if direct.is_file():
            return direct

    for c in candidates:
        low = c.lower()
        if "," in low:
            tail = low.split(",", 1)[1]
            hits = by_tail.get(tail)
            if hits and len(hits) == 1:
                return hits[0]

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

    if not args.images_dir.is_dir():
        raise SystemExit(f"images-dir not found: {args.images_dir}")

    text = args.odgt.read_text(encoding="utf-8-sig")
    odgt_lines = text.splitlines()

    by_lower, by_tail = collect_images(args.images_dir)
    if not by_lower:
        raise SystemExit(f"No images under {args.images_dir} (extensions {_IMG_EXT})")

    if args.peek:
        first = next((ln for ln in odgt_lines if ln.strip()), "")
        print("First ODGT line keys:", list(json.loads(first).keys()) if first else "EMPTY")
        sample = sorted(by_lower.values(), key=lambda p: p.name.lower())[:8]
        print("Sample files on disk:", [p.name for p in sample])
        print(json.dumps(json.loads(first), indent=2)[:2000])
        return

    skipped_no_img = 0
    bad_imread = 0

    for line in tqdm(odgt_lines, desc=args.odgt.name):
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)

        img_path = resolve_image(raw, args.images_dir, by_lower, by_tail)
        if img_path is None:
            skipped_no_img += 1
            continue

        stem = img_path.stem
        im = cv2.imread(str(img_path))
        if im is None:
            bad_imread += 1
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
        print(
            f"Note: no image match for {skipped_no_img} ODGT lines "
            f"(run with --peek if all skipped)."
        )
    if bad_imread:
        print(f"Note: OpenCV failed to read {bad_imread} matched paths.")
    print(f"Labels written under {args.out_dir.resolve()}")
    print(f"Label files: {len(list(args.out_dir.glob('*.txt')))}")


if __name__ == "__main__":
    main()
