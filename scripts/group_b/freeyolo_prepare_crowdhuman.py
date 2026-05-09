#!/usr/bin/env python3
"""
Собирает дерево, которое ожидают FreeYOLO eval.py и CrowdHumanDataset:
  BRIDGE/CrowdHuman/annotations/val.json
  BRIDGE/CrowdHuman/CrowdHuman_val/Images -> CROWDHUMAN_ROOT/Images

Источник — типичный Vast layout после download_crowdhuman_val.sh:
  CROWDHUMAN_ROOT/Images/*.jpg
  CROWDHUMAN_ROOT/annotation_val.odgt
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def load_odgt(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--crowdhuman-root",
        default=os.environ.get("CROWDHUMAN_ROOT", "/workspace/data/crowdhuman"),
        help="Где лежат Images/ и annotation_val.odgt",
    )
    p.add_argument(
        "--bridge-root",
        default=os.environ.get("FREEYOLO_CH_BRIDGE", "/workspace/group_b/freeyolo_crowdhuman_bridge"),
        help="Каталог-родитель для поддерева CrowdHuman/ (его передают как --root в eval.py)",
    )
    args = p.parse_args()

    ch = Path(args.crowdhuman_root).resolve()
    bridge_parent = Path(args.bridge_root).resolve()
    crowd_dir = bridge_parent / "CrowdHuman"
    ann_dir = crowd_dir / "annotations"
    val_img_link = crowd_dir / "CrowdHuman_val" / "Images"
    odgt = ch / "annotation_val.odgt"
    images_src = ch / "Images"

    if not odgt.is_file():
        raise SystemExit(f"Нет {odgt}")
    if not images_src.is_dir():
        raise SystemExit(f"Нет каталога {images_src}")

    ann_dir.mkdir(parents=True, exist_ok=True)
    val_img_link.parent.mkdir(parents=True, exist_ok=True)
    if val_img_link.is_symlink() or val_img_link.exists():
        val_img_link.unlink()
    val_img_link.symlink_to(images_src.resolve(), target_is_directory=True)

    records = load_odgt(odgt)
    out = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "person"}]}
    image_cnt = 0
    ann_cnt = 0

    for ann_data in records:
        image_cnt += 1
        img_id = ann_data["ID"]
        img_path = images_src / f"{img_id}.jpg"
        if not img_path.is_file():
            raise SystemExit(f"Нет изображения {img_path}")

        from PIL import Image

        with Image.open(img_path) as im:
            w, h = im.size

        out["images"].append(
            {
                "file_name": f"{img_id}.jpg",
                "id": image_cnt,
                "height": h,
                "width": w,
            }
        )

        if "gtboxes" not in ann_data:
            continue
        for box_entry in ann_data["gtboxes"]:
            ann_cnt += 1
            fbox = box_entry["fbox"]
            ignore = (
                "extra" in box_entry
                and "ignore" in box_entry["extra"]
                and box_entry["extra"]["ignore"] == 1
            )
            ann = {
                "id": ann_cnt,
                "category_id": 1,
                "image_id": image_cnt,
                "track_id": -1,
                "bbox_vis": box_entry.get("vbox", []),
                "bbox": fbox,
                "area": float(fbox[2]) * float(fbox[3]),
                "iscrowd": 1 if ignore else 0,
            }
            out["annotations"].append(ann)

    val_json = ann_dir / "val.json"
    val_json.write_text(json.dumps(out), encoding="utf-8")
    print(json.dumps({"bridge_root": str(bridge_parent), "val_json": str(val_json), "images": len(out["images"]), "anns": len(out["annotations"])}, indent=2))


if __name__ == "__main__":
    main()
