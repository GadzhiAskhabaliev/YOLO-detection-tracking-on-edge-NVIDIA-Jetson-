#!/usr/bin/env python3
"""
FreeYOLO GitHub release `yolo_free_tiny_ch.pth` matches `head_depthwise: True` in the head,
while current upstream `config/yolo_free_config.py` sets `head_depthwise: False` for
`yolo_free_tiny` → torch.load / load_state_dict fails with reg_feats conv shape mismatch.

This script sets `head_depthwise` to True **only** inside the `yolo_free_tiny` config block
(idempotent if already True).

  python3 scripts/group_b/patch_freeyolo_tiny_ckpt_compat.py --freeyolo-home /path/to/FreeYOLO
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--freeyolo-home", type=Path, required=True)
    args = p.parse_args()

    cfg_path = args.freeyolo_home.expanduser().resolve() / "config" / "yolo_free_config.py"
    if not cfg_path.is_file():
        raise SystemExit(f"Missing {cfg_path}")

    text = cfg_path.read_text(encoding="utf-8")
    m = re.search(r"('yolo_free_tiny'\s*:\s*\{)(.*?)(^\s*'yolo_free_large'\s*:)", text, re.S | re.M)
    if not m:
        raise SystemExit("Could not locate yolo_free_tiny block in yolo_free_config.py")

    block = m.group(2)
    if re.search(r"'head_depthwise'\s*:\s*True", block):
        print(f"Already patched or native True: {cfg_path}")
        return

    new_block, n = re.subn(
        r"('head_depthwise'\s*:\s*)False(\s*,)",
        r"\1True\2",
        block,
        count=1,
    )
    if n != 1:
        raise SystemExit(
            f"Expected one head_depthwise False in yolo_free_tiny block, replaced {n}. "
            "Check FreeYOLO version."
        )

    new_text = text[: m.start(2)] + new_block + text[m.start(3) :]
    cfg_path.write_text(new_text, encoding="utf-8")
    print(f"Patched yolo_free_tiny head_depthwise → True: {cfg_path}")


if __name__ == "__main__":
    main()
