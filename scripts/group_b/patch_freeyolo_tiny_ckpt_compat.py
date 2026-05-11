#!/usr/bin/env python3
"""
FreeYOLO release `yolo_free_tiny_ch.pth` was trained with depthwise FPN convs and a
depthwise decoupled head. Upstream `yolo_free_config.py` sets both to False for tiny,
which breaks `load_state_dict`:
  - head: reg/cls branch weight shapes (standard vs depthwise Conv)
  - FPN: e.g. `fpn.head_conv_3.convs.1` BN is c_in (256) when depthwise=True, else c_out (512)

This script sets `fpn_depthwise` and `head_depthwise` to True inside the `yolo_free_tiny`
block only (idempotent if already True).

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
    if re.search(r"'head_depthwise'\s*:\s*True", block) and re.search(
        r"'fpn_depthwise'\s*:\s*True", block
    ):
        print(f"Already patched or native True: {cfg_path}")
        return

    changes: list[str] = []
    if re.search(r"'fpn_depthwise'\s*:\s*False", block):
        block, n = re.subn(
            r"('fpn_depthwise'\s*:\s*)False(\s*,)",
            r"\1True\2",
            block,
            count=1,
        )
        if n != 1:
            raise SystemExit(
                f"Expected one fpn_depthwise False in yolo_free_tiny block, replaced {n}."
            )
        changes.append("fpn_depthwise")

    if re.search(r"'head_depthwise'\s*:\s*False", block):
        block, n = re.subn(
            r"('head_depthwise'\s*:\s*)False(\s*,)",
            r"\1True\2",
            block,
            count=1,
        )
        if n != 1:
            raise SystemExit(
                f"Expected one head_depthwise False in yolo_free_tiny block, replaced {n}."
            )
        changes.append("head_depthwise")

    if not changes:
        print(f"No tiny depthwise flags to patch: {cfg_path}")
        return

    new_text = text[: m.start(2)] + block + text[m.start(3) :]
    cfg_path.write_text(new_text, encoding="utf-8")
    print(f"Patched yolo_free_tiny ({', '.join(changes)} → True): {cfg_path}")


if __name__ == "__main__":
    main()
