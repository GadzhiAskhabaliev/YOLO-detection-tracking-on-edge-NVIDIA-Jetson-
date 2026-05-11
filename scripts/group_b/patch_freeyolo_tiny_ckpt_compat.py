#!/usr/bin/env python3
"""
Align FreeYOLO `yolo_free_tiny` config with the **official** GitHub
`yolo_free_tiny_ch.pth` (~49.5 MB): upstream uses `fpn_depthwise: False` and
`head_depthwise: False` (reg head convs are [64, 64, 3, 3], not depthwise [64, 1, 3, 3]).

An earlier bench revision flipped both to True after a **wrong** download: the
nano checkpoint (~16.2 MB) was saved as `yolo_free_tiny_ch.pth`, which confused
diagnostics. If your FreeYOLO config was patched that way, this script **reverts**
`fpn_depthwise` and `head_depthwise` to **False** inside the `yolo_free_tiny` block.

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
    if re.search(r"'fpn_depthwise'\s*:\s*False", block) and re.search(
        r"'head_depthwise'\s*:\s*False", block
    ):
        print(f"Already upstream (tiny fpn_depthwise/head_depthwise False): {cfg_path}")
        return

    changes: list[str] = []
    if re.search(r"'fpn_depthwise'\s*:\s*True", block):
        block, n = re.subn(
            r"('fpn_depthwise'\s*:\s*)True(\s*,)",
            r"\1False\2",
            block,
            count=1,
        )
        if n != 1:
            raise SystemExit(
                f"Expected one fpn_depthwise True in yolo_free_tiny block, replaced {n}."
            )
        changes.append("fpn_depthwise")

    if re.search(r"'head_depthwise'\s*:\s*True", block):
        block, n = re.subn(
            r"('head_depthwise'\s*:\s*)True(\s*,)",
            r"\1False\2",
            block,
            count=1,
        )
        if n != 1:
            raise SystemExit(
                f"Expected one head_depthwise True in yolo_free_tiny block, replaced {n}."
            )
        changes.append("head_depthwise")

    if not changes:
        print(f"No mistaken True flags to revert in tiny block: {cfg_path}")
        return

    new_text = text[: m.start(2)] + block + text[m.start(3) :]
    cfg_path.write_text(new_text, encoding="utf-8")
    print(f"Reverted yolo_free_tiny ({', '.join(changes)} → False): {cfg_path}")


if __name__ == "__main__":
    main()
