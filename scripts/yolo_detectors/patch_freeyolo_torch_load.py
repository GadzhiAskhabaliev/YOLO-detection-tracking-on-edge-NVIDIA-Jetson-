#!/usr/bin/env python3
"""
FreeYOLO checkpoints use legacy pickle (numpy.scalar, etc.).
PyTorch 2.6+ defaults torch.load(..., weights_only=True) → UnpicklingError.

Patches utils/misc.py load_weight to pass weights_only=False (fallback for torch<2.6).

Run after clone / before eval.py. Idempotent.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

OLD = "    checkpoint = torch.load(path_to_ckpt, map_location='cpu')"
NEW = """    try:
        checkpoint = torch.load(path_to_ckpt, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(path_to_ckpt, map_location='cpu')"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--freeyolo-home", type=Path, required=True)
    args = p.parse_args()
    misc = args.freeyolo_home / "utils" / "misc.py"
    if not misc.is_file():
        print(f"skip: missing {misc}", file=sys.stderr)
        sys.exit(0)
    text = misc.read_text(encoding="utf-8")
    if "weights_only=False" in text and "path_to_ckpt" in text:
        print("patch: already applied")
        return
    if OLD not in text:
        print("patch: torch.load pattern not found — check FreeYOLO version", file=sys.stderr)
        sys.exit(1)
    misc.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
    print(f"patch: updated {misc}")


if __name__ == "__main__":
    main()
