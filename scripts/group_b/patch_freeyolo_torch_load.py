#!/usr/bin/env python3
"""
FreeYOLO хранит чекпойнты со старым pickle (numpy.scalar и т.д.).
PyTorch 2.6+ делает torch.load(..., weights_only=True) по умолчанию → UnpicklingError.

Патчит utils/misc.py load_weight: weights_only=False (+ fallback без аргумента для torch<2.6).

Запускать после git clone / перед eval.py. Идемпотентен.
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
        print(f"skip: нет {misc}", file=sys.stderr)
        sys.exit(0)
    text = misc.read_text(encoding="utf-8")
    if "weights_only=False" in text and "path_to_ckpt" in text:
        print("patch: уже применён")
        return
    if OLD not in text:
        print("patch: паттерн torch.load не найден — проверьте версию FreeYOLO", file=sys.stderr)
        sys.exit(1)
    misc.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
    print(f"patch: обновлён {misc}")


if __name__ == "__main__":
    main()
