#!/usr/bin/env python3
"""
FreeYOLO использует np.int / np.float / np.bool и т.д.; в NumPy 2.x эти алиасы удалены.

Заменяем только целые токены (не трогаем np.int32, np.float64, …). Идемпотентно для повторных запусков.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# порядок важен: более длинные имена не нужны — границы слова отсекают np.int64 и т.п.
SUBSTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bnp\.bool\b"), "bool"),
    (re.compile(r"\bnp\.long\b"), "int"),
    (re.compile(r"\bnp\.unicode\b"), "str"),
    (re.compile(r"\bnp\.complex\b"), "complex"),
    (re.compile(r"\bnp\.object\b"), "object"),
    (re.compile(r"\bnp\.float\b"), "float"),
    (re.compile(r"\bnp\.int\b"), "int"),
]


def patch_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    orig = text
    for pat, repl in SUBSTS:
        text = pat.sub(repl, text)
    if text != orig:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--freeyolo-home", type=Path, required=True)
    args = p.parse_args()
    root: Path = args.freeyolo_home
    if not root.is_dir():
        print(f"skip: нет каталога {root}", file=sys.stderr)
        sys.exit(0)

    n = 0
    for py in root.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        if patch_file(py):
            print(f"patch_numpy_aliases: {py.relative_to(root)}")
            n += 1
    print(f"patch_numpy_aliases: обновлено файлов: {n}")


if __name__ == "__main__":
    main()
