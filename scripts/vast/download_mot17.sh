#!/usr/bin/env bash
set -euo pipefail
# Download MOT17 via kagglehub (public Kaggle dataset; API key usually not required).
# Result: DATA_ROOT/MOT17/train and DATA_ROOT/MOT17/test
#
# Usage:
#   bash scripts/vast/download_mot17.sh
# Override destination:
#   DATA_ROOT=/mnt/data/mot17 bash scripts/vast/download_mot17.sh

DATA_ROOT="${MOT17_ROOT:-${DATA_ROOT:-/workspace/data/mot17}}"

mkdir -p "${DATA_ROOT}"

if python3 -c "import kagglehub" 2>/dev/null; then
  :
else
  echo "Installing kagglehub..."
  python3 -m pip install --quiet kagglehub
fi

if [[ -d "${DATA_ROOT}/MOT17/train" ]]; then
  echo "Already present: ${DATA_ROOT}/MOT17/train — skipping copy."
  ls "${DATA_ROOT}/MOT17/train/"
  exit 0
fi

# kagglehub/tqdm often bypass contextlib.redirect_stdout — do not use MOT17_SRC=$(python …).
MOT17_SRC_FILE="$(mktemp)"
export MOT17_SRC_FILE
python3 <<'PY'
import os
import sys
from pathlib import Path

import kagglehub

cache_root = kagglehub.dataset_download("wenhoujinjust/mot-17")
base = Path(cache_root).resolve()
print(f"Kagglehub cache path: {base}", file=sys.stderr)

chosen = None
for candidate in sorted(base.rglob("MOT17"), key=lambda p: len(p.parts)):
    if candidate.is_dir() and (candidate / "train").is_dir():
        chosen = candidate.resolve()
        break

if chosen is None:
    print("Could not find MOT17/train under downloaded tree.", file=sys.stderr)
    sys.exit(1)

out = os.environ["MOT17_SRC_FILE"]
Path(out).write_text(str(chosen) + "\n", encoding="utf-8")
print(f"MOT17 source dir (written to {out}): {chosen}", file=sys.stderr)
PY

MOT17_SRC="$(tr -d '\r\n' <"${MOT17_SRC_FILE}")"
rm -f "${MOT17_SRC_FILE}"
if [[ ! -d "${MOT17_SRC}" ]]; then
  echo "Resolved MOT17 path missing: ${MOT17_SRC}" >&2
  exit 1
fi

echo "Copying MOT17 → ${DATA_ROOT}/"
cp -r "${MOT17_SRC}" "${DATA_ROOT}/"

echo "Train sequences:"
ls "${DATA_ROOT}/MOT17/train/"
