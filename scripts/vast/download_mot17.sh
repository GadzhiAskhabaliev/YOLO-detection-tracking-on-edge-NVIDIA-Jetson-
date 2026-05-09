#!/usr/bin/env bash
set -euo pipefail
# Download MOT17 via kagglehub (public Kaggle dataset; API key usually not required).
# Result: DATA_ROOT/MOT17/train and DATA_ROOT/MOT17/test
#
# Usage:
#   bash scripts/vast/download_mot17.sh
# Override destination:
#   DATA_ROOT=/mnt/data/mot17 bash scripts/vast/download_mot17.sh

DATA_ROOT="${DATA_ROOT:-/workspace/data/mot17}"

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

MOT17_SRC="$(python3 <<'PY'
import sys
from pathlib import Path

import kagglehub

base = Path(kagglehub.dataset_download("wenhoujinjust/mot-17"))
print(f"Kagglehub cache path: {base}", file=sys.stderr)

chosen = None
for candidate in sorted(base.rglob("MOT17"), key=lambda p: len(p.parts)):
    if candidate.is_dir() and (candidate / "train").is_dir():
        chosen = candidate.resolve()
        break

if chosen is None:
    print("Could not find MOT17/train under downloaded tree.", file=sys.stderr)
    sys.exit(1)

print(chosen)
PY
)"

echo "Copying MOT17 → ${DATA_ROOT}/"
cp -r "${MOT17_SRC}" "${DATA_ROOT}/"

echo "Train sequences:"
ls "${DATA_ROOT}/MOT17/train/"
