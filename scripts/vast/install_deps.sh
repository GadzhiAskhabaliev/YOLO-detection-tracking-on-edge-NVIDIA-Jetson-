#!/usr/bin/env bash
set -euo pipefail
# Fresh Ubuntu/Linux on Vast (no NVIDIA container image): install CLI deps + CUDA PyTorch + project pip packages.
#
# Prerequisites on the instance:
#   - GPU passthrough + NVIDIA driver working (`nvidia-smi` must run).
#   - This script does NOT install kernel drivers (do that via host/Vast UI or Ubuntu if needed).
#
# PyTorch wheels use NVIDIA's CUDA runtime bundled inside the wheel; pick an index that matches
# your driver (see https://pytorch.org/get-started/locally/).
#
# Defaults:
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124   (CUDA 12.4 wheel bundle)
# Override examples:
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 bash scripts/vast/install_deps.sh
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu bash scripts/vast/install_deps.sh   # no GPU
#
# Skip reinstalling torch if already satisfied:
#   SKIP_TORCH=1 bash scripts/vast/install_deps.sh

TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"

apt-get update
apt-get install -y \
  python3 \
  python3-pip \
  python3-venv \
  wget \
  unzip \
  git \
  tmux \
  ca-certificates \
  curl

python3 -m pip install --upgrade pip wheel setuptools

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU driver report:"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
else
  echo "WARNING: nvidia-smi not found — GPU drivers may be missing; torch.cuda.is_available() will likely be False." >&2
fi

if [[ "${SKIP_TORCH:-0}" != "1" ]]; then
  echo "Installing torch + torchvision from ${TORCH_INDEX_URL}"
  python3 -m pip install --upgrade torch torchvision --index-url "${TORCH_INDEX_URL}"
fi

python3 -m pip install ultralytics kagglehub opencv-python-headless tqdm

echo "--- Sanity check ---"
python3 <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

python3 -c "import ultralytics, kagglehub, cv2; print('ultralytics, kagglehub, cv2: OK')"
echo "Done."
