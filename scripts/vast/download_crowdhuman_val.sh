#!/usr/bin/env bash
set -euo pipefail
# CrowdHuman validation split (~2.4 GB): archive + annotation_val.odgt.
# Default DATA_ROOT: /workspace/data/crowdhuman
# Override: DATA_ROOT=/path/to/crowdhuman bash scripts/vast/download_crowdhuman_val.sh

DATA_ROOT="${CROWDHUMAN_ROOT:-${DATA_ROOT:-/workspace/data/crowdhuman}}"
ZIP_URL="https://huggingface.co/datasets/sshao0516/CrowdHuman/resolve/main/CrowdHuman_val.zip"
ODGT_URL="https://huggingface.co/datasets/sshao0516/CrowdHuman/resolve/main/annotation_val.odgt"

mkdir -p "${DATA_ROOT}"
cd "${DATA_ROOT}"

echo "Downloading CrowdHuman val → ${DATA_ROOT}"

if [[ ! -f CrowdHuman_val.zip ]]; then
  wget -O CrowdHuman_val.zip "${ZIP_URL}"
else
  echo "CrowdHuman_val.zip already exists, skipping wget"
fi

if [[ ! -f annotation_val.odgt ]]; then
  wget -O annotation_val.odgt "${ODGT_URL}"
else
  echo "annotation_val.odgt already exists, skipping wget"
fi

if [[ ! -d Images ]] || [[ -z "$(ls -A Images 2>/dev/null || true)" ]]; then
  unzip -q CrowdHuman_val.zip
else
  echo "Images/ already populated, skipping unzip"
fi

rm -f CrowdHuman_val.zip

echo "Done. Expected: ${DATA_ROOT}/Images/ and ${DATA_ROOT}/annotation_val.odgt"
