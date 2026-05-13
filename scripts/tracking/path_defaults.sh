#!/usr/bin/env bash
set -euo pipefail

# Shared path defaults for tracking shell wrappers.
#
# Priority:
# 1) EDGE_WORK_ROOT env
# 2) Existing /workspace
# 3) Existing /root/workspace
# 4) /root

if [[ -n "${EDGE_WORK_ROOT:-}" ]]; then
  TRACKING_WORK_ROOT="${EDGE_WORK_ROOT}"
elif [[ -d "/workspace" ]]; then
  TRACKING_WORK_ROOT="/workspace"
elif [[ -d "/root/workspace" ]]; then
  TRACKING_WORK_ROOT="/root/workspace"
else
  TRACKING_WORK_ROOT="/root"
fi

TRACKING_DATA_ROOT="${TRACKING_DATA_ROOT:-${TRACKING_WORK_ROOT}/data}"
TRACKING_MODEL_ROOT="${TRACKING_MODEL_ROOT:-${TRACKING_WORK_ROOT}/models}"
