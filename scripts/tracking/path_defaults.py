#!/usr/bin/env python3
"""Shared path defaults for tracking scripts."""

from __future__ import annotations

import os
from pathlib import Path


def detect_work_root() -> Path:
    """Return base directory used for data/models on cloud boxes."""
    env = os.getenv("EDGE_WORK_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()

    for raw in ("/workspace", "/root/workspace", "/root"):
        p = Path(raw)
        if p.exists():
            return p.resolve()
    return Path("/root").resolve()


def default_models_dir() -> Path:
    return detect_work_root() / "models"


def default_data_dir() -> Path:
    return detect_work_root() / "data"
