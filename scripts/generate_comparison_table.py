#!/usr/bin/env python3
"""
Aggregate results/runs/*.json into results/model_comparison.md.

Produces a Markdown table (latest run per model) and a coarse ASCII scatter:
AP50 (vertical) vs FPS forward (horizontal).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "results" / "runs"
OUT_MD = REPO_ROOT / "results" / "model_comparison.md"


def _load_runs() -> list[tuple[Path, dict[str, Any]]]:
    if not RUNS_DIR.is_dir():
        return []
    rows: list[tuple[Path, dict[str, Any]]] = []
    for p in sorted(RUNS_DIR.glob("*.json")):
        try:
            rows.append((p, json.loads(p.read_text(encoding="utf-8"))))
        except json.JSONDecodeError:
            continue
    return rows


def _latest_per_model(rows: list[tuple[Path, dict[str, Any]]]) -> list[dict[str, Any]]:
    best: dict[str, tuple[str, dict[str, Any]]] = {}
    for _, data in rows:
        name = str(data.get("model", ""))
        dt = str(data.get("date", ""))
        cur = best.get(name)
        if cur is None or dt > cur[0]:
            best[name] = (dt, data)
    return [v[1] for v in best.values()]


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4f}".rstrip("0").rstrip(".")
    return str(v)


def ascii_scatter(
    points: list[tuple[float, float, str]],
    *,
    width: int = 56,
    height: int = 14,
) -> str:
    """FPS on X, AP50 on Y (upper rows = higher AP50)."""
    if not points:
        return "(no points — run benchmarks first)\n"

    fps_vals = [p[0] for p in points]
    map_vals = [p[1] for p in points]
    min_fps, max_fps = min(fps_vals), max(fps_vals)
    min_m, max_m = min(map_vals), max(map_vals)
    if max_fps <= min_fps:
        max_fps = min_fps + 1e-6
    if max_m <= min_m:
        max_m = min_m + 1e-6

    grid = [[" " for _ in range(width)] for _ in range(height)]

    def clamp_ix(x: float) -> int:
        return int(round((x - min_fps) / (max_fps - min_fps) * (width - 1)))

    def clamp_iy(y: float) -> int:
        # invert so top is high AP50
        return int(round((max_m - y) / (max_m - min_m) * (height - 1)))

    labels: list[str] = []
    for i, (fps, m50, label) in enumerate(sorted(points, key=lambda t: (-t[1], -t[0]))):
        ix = clamp_ix(fps)
        iy = clamp_iy(m50)
        ch = chr(ord("A") + (i % 26))
        if grid[iy][ix] == " ":
            grid[iy][ix] = ch
        else:
            grid[iy][ix] = "*"
        labels.append(f"  {ch}: {label} — AP50 {_fmt(m50)}, FPS fwd {_fmt(fps)}")

    lines = []
    lines.append(f"FPS forward: {_fmt(min_fps)} … {_fmt(max_fps)}  |  AP50: {_fmt(min_m)} … {_fmt(max_m)}\n")
    lines.append("```")
    for row in grid:
        lines.append("|" + "".join(row) + "|")
    lines.append("```")
    lines.append("\nLegend:\n" + "\n".join(labels) + "\n")
    return "\n".join(lines)


def regenerate(out_path: Path | None = None) -> Path:
    path = out_path or OUT_MD
    rows = _load_runs()
    latest = _latest_per_model(rows)

    def sort_key(d: dict[str, Any]) -> float:
        m = d.get("metrics") or {}
        v = m.get("AP50")
        if v is None:
            v = m.get("mAP50")
        return float(v or 0.0)

    latest.sort(key=sort_key, reverse=True)

    parts: list[str] = []
    parts.append("# Model comparison (auto-generated)\n\n")
    parts.append("Source: latest run per model from `results/runs/*.json`.\n\n")

    parts.append("## Table\n\n")
    hdr = "| Backend | Model | AP50 | FPS forward | FPS predict | MOTA | Date |\n"
    sep = "|---------|-------|-------|-------------|-------------|------|------|\n"
    parts.append(hdr + sep)
    scatter_pts: list[tuple[float, float, str]] = []

    for d in latest:
        met = d.get("metrics") or {}
        tr = d.get("tracking") or {}
        bk = d.get("backend") or d.get("framework") or ""
        parts.append(
            "| "
            + " | ".join(
                [
                    str(bk).strip(),
                    str(d.get("model", "")),
                    _fmt(met.get("AP50") if met.get("AP50") is not None else met.get("mAP50")),
                    _fmt(met.get("fps_forward")),
                    _fmt(met.get("fps_predict")),
                    _fmt(tr.get("MOTA")),
                    str(d.get("date", "")),
                ]
            )
            + " |\n"
        )
        fps = met.get("fps_forward")
        m50 = met.get("AP50") if met.get("AP50") is not None else met.get("mAP50")
        if isinstance(fps, (int, float)) and isinstance(m50, (int, float)):
            scatter_pts.append((float(fps), float(m50), str(d.get("model", ""))))

    parts.append("\n## AP50 vs FPS (ASCII)\n\n")
    parts.append(ascii_scatter(scatter_pts))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(parts), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=None, help="Output markdown path")
    args = p.parse_args(argv)
    out = regenerate(args.out)
    print(out.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main(sys.argv[1:])
