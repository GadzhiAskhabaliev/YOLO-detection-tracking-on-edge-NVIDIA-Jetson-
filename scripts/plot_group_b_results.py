#!/usr/bin/env python3
"""
Строит графики и markdown-отчёт для группы B по docs/group_b_pedestrian_detectors.yaml
и последним прогонам в results/runs/*.json.

Требования: pip install matplotlib pyyaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "results" / "runs"
MANIFEST_PATH = REPO_ROOT / "docs" / "group_b_pedestrian_detectors.yaml"
FIG_DIR = REPO_ROOT / "results" / "figures"
OUT_MD = REPO_ROOT / "results" / "group_b_report.md"


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as e:
        raise SystemExit("Установите PyYAML: pip install pyyaml") from e
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_runs() -> list[dict[str, Any]]:
    if not RUNS_DIR.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for p in sorted(RUNS_DIR.glob("*.json")):
        try:
            rows.append(json.loads(p.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return rows


def _pick_latest_for_detector(runs: list[dict[str, Any]], group: str, det: dict[str, Any]) -> dict[str, Any] | None:
    did = det["id"]
    slug = det["bench_slug"]
    cand = []
    for r in runs:
        if r.get("model") == slug:
            cand.append(r)
            continue
        if str(r.get("group", "")).upper() == group.upper() and r.get("detector_id") == did:
            cand.append(r)
    if not cand:
        return None

    def dt(r: dict[str, Any]) -> str:
        return str(r.get("date_updated") or r.get("date") or "")

    cand.sort(key=dt, reverse=True)
    return cand[0]


def _fps_metric(run: dict[str, Any] | None) -> float | None:
    if not run:
        return None
    m = run.get("metrics") or {}
    for k in ("fps_predict", "fps_forward"):
        v = m.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _map50(run: dict[str, Any] | None) -> float | None:
    if not run:
        return None
    m = run.get("metrics") or {}
    v = m.get("mAP50")
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _plot_scatter(
    series: list[tuple[str, float, float]],
    out_path: Path,
    *,
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit("Установите matplotlib: pip install matplotlib") from e

    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    if not series:
        ax.text(0.5, 0.5, "Нет точек с mAP50 и FPS", ha="center", va="center")
        ax.set_axis_off()
    else:
        xs = [t[1] for t in series]
        ys = [t[2] for t in series]
        ax.scatter(xs, ys, s=80, alpha=0.85)
        for name, x, y in series:
            ax.annotate(name, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
        ax.set_xlabel("FPS (predict или forward)")
        ax.set_ylabel("mAP50")
        ax.grid(True, alpha=0.3)
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_bars(labels: list[str], values: list[float | None], out_path: Path, *, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:
        raise SystemExit("Установите matplotlib: pip install matplotlib") from e

    fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
    y = np.arange(len(labels))
    vals = [v if v is not None else 0.0 for v in values]
    colors = ["#4C72B0" if v is not None else "#CCCCCC" for v in values]
    ax.barh(y, vals, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("mAP50")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def regenerate(*, manifest_path: Path | None = None, out_md: Path | None = None) -> Path:
    mf_path = manifest_path or MANIFEST_PATH
    manifest = _load_yaml(mf_path)
    group = str(manifest.get("group", "B"))
    detectors: list[dict[str, Any]] = manifest["detectors"]
    runs = _load_runs()

    rows_md: list[str] = []
    rows_md.append(f"# Группа {group}: {manifest.get('title', '')}\n\n")
    rows_md.append("Автогенерация: `scripts/plot_group_b_results.py`\n\n")
    rows_md.append("| № | Модель | mAP50 | FPS | Прогон |\n")
    rows_md.append("|---|--------|-------|-----|--------|\n")

    scatter_series: list[tuple[str, float, float]] = []
    bar_labels: list[str] = []
    bar_vals: list[float | None] = []

    for det in sorted(detectors, key=lambda d: d["id"]):
        run = _pick_latest_for_detector(runs, group, det)
        label = det.get("short_name", det["bench_slug"])
        m50 = _map50(run)
        fps = _fps_metric(run)
        status = "да" if run else "**нет**"
        rows_md.append(
            f"| {det['id']} | {label} | "
            f"{m50 if m50 is not None else '—'} | "
            f"{fps if fps is not None else '—'} | {status} |\n"
        )
        bar_labels.append(f"{det['id']}: {label}")
        bar_vals.append(m50)
        if m50 is not None and fps is not None:
            scatter_series.append((label, fps, m50))

    scatter_path = FIG_DIR / "group_b_scatter_map_fps.png"
    bars_path = FIG_DIR / "group_b_map50_bars.png"
    _plot_scatter(
        scatter_series,
        scatter_path,
        title=f"Группа {group}: mAP50 vs FPS",
    )
    _plot_bars(
        bar_labels,
        bar_vals,
        bars_path,
        title=f"Группа {group}: mAP50 (серые — нет данных)",
    )

    rows_md.append("\n## Графики\n\n")
    rows_md.append(f"![scatter](figures/group_b_scatter_map_fps.png)\n\n")
    rows_md.append(f"![bars](figures/group_b_map50_bars.png)\n\n")

    out = out_md or OUT_MD
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(rows_md), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--out-md", type=Path, default=None)
    args = p.parse_args(argv)
    path = regenerate(manifest_path=args.manifest, out_md=args.out_md)
    print(path.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main(sys.argv[1:])
