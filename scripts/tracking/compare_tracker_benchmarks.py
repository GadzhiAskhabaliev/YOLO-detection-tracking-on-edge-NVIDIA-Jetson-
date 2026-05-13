#!/usr/bin/env python3
"""Build unified comparison across multiple tracker benchmark JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def norm(value: float, low: float, high: float) -> float:
    if high <= low:
        return 1.0
    return (value - low) / (high - low)


def load_benchmarks(results_dir: Path, trackers: list[str]) -> list[dict[str, Any]]:
    benches: list[dict[str, Any]] = []
    for tracker in trackers:
        bench_path = results_dir / f"yolov8_{tracker}_mot17_benchmark.json"
        if not bench_path.is_file():
            raise SystemExit(f"Missing benchmark JSON: {bench_path}")
        payload = json.loads(bench_path.read_text(encoding="utf-8"))
        rows = payload.get("rows", [])
        if not rows:
            raise SystemExit(f"No rows in benchmark: {bench_path}")
        benches.append(
            {
                "tracker": tracker,
                "tracker_name": payload.get("tracker_name", f"yolov8_{tracker}"),
                "path": str(bench_path),
                "rows": rows,
            }
        )
    return benches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/tracking")
    parser.add_argument(
        "--trackers",
        default="bytetrack,strongsort,botsort,hybridsort,deepocsort",
        help="Comma-separated tracker ids matching yolov8_<id>_mot17_benchmark.json",
    )
    parser.add_argument("--out-json", default="results/tracking/tracker_comparison.json")
    parser.add_argument("--out-md", default="results/tracking/tracker_comparison.md")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    results_dir = (repo_root / args.results_dir).resolve()
    trackers = [x.strip().lower() for x in args.trackers.split(",") if x.strip()]

    benches = load_benchmarks(results_dir, trackers)
    all_rows: list[dict[str, Any]] = []
    for bench in benches:
        for row in bench["rows"]:
            all_rows.append(
                {
                    "tracker": bench["tracker"],
                    "tracker_name": bench["tracker_name"],
                    "config": row.get("config", {}),
                    "fps": to_float(row.get("fps_e2e")),
                    "mota": to_float(row.get("MOTA")),
                    "idf1": to_float(row.get("IDF1")),
                    "hota": to_float(row.get("HOTA")),
                    "run_report": row.get("run_report", ""),
                    "trackeval_json": row.get("trackeval_json", ""),
                }
            )

    fps_vals = [r["fps"] for r in all_rows]
    mota_vals = [r["mota"] for r in all_rows]
    idf1_vals = [r["idf1"] for r in all_rows]
    hota_vals = [r["hota"] for r in all_rows]
    fps_min, fps_max = min(fps_vals), max(fps_vals)
    mota_min, mota_max = min(mota_vals), max(mota_vals)
    idf1_min, idf1_max = min(idf1_vals), max(idf1_vals)
    hota_min, hota_max = min(hota_vals), max(hota_vals)

    for row in all_rows:
        n_fps = norm(row["fps"], fps_min, fps_max)
        n_mota = norm(row["mota"], mota_min, mota_max)
        n_idf1 = norm(row["idf1"], idf1_min, idf1_max)
        n_hota = norm(row["hota"], hota_min, hota_max)
        row["score_normalized"] = n_hota + n_idf1 + n_mota + n_fps
        row["score_raw_sum"] = row["hota"] + row["idf1"] + row["mota"] + row["fps"]

    best_per_tracker: list[dict[str, Any]] = []
    for tracker in trackers:
        rows = [r for r in all_rows if r["tracker"] == tracker]
        if not rows:
            continue
        best = max(rows, key=lambda r: r["score_normalized"])
        best_per_tracker.append(best)

    ranked = sorted(best_per_tracker, key=lambda r: r["score_normalized"], reverse=True)
    for idx, row in enumerate(ranked, start=1):
        row["rank"] = idx

    out_json = (repo_root / args.out_json).resolve()
    out_md = (repo_root / args.out_md).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "trackers": trackers,
        "scoring": {
            "formula": "score_normalized = norm(HOTA)+norm(IDF1)+norm(MOTA)+norm(FPS)",
            "normalization_scope": "all configs across all compared trackers",
            "ranges": {
                "HOTA": [hota_min, hota_max],
                "IDF1": [idf1_min, idf1_max],
                "MOTA": [mota_min, mota_max],
                "FPS": [fps_min, fps_max],
            },
        },
        "ranked_trackers": ranked,
        "winner": ranked[0] if ranked else {},
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Tracker comparison (MOT17)",
        "",
        "Score formula: `norm(HOTA)+norm(IDF1)+norm(MOTA)+norm(FPS)`.",
        "",
        "| Rank | Tracker | conf | iou | HOTA | IDF1 | MOTA | FPS | Score |",
        "|------|---------|------|-----|------|------|------|-----|-------|",
    ]
    for row in ranked:
        cfg = row.get("config", {})
        lines.append(
            f"| {row['rank']} | {row['tracker_name']} | {cfg.get('conf')} | {cfg.get('iou')} | "
            f"{row['hota']:.6f} | {row['idf1']:.6f} | {row['mota']:.6f} | {row['fps']:.4f} | {row['score_normalized']:.4f} |"
        )
    if ranked:
        winner = ranked[0]
        wcfg = winner.get("config", {})
        lines.extend(
            [
                "",
                f"Winner: `{winner['tracker_name']}` with conf={wcfg.get('conf')}, iou={wcfg.get('iou')}.",
            ]
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"comparison_json": str(out_json), "comparison_md": str(out_md)}, indent=2))


if __name__ == "__main__":
    main()
