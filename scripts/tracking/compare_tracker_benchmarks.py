#!/usr/bin/env python3
"""Build unified comparison across all detectors and trackers."""

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


def raw_score(row: dict[str, Any]) -> float:
    return row["hota"] + row["idf1"] + row["mota"] + row["fps"]


def load_yolov8_candidates(results_dir: Path, trackers: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tracker in trackers:
        bench_path = results_dir / f"yolov8_{tracker}_mot17_benchmark.json"
        if not bench_path.is_file():
            raise SystemExit(f"Missing benchmark JSON: {bench_path}")
        payload = json.loads(bench_path.read_text(encoding="utf-8"))
        bench_rows = payload.get("rows", [])
        if not bench_rows:
            raise SystemExit(f"No rows in benchmark: {bench_path}")
        for row in bench_rows:
            cfg = row.get("config", {})
            rows.append(
                {
                    "detector": "yolov8n_crowdhuman",
                    "tracker": tracker,
                    "tracker_name": payload.get("tracker_name", f"yolov8_{tracker}"),
                    "conf": cfg.get("conf"),
                    "iou": cfg.get("iou"),
                    "fps": to_float(row.get("fps_e2e")),
                    "mota": to_float(row.get("MOTA")),
                    "idf1": to_float(row.get("IDF1")),
                    "hota": to_float(row.get("HOTA")),
                    "run_report": row.get("run_report", ""),
                    "trackeval_json": row.get("trackeval_json", ""),
                    "source_benchmark": str(bench_path.name),
                }
            )
    return rows


def load_freeyolo_candidates(results_dir: Path, trackers: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for det_slug in ("freeyolo_tiny", "freeyolo_nano"):
        suite_path = results_dir / f"{det_slug}_MOT17-02-FRCNN_tracker_suite_benchmark.json"
        if not suite_path.is_file():
            continue
        payload = json.loads(suite_path.read_text(encoding="utf-8"))
        for row in payload.get("rows", []):
            tracker = str(row.get("tracker", "")).strip().lower()
            if tracker not in trackers:
                continue
            rows.append(
                {
                    "detector": det_slug,
                    "tracker": tracker,
                    "tracker_name": tracker,
                    "conf": None,
                    "iou": None,
                    "fps": to_float(row.get("fps_e2e")),
                    "mota": to_float(row.get("MOTA")),
                    "idf1": to_float(row.get("IDF1")),
                    "hota": to_float(row.get("HOTA")),
                    "run_report": row.get("run_report", ""),
                    "trackeval_json": row.get("trackeval_json", ""),
                    "source_benchmark": str(suite_path.name),
                }
            )
    return rows


def pick_best_per_detector_tracker(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_map: dict[tuple[str, str], dict[str, Any]] = {}
    for row in candidates:
        key = (row["detector"], row["tracker"])
        cur = best_map.get(key)
        if cur is None or raw_score(row) > raw_score(cur):
            best_map[key] = row
    return list(best_map.values())


def build_detector_summaries(ranked_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_detector: dict[str, list[dict[str, Any]]] = {}
    for row in ranked_rows:
        by_detector.setdefault(row["detector"], []).append(row)

    summaries: list[dict[str, Any]] = []
    for detector, rows in by_detector.items():
        best = max(rows, key=lambda x: x["score_normalized"])
        summaries.append(
            {
                "detector": detector,
                "best_tracker": best["tracker_name"],
                "conf": best.get("conf"),
                "iou": best.get("iou"),
                "hota": best["hota"],
                "idf1": best["idf1"],
                "mota": best["mota"],
                "fps": best["fps"],
                "score_normalized": best["score_normalized"],
            }
        )
    summaries.sort(key=lambda x: x["score_normalized"], reverse=True)
    for idx, row in enumerate(summaries, start=1):
        row["rank"] = idx
    return summaries


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

    yolov8_candidates = load_yolov8_candidates(results_dir, trackers)
    freeyolo_candidates = load_freeyolo_candidates(results_dir, trackers)
    selected_rows = pick_best_per_detector_tracker(yolov8_candidates + freeyolo_candidates)
    if not selected_rows:
        raise SystemExit("No rows found for comparison.")

    fps_vals = [r["fps"] for r in selected_rows]
    mota_vals = [r["mota"] for r in selected_rows]
    idf1_vals = [r["idf1"] for r in selected_rows]
    hota_vals = [r["hota"] for r in selected_rows]
    fps_min, fps_max = min(fps_vals), max(fps_vals)
    mota_min, mota_max = min(mota_vals), max(mota_vals)
    idf1_min, idf1_max = min(idf1_vals), max(idf1_vals)
    hota_min, hota_max = min(hota_vals), max(hota_vals)

    for row in selected_rows:
        n_fps = norm(row["fps"], fps_min, fps_max)
        n_mota = norm(row["mota"], mota_min, mota_max)
        n_idf1 = norm(row["idf1"], idf1_min, idf1_max)
        n_hota = norm(row["hota"], hota_min, hota_max)
        row["score_normalized"] = n_hota + n_idf1 + n_mota + n_fps
        row["score_raw_sum"] = raw_score(row)

    ranked = sorted(selected_rows, key=lambda r: r["score_normalized"], reverse=True)
    for idx, row in enumerate(ranked, start=1):
        row["rank"] = idx

    detector_summaries = build_detector_summaries(ranked)

    out_json = (repo_root / args.out_json).resolve()
    out_md = (repo_root / args.out_md).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "trackers": trackers,
        "detectors": sorted({r["detector"] for r in ranked}),
        "scoring": {
            "formula": "score_normalized = norm(HOTA)+norm(IDF1)+norm(MOTA)+norm(FPS)",
            "normalization_scope": "best row per detector+tracker",
            "ranges": {
                "HOTA": [hota_min, hota_max],
                "IDF1": [idf1_min, idf1_max],
                "MOTA": [mota_min, mota_max],
                "FPS": [fps_min, fps_max],
            },
        },
        "ranked_rows": ranked,
        "detector_summaries": detector_summaries,
        "winner": ranked[0] if ranked else {},
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Tracker comparison (all models, MOT17)",
        "",
        "Score formula: `norm(HOTA)+norm(IDF1)+norm(MOTA)+norm(FPS)`.",
    ]

    preferred_detector_order = [
        "yolov8n_crowdhuman",
        "freeyolo_tiny",
        "freeyolo_nano",
    ]
    detector_order = preferred_detector_order + sorted(
        [d for d in {r["detector"] for r in ranked} if d not in preferred_detector_order]
    )

    for detector in detector_order:
        rows = [r for r in ranked if r["detector"] == detector]
        if not rows:
            continue
        local_ranked = sorted(rows, key=lambda x: x["score_normalized"], reverse=True)
        lines.extend(
            [
                "",
                f"## {detector}",
                "",
                "| Rank | Tracker | conf | iou | HOTA | IDF1 | MOTA | FPS | Score |",
                "|------|---------|------|-----|------|------|------|-----|-------|",
            ]
        )
        for idx, row in enumerate(local_ranked, start=1):
            conf = "-" if row.get("conf") is None else row.get("conf")
            iou = "-" if row.get("iou") is None else row.get("iou")
            lines.append(
                f"| {idx} | {row['tracker_name']} | {conf} | {iou} | "
                f"{row['hota']:.6f} | {row['idf1']:.6f} | {row['mota']:.6f} | {row['fps']:.4f} | {row['score_normalized']:.4f} |"
            )
        best_local = local_ranked[0]
        bconf = "-" if best_local.get("conf") is None else best_local.get("conf")
        biou = "-" if best_local.get("iou") is None else best_local.get("iou")
        lines.append("")
        lines.append(
            f"Winner ({detector}): `{best_local['tracker_name']}` with conf={bconf}, iou={biou}."
        )

    if detector_summaries:
        lines.extend(
            [
                "",
                "## Overall Summary",
                "",
                "| Rank | Detector | Best tracker | conf | iou | HOTA | IDF1 | MOTA | FPS | Score |",
                "|------|----------|--------------|------|-----|------|------|------|-----|-------|",
            ]
        )
        for row in detector_summaries:
            conf = "-" if row.get("conf") is None else row.get("conf")
            iou = "-" if row.get("iou") is None else row.get("iou")
            lines.append(
                f"| {row['rank']} | {row['detector']} | {row['best_tracker']} | {conf} | {iou} | "
                f"{row['hota']:.6f} | {row['idf1']:.6f} | {row['mota']:.6f} | {row['fps']:.4f} | {row['score_normalized']:.4f} |"
            )
        best_detector = detector_summaries[0]
        lines.extend(
            [
                "",
                f"Best detector baseline: `{best_detector['detector']} + {best_detector['best_tracker']}`.",
            ]
        )

    if ranked:
        winner = ranked[0]
        wconf = "-" if winner.get("conf") is None else winner.get("conf")
        wiou = "-" if winner.get("iou") is None else winner.get("iou")
        lines.extend(
            [
                "",
                f"Overall winner: `{winner['detector']} + {winner['tracker_name']}` with conf={wconf}, iou={wiou}.",
            ]
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"comparison_json": str(out_json), "comparison_md": str(out_md)}, indent=2))


if __name__ == "__main__":
    main()
