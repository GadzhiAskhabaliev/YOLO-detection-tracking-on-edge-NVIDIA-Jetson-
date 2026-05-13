#!/usr/bin/env python3
"""Repair YOLOv8 benchmark rows with per-config TrackEval artifacts."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


SEQ = "MOT17-02-FRCNN"
BENCHMARK_FILES = [
    "yolov8_bytetrack_mot17_benchmark.json",
    "yolov8_strongsort_mot17_benchmark.json",
    "yolov8_botsort_mot17_benchmark.json",
    "yolov8_hybridsort_mot17_benchmark.json",
    "yolov8_deepocsort_mot17_benchmark.json",
]


def cfg_suffix(conf: float, iou: float) -> str:
    return f"c{str(conf).replace('.', '')}_i{str(iou).replace('.', '')}"


def run_eval(repo_root: Path, pred_txt: Path, tracker_name: str) -> Path:
    env = os.environ.copy()
    env["PRED_TXT"] = str(pred_txt)
    env["MOT17_SEQ"] = SEQ
    env["TRACKER_NAME"] = tracker_name
    subprocess.run(
        ["bash", "scripts/tracking/eval_trackeval_mot17.sh"],
        cwd=str(repo_root),
        env=env,
        check=True,
    )
    metrics_path = repo_root / "results" / "tracking" / "trackeval" / f"{tracker_name}_{SEQ}_metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing TrackEval output: {metrics_path}")
    return metrics_path


def refresh_benchmark_md(json_path: Path, payload: dict) -> None:
    rows = payload.get("rows", [])
    out_md = json_path.with_suffix(".md")
    title = f"# {payload.get('tracker_name', json_path.stem.replace('_mot17_benchmark', ''))} benchmark (MOT17)"
    lines = [
        title,
        "",
        "| conf | iou | FPS e2e | latency ms | MOTA | IDF1 | HOTA |",
        "|------|-----|---------|------------|------|------|------|",
    ]
    for row in rows:
        cfg = row.get("config", {})
        lines.append(
            f"| {cfg.get('conf')} | {cfg.get('iou')} | {row.get('fps_e2e')} | {row.get('latency_ms_e2e')} | "
            f"{row.get('MOTA')} | {row.get('IDF1')} | {row.get('HOTA')} |"
        )
    if rows:
        best = rows[0]
        lines.extend(
            [
                "",
                f"Recommended baseline: conf={best['config']['conf']}, iou={best['config']['iou']}.",
            ]
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    results_dir = repo_root / "results" / "tracking"

    for name in BENCHMARK_FILES:
        bp = results_dir / name
        if not bp.is_file():
            print(f"skip missing: {bp}")
            continue
        payload = json.loads(bp.read_text(encoding="utf-8"))
        tracker_name = payload.get("tracker_name") or bp.name.replace("_mot17_benchmark.json", "")
        rows = payload.get("rows", [])

        for row in rows:
            cfg = row.get("config", {})
            conf = cfg.get("conf")
            iou = cfg.get("iou")
            suffix = cfg_suffix(conf, iou)
            cfg_tracker_name = f"{tracker_name}_{suffix}"

            report_path = repo_root / row["run_report"]
            report = json.loads(report_path.read_text(encoding="utf-8"))
            pred_txt = Path(report["artifacts"]["mot_txt"])
            if not pred_txt.is_file():
                pred_txt = (repo_root / report["artifacts"]["mot_txt"]).resolve()
            metrics_path = run_eval(repo_root, pred_txt, cfg_tracker_name)

            teval = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics = teval.get("metrics", {})
            row["MOTA"] = metrics.get("MOTA")
            row["IDF1"] = metrics.get("IDF1")
            row["HOTA"] = metrics.get("HOTA___AUC", metrics.get("HOTA"))
            row["trackeval_json"] = str(metrics_path.relative_to(repo_root))

        def score(item: dict) -> float:
            return float(item.get("MOTA") or 0) + float(item.get("IDF1") or 0) + float(item.get("HOTA") or 0)

        rows_sorted = sorted(rows, key=score, reverse=True)
        payload["rows"] = rows_sorted
        payload["recommended_baseline"] = rows_sorted[0] if rows_sorted else {}
        bp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        refresh_benchmark_md(bp, payload)
        print(f"updated: {bp}")

    print("repair complete")


if __name__ == "__main__":
    main()
