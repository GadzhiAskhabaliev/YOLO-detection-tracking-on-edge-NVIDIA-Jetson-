#!/usr/bin/env python3
"""Run 3 default YOLOv8+ByteTrack configs and aggregate speed + TrackEval metrics."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
from path_defaults import default_data_dir, default_models_dir


CONFIGS = [
    {"conf": 0.25, "iou": 0.7},
    {"conf": 0.35, "iou": 0.7},
    {"conf": 0.20, "iou": 0.6},
]


def run_cmd(cmd: list[str], cwd: Path, env_overrides: dict[str, str] | None = None) -> None:
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    subprocess.run(cmd, cwd=str(cwd), check=True, env=env)


def load_latest_json(glob_expr: str) -> Path:
    files = sorted(Path(".").glob(glob_expr), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No files matched: {glob_expr}")
    return files[-1]


def main() -> None:
    default_weights = default_models_dir() / "yolov8n_crowdhuman.pt"
    default_mot17_root = default_data_dir() / "mot17"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mot17-seq", default="MOT17-02-FRCNN")
    parser.add_argument("--weights", default=str(default_weights))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--mot17-root", default=str(default_mot17_root))
    parser.add_argument("--results-dir", default="results/tracking")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    results_dir = (repo_root / args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    trackeval_dir = results_dir / "trackeval"
    trackeval_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for cfg in CONFIGS:
        conf = cfg["conf"]
        iou = cfg["iou"]
        run_name = f"yolov8_bytetrack_{args.mot17_seq}_c{str(conf).replace('.', '')}_i{str(iou).replace('.', '')}"
        source = f"{args.mot17_root}/MOT17/train/{args.mot17_seq}/img1"

        run_cmd(
            [
                "python3",
                "scripts/tracking/run_yolov8_bytetrack_mot17.py",
                "--weights",
                args.weights,
                "--source",
                source,
                "--imgsz",
                str(args.imgsz),
                "--conf",
                str(conf),
                "--iou",
                str(iou),
                "--device",
                args.device,
                "--name",
                run_name,
                "--report-dir",
                str(results_dir),
                "--save-mot-txt",
            ],
            repo_root,
        )

        report_path = load_latest_json(f"{results_dir.relative_to(repo_root)}/{run_name}_*_run_report.json")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        mot_txt = report["artifacts"]["mot_txt"]

        run_cmd(
            ["bash", "scripts/tracking/eval_trackeval_mot17.sh"],
            repo_root,
            env_overrides={"PRED_TXT": mot_txt, "MOT17_SEQ": args.mot17_seq},
        )

        metrics_path = load_latest_json(
            f"{trackeval_dir.relative_to(repo_root)}/yolov8_bytetrack_{args.mot17_seq}_metrics.json"
        )
        teval = json.loads(metrics_path.read_text(encoding="utf-8"))
        m = teval.get("metrics", {})
        hota = m.get("HOTA")
        if hota is None:
            hota = m.get("HOTA___AUC")
        rows.append(
            {
                "config": {"conf": conf, "iou": iou},
                "fps_e2e": report.get("fps_e2e"),
                "latency_ms_e2e": report.get("latency_ms_e2e"),
                "MOTA": m.get("MOTA"),
                "IDF1": m.get("IDF1"),
                "HOTA": hota,
                "run_report": str(report_path),
                "trackeval_json": str(metrics_path),
            }
        )

    def score(row: dict) -> float:
        def f(v: str | float | None) -> float:
            try:
                return float(v) if v is not None else 0.0
            except ValueError:
                return 0.0

        return f(row.get("MOTA")) + f(row.get("IDF1")) + f(row.get("HOTA"))

    ranked = sorted(rows, key=score, reverse=True)
    best = ranked[0] if ranked else {}

    out_json = results_dir / "yolov8_bytetrack_mot17_benchmark.json"
    out_md = results_dir / "yolov8_bytetrack_mot17_benchmark.md"
    payload = {
        "sequence": args.mot17_seq,
        "weights": args.weights,
        "device": args.device,
        "rows": ranked,
        "recommended_baseline": best,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# YOLOv8n + ByteTrack benchmark (MOT17)",
        "",
        "| conf | iou | FPS e2e | latency ms | MOTA | IDF1 | HOTA |",
        "|------|-----|---------|------------|------|------|------|",
    ]
    for row in ranked:
        cfg = row["config"]
        lines.append(
            f"| {cfg['conf']} | {cfg['iou']} | {row.get('fps_e2e')} | {row.get('latency_ms_e2e')} | "
            f"{row.get('MOTA')} | {row.get('IDF1')} | {row.get('HOTA')} |"
        )
    if best:
        lines.extend(
            [
                "",
                f"Recommended baseline: conf={best['config']['conf']}, iou={best['config']['iou']}.",
            ]
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"benchmark_json": str(out_json), "benchmark_md": str(out_md)}, indent=2))


if __name__ == "__main__":
    main()
