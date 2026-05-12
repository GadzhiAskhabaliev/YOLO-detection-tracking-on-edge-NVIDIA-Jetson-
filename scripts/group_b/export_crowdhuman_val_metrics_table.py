#!/usr/bin/env python3
"""
Emit a wide Markdown table: all numeric keys in metrics{} across runs + log link column.

  python3 scripts/group_b/export_crowdhuman_val_metrics_table.py > docs/crowdhuman_val_full_metrics_table.md

Edit DEFAULT_LOG_HREF in this file for per-model log URLs (in-repo paths or external).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "results" / "runs"
LOGS_DIR = REPO_ROOT / "results" / "logs"

# Newest match wins (mtime). Same basenames as `run_unified_coco_eval_group_b_three_yolo.sh` tees.
UNIFIED_LOG_GLOB_BY_MODEL: dict[str, str] = {
    "yolov8n_crowdhuman": "yolov8n_crowdhuman_unified_cocoeval_*.log",
    "freeyolo_ch_tiny": "freeyolo_yolo_free_tiny_unified_cocoeval_*.log",
    "freeyolo_yolox_mot17": "freeyolo_yolo_free_nano_unified_cocoeval_*.log",
    "peoplenet_crowdhuman": "peoplenet_crowdhuman_eval.log",
}

# model slug -> markdown link text for "Log" column if no unified log file yet (relative to docs/ or URL)
DEFAULT_LOG_HREF: dict[str, str] = {
    "yolov8n_crowdhuman": "[`yolov8n...203804Z.log`](../results/logs/yolov8n_crowdhuman_unified_cocoeval_2026-05-11T203804Z.log)",
    "freeyolo_ch_tiny": "latest `freeyolo_yolo_free_tiny_unified_cocoeval_*.log` (or add a fallback link in `DEFAULT_LOG_HREF`)",
    "freeyolo_yolox_mot17": "latest `freeyolo_yolo_free_nano_unified_cocoeval_*.log` (or add a fallback link in `DEFAULT_LOG_HREF`)",
    "ssd300_crowdhuman": "[CV-MMdetect `crowdhuman_val_run_2026-05-11.log`](https://raw.githubusercontent.com/GadzhiAskhabaliev/CV-MMdetect/main/results/logs/crowdhuman_val_run_2026-05-11.log) (shared tee: SSD + FCOS)",
    "fcos_r50_crowdhuman": "[same log as SSD](https://raw.githubusercontent.com/GadzhiAskhabaliev/CV-MMdetect/main/results/logs/crowdhuman_val_run_2026-05-11.log)",
    "crowddet_rcnn_emd_refine_e30": "[`crowddet_unified_metrics_epoch30.json`](../results/crowdhuman/crowddet_unified_metrics_epoch30.json); fork [CrowdDet-detection](https://github.com/GadzhiAskhabaliev/CrowdDet-detection) (`docs/UNIFIED_EVAL.md`)",
    "peoplenet_crowdhuman": "[`peoplenet_crowdhuman_eval.log`](../results/logs/peoplenet_crowdhuman_eval.log) (unified metrics JSON); [`peoplenet_unified_metrics_v2.json`](../results/crowdhuman/peoplenet_unified_metrics_v2.json)",
}

# Preferred column order (then append any other keys sorted)
PREFERRED = [
    "AP25",
    "AP50",
    "mAP50",
    "AP75",
    "AP50-95",
    "mAP50-95",
    "recall",
    "coco_ar_iou25",
    "coco_precision_r50_iou25",
    "coco_fdr_r50_iou25",
    "coco_ar_iou50",
    "coco_precision_r50_iou50",
    "coco_fdr_r50_iou50",
    "coco_ar_iou75",
    "coco_precision_r50_iou75",
    "coco_fdr_r50_iou75",
    "precision_iou25",
    "recall_iou25",
    "fdr_iou25",
    "precision_iou50",
    "recall_iou50",
    "fdr_iou50",
    "precision_iou75",
    "recall_iou75",
    "fdr_iou75",
    "precision",
    "fdr",
    "fps_forward",
    "fps_predict",
    "forward_time_ms_mean",
    "inference_time_ms_predict",
    "inference_time_ms",
    "eval_throughput_fps",
    "eval_wall_seconds",
    "eval_images",
]


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6f}".rstrip("0").rstrip(".")
    return str(v)


def _dedupe_latest_run_json(
    rows: list[tuple[Path, dict[str, Any]]],
) -> list[tuple[Path, dict[str, Any]]]:
    """If several `results/runs/*.json` share the same (backend, model), keep newest file by mtime."""
    best: dict[tuple[str, str], tuple[Path, dict[str, Any], float]] = {}
    for path, data in rows:
        model = str(data.get("model", path.stem))
        backend = str(data.get("backend", ""))
        key = (backend, model)
        mtime = path.stat().st_mtime
        if key not in best or mtime > best[key][2]:
            best[key] = (path, data, mtime)
    out = [(p, d) for p, d, _ in best.values()]
    out.sort(key=lambda t: (str(t[1].get("backend", "")), str(t[1].get("model", t[0].stem)), t[0].name))
    return out


def _unified_log_markdown(model: str) -> str | None:
    pat = UNIFIED_LOG_GLOB_BY_MODEL.get(model)
    if not pat or not LOGS_DIR.is_dir():
        return None
    matches = sorted(LOGS_DIR.glob(pat), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        return None
    name = matches[0].name
    return f"[`{name}`](../results/logs/{name})"


def _unified_logs_for_model(model: str) -> list[Path]:
    pat = UNIFIED_LOG_GLOB_BY_MODEL.get(model)
    if not pat or not LOGS_DIR.is_dir():
        return []
    return sorted(LOGS_DIR.glob(pat), key=lambda p: p.stat().st_mtime, reverse=True)


def main() -> None:
    rows: list[tuple[Path, dict[str, Any]]] = []
    for p in sorted(RUNS_DIR.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        rows.append((p, data))

    run_history: dict[tuple[str, str], list[Path]] = {}
    for path, data in rows:
        model = str(data.get("model", path.stem))
        backend = str(data.get("backend", ""))
        run_history.setdefault((backend, model), []).append(path)
    for key, paths in run_history.items():
        run_history[key] = sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)

    rows = _dedupe_latest_run_json(rows)

    all_keys: set[str] = set()
    for _, data in rows:
        m = data.get("metrics") or {}
        if isinstance(m, dict):
            all_keys.update(m.keys())

    ordered: list[str] = [k for k in PREFERRED if k in all_keys]
    ordered.extend(sorted(k for k in all_keys if k not in ordered))

    hdr = (
        ["Backend", "Model", "Run JSON"]
        + ordered
        + ["Log", "Artifacts"]
    )
    sep = ["---"] * len(hdr)

    parts: list[str] = []
    parts.append("# Unified benchmark table — all models, full `metrics`\n\n")
    parts.append(
        "Auto-generated: `python3 scripts/group_b/export_crowdhuman_val_metrics_table.py`. "
        "Numeric columns come from `results/runs/*.json` → `metrics`. "
        "Empty cells mean that key is absent for that run (e.g. unified merge not applied yet). "
        "**Log** for YOLO rows: newest `results/logs/*_unified_cocoeval_*.log` matching that model; "
        "otherwise the fallback upstream tee from "
        "`export_crowdhuman_val_metrics_table.py`. "
        "**SSD and FCOS** share one external tee from [CV-MMdetect](https://github.com/GadzhiAskhabaliev/CV-MMdetect). "
        "**CrowdDet** row: unified eval via [CrowdDet-detection](https://github.com/GadzhiAskhabaliev/CrowdDet-detection) fork (same `eval_coco_predictions.py` protocol). "
        "**PeopleNet** row: ONNXRuntime + NGC PeopleNet → COCO DT; unified metrics in `results/logs/peoplenet_crowdhuman_eval.log` (JSON) and `results/crowdhuman/peoplenet_unified_metrics_v2.json`.\n\n"
    )
    parts.append("| " + " | ".join(hdr) + " |\n")
    parts.append("| " + " | ".join(sep) + " |\n")

    for path, data in rows:
        model = str(data.get("model", path.stem))
        backend = str(data.get("backend", ""))
        met = data.get("metrics") or {}
        if not isinstance(met, dict):
            met = {}
        log_cell = _unified_log_markdown(model) or DEFAULT_LOG_HREF.get(
            model, "— (set `DEFAULT_LOG_HREF` in `export_crowdhuman_val_metrics_table.py`)"
        )
        cells = [
            backend,
            model,
            f"[`{path.name}`](../results/runs/{path.name})",
        ]
        for k in ordered:
            cells.append(_fmt(met.get(k)) if k in met else "")
        cells.append(log_cell)
        history = run_history.get((backend, model), [path])
        history_links = ", ".join(
            f"[`{p.name}`](../results/runs/{p.name})" for p in history[:3]
        )
        logs = _unified_logs_for_model(model)
        if logs:
            log_links = ", ".join(
                f"[`{p.name}`](../results/logs/{p.name})" for p in logs[:3]
            )
        else:
            log_links = "—"
        artifacts_cell = f"Run snapshots: {history_links}; Unified logs: {log_links}"
        cells.append(artifacts_cell)
        parts.append("| " + " | ".join(cells) + " |\n")

    parts.append("\n## Column hints\n\n")
    parts.append(
        "- **`recall`** — COCO AR maxDets=100, IoU 0.50:0.95.\n"
        "- **`coco_ar_iou*`** / **`coco_precision_r50_iou*`** / **`coco_fdr_r50_iou*`** — official COCO tensors (see `eval_coco_predictions.py`).\n"
        "- **`precision_iou*` / `recall_iou*` / `fdr_iou*`** — greedy micro metrics at score≥0.5.\n"
        "- **`mAP50`** vs **`AP50`** — legacy vs unified naming; same semantic when both present.\n"
    )

    sys.stdout.write("".join(parts))


if __name__ == "__main__":
    main()
