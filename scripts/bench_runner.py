#!/usr/bin/env python3
"""
Orchestrate pedestrian-detector benchmarks, persist unified JSON under results/runs/,
and refresh README + benchmark_summary.md.

Canonical metric names (`AP50`, …) and `backend`: docs/benchmark_metrics_schema.md.
Built-in Ultralytics driver for `.pt` weights is one backend, not the sole reference.

Examples:
  python scripts/bench_runner.py --model-name yolov8n_crowdhuman \\
    --weights models/yolov8n_crowdhuman.pt --weights-hub yakhyo/yolov8-crowdhuman \\
    --bench-mode predict

  python scripts/bench_runner.py --merge-json results/runs/foo.json \\
    --tracking-file results/runs/tracking_stub.json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

README_PATH = REPO_ROOT / "README.md"
RUNS_DIR = REPO_ROOT / "results" / "runs"
SUMMARY_MD = REPO_ROOT / "results" / "benchmark_summary.md"
TABLE_START = "<!-- TABLE_START -->"
TABLE_END = "<!-- TABLE_END -->"


def metric_ap50(m: dict[str, Any]) -> Any:
    """Unified eval uses AP50; legacy runs use mAP50 (COCO naming)."""
    v = m.get("AP50")
    return v if v is not None else m.get("mAP50")


def metric_ap5095(m: dict[str, Any]) -> Any:
    v = m.get("AP50-95")
    return v if v is not None else m.get("mAP50-95")


def run_backend_label(data: dict[str, Any]) -> str:
    """Short framework tag for tables (see docs/benchmark_metrics_schema.md)."""
    v = data.get("backend") or data.get("framework")
    return str(v).strip() if v else ""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sanitize_model_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip()).strip("_") or "model"


def detect_hardware() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
        if out:
            return out.splitlines()[0].strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "CPU / unknown"


def deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    for k, v in patch.items():
        if k == "notes" and isinstance(v, list):
            existing = base.get("notes")
            if isinstance(existing, list):
                existing.extend(v)
            else:
                base["notes"] = list(v)
            continue
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def tensorrt_meta(weights_path: Path) -> dict[str, Any]:
    engine = weights_path.with_suffix(".engine")
    return {"engine_exists": engine.is_file(), "fps_fp16": None}


def bench_predict_fps(
    weights: Path,
    *,
    imgsz: int,
    warmup: int,
    iters: int,
    device: str,
) -> dict[str, Any]:
    from ultralytics import YOLO

    model = YOLO(str(weights)).to(device)
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(warmup):
        model.predict(source=dummy, imgsz=imgsz, verbose=False, half=False)

    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        model.predict(source=dummy, imgsz=imgsz, verbose=False, half=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    ms = (dt / iters) * 1000.0
    fps = iters / dt
    return {"fps_predict": round(fps, 3), "inference_time_ms_predict": round(ms, 4)}


def bench_forward_fps(
    weights: Path,
    *,
    imgsz: int,
    warmup: int,
    iters: int,
    device: str,
) -> dict[str, Any]:
    from ultralytics import YOLO
    import torch

    model = YOLO(str(weights)).to(device)
    if device.startswith("cuda") and torch.cuda.is_available():
        dev = torch.device(device)
    else:
        dev = torch.device("cpu")

    m = model.model
    m.eval()
    x = torch.zeros((1, 3, imgsz, imgsz), device=dev, dtype=torch.float32)

    with torch.no_grad():
        for _ in range(warmup):
            _ = m(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = m(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

    fps = iters / dt
    ms = (dt / iters) * 1000.0
    return {"fps_forward": round(fps, 3), "forward_time_ms_mean": round(ms, 4)}


def bench_val_metrics(
    weights: Path,
    *,
    data_yaml: Path,
    imgsz: int,
    batch: int,
    device: str,
) -> dict[str, Any]:
    from ultralytics import YOLO

    model = YOLO(str(weights)).to(device)
    metrics = model.val(data=str(data_yaml), imgsz=imgsz, batch=batch, plots=False, verbose=False)
    box = getattr(metrics, "box", None)
    out: dict[str, Any] = {}

    if box is not None:
        out["AP50"] = round(float(getattr(box, "map50", 0.0) or 0.0), 6)
        out["AP50-95"] = round(float(getattr(box, "map", 0.0) or 0.0), 6)
        mp = getattr(box, "mp", None) or getattr(box, "p", None)
        mr = getattr(box, "mr", None) or getattr(box, "r", None)
        if mp is not None:
            out["precision"] = round(float(mp), 6)
        if mr is not None:
            out["recall"] = round(float(mr), 6)

    speed = getattr(metrics, "speed", None)
    if isinstance(speed, dict) and "inference" in speed:
        out["inference_time_ms"] = round(float(speed["inference"]), 4)

    return out


def default_payload(
    *,
    model_name: str,
    weights_path: Path,
    weights_hub: str,
    batch_size: int,
    imgsz: int,
    hardware: str | None = None,
    backend: str = "",
    group: str = "",
    detector_id: int | None = None,
    detector_label: str = "",
) -> dict[str, Any]:
    hw = hardware or detect_hardware()
    wp = weights_path.resolve()
    out: dict[str, Any] = {
        "model": model_name,
        "weights": str(wp),
        "weights_hub": weights_hub or "",
        "date": _utc_now_iso(),
        "hardware": hw,
        "batch_size": batch_size,
        "imgsz": imgsz,
        "metrics": {},
        "tracking": {},
        "tensorrt": tensorrt_meta(wp),
        "notes": [],
    }
    if group.strip():
        out["group"] = group.strip()
    if detector_id is not None:
        out["detector_id"] = detector_id
    if detector_label.strip():
        out["detector_label"] = detector_label.strip()
    if backend.strip():
        out["backend"] = backend.strip()
    return out


def refresh_comparison_md() -> None:
    try:
        from generate_comparison_table import regenerate as regen_compare

        regen_compare()
    except Exception:
        pass


def save_result(payload: dict[str, Any], *, out_path: Path | None = None) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    if out_path is None:
        stem = _sanitize_model_name(str(payload.get("model", "run")))
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
        out_path = RUNS_DIR / f"{stem}_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    update_readme_table()
    generate_benchmark_summary_md()
    refresh_comparison_md()
    return out_path


def merge_run_json(run_path: Path, patch: dict[str, Any]) -> Path:
    data = json.loads(run_path.read_text(encoding="utf-8"))
    deep_merge(data, patch)
    data["date_updated"] = _utc_now_iso()
    run_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    update_readme_table()
    generate_benchmark_summary_md()
    refresh_comparison_md()
    return run_path


def _iter_run_json() -> list[tuple[Path, dict[str, Any]]]:
    if not RUNS_DIR.is_dir():
        return []
    out: list[tuple[Path, dict[str, Any]]] = []
    for p in sorted(RUNS_DIR.glob("*.json")):
        try:
            out.append((p, json.loads(p.read_text(encoding="utf-8"))))
        except json.JSONDecodeError:
            continue
    return out


def _latest_per_model(rows: list[tuple[Path, dict[str, Any]]]) -> list[dict[str, Any]]:
    best: dict[str, tuple[str, dict[str, Any]]] = {}
    for _, data in rows:
        name = str(data.get("model", ""))
        dt = str(data.get("date", ""))
        cur = best.get(name)
        if cur is None or dt > cur[0]:
            best[name] = (dt, data)
    return [v[1] for v in best.values()]


def _fmt_cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4f}".rstrip("0").rstrip(".")
    return str(v)


def _append_unified_eval_bullets(parts: list[str], met: dict[str, Any]) -> None:
    """Extra lines when metrics come from scripts/eval_coco_predictions.py."""
    if "AP25" not in met and "coco_ar_iou25" not in met:
        return
    ap25 = met.get("AP25")
    ap75 = met.get("AP75")
    if ap25 is not None:
        parts.append(f"- **AP25**: {_fmt_cell(ap25)}\n")
    if ap75 is not None:
        parts.append(f"- **AP75**: {_fmt_cell(ap75)}\n")
    if met.get("recall") is not None:
        parts.append(
            f"- **COCO AR (recall, IoU=0.50:0.95, maxDets=100)**: {_fmt_cell(met.get('recall'))}\n"
        )
    for suf in ("25", "50", "75"):
        k = f"coco_ar_iou{suf}"
        if met.get(k) is not None:
            parts.append(f"- **coco AR @IoU0.{suf}**: {_fmt_cell(met.get(k))}\n")
    for suf in ("25", "50", "75"):
        pr, rc, fd = (
            met.get(f"precision_iou{suf}"),
            met.get(f"recall_iou{suf}"),
            met.get(f"fdr_iou{suf}"),
        )
        if pr is None and rc is None and fd is None:
            continue
        parts.append(
            f"- **Greedy micro @IoU0.{suf}** (score≥thr in eval): "
            f"P={_fmt_cell(pr)} R={_fmt_cell(rc)} FDR={_fmt_cell(fd)}\n"
        )
    if met.get("precision") is not None:
        parts.append(
            f"- **Greedy legacy** (`precision` / `fdr` @ `--precision-iou-thr`): "
            f"{_fmt_cell(met.get('precision'))} / {_fmt_cell(met.get('fdr'))}\n"
        )


def update_readme_table() -> None:
    rows_data = _iter_run_json()
    models = _latest_per_model(rows_data)

    def sort_key(d: dict[str, Any]) -> float:
        m = d.get("metrics") or {}
        return float(metric_ap50(m) or 0.0)

    models.sort(key=sort_key, reverse=True)

    lines = [
        "| Backend | Model | AP25 | AP50 | AP75 | AP50-95 | AR_coco | grP50 | grR50 | grF50 | cAR50 | "
        "FPS (forward) | FPS (predict) | MOTA | TRT FP16 | Date |",
        "|---------|-------|------|------|------|---------|---------|-------|-------|-------|-------|"
        "---------------|---------------|------|----------|------|",
    ]
    for d in models:
        met = d.get("metrics") or {}
        tr = d.get("tracking") or {}
        tt = d.get("tensorrt") or {}
        mota = tr.get("MOTA")
        trt = "yes" if tt.get("engine_exists") else "no"
        fp16 = tt.get("fps_fp16")
        trt_cell = f"{trt}" + (f" ({_fmt_cell(fp16)} FPS)" if fp16 is not None else "")
        lines.append(
            "| "
            + " | ".join(
                [
                    run_backend_label(d),
                    str(d.get("model", "")),
                    _fmt_cell(met.get("AP25")),
                    _fmt_cell(metric_ap50(met)),
                    _fmt_cell(met.get("AP75")),
                    _fmt_cell(metric_ap5095(met)),
                    _fmt_cell(met.get("recall")),
                    _fmt_cell(met.get("precision_iou50")),
                    _fmt_cell(met.get("recall_iou50")),
                    _fmt_cell(met.get("fdr_iou50")),
                    _fmt_cell(met.get("coco_ar_iou50")),
                    _fmt_cell(met.get("fps_forward")),
                    _fmt_cell(met.get("fps_predict")),
                    _fmt_cell(mota) if mota is not None else "",
                    trt_cell,
                    str(d.get("date", "")),
                ]
            )
            + " |"
        )

    table = "\n".join(lines)
    if not README_PATH.is_file():
        return
    text = README_PATH.read_text(encoding="utf-8")
    if TABLE_START not in text or TABLE_END not in text:
        return
    pre, rest = text.split(TABLE_START, 1)
    _, post = rest.split(TABLE_END, 1)
    new_body = f"\n\n{table}\n\n"
    README_PATH.write_text(pre + TABLE_START + new_body + TABLE_END + post, encoding="utf-8")


def generate_benchmark_summary_md() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_MD.parent.mkdir(parents=True, exist_ok=True)

    rows_data = _iter_run_json()
    # Newest first within same model is approximate: sort all runs by date desc
    def run_sort_key(item: tuple[Path, dict[str, Any]]) -> str:
        return str(item[1].get("date", ""))

    rows_sorted = sorted(rows_data, key=run_sort_key, reverse=True)

    parts: list[str] = []
    parts.append("# Benchmark runs (auto-generated)\n")
    parts.append(
        "Sources: `results/runs/*.json`. Updated by `scripts/bench_runner.py` and callers of "
        "`save_result` / `merge_run_json`.\n"
    )

    for path, data in rows_sorted:
        model = str(data.get("model", ""))
        dt = str(data.get("date", ""))
        hub = str(data.get("weights_hub", "") or "")
        weights = str(data.get("weights", ""))
        hw = str(data.get("hardware", ""))
        met = data.get("metrics") or {}
        notes = data.get("notes") or []

        parts.append(f"\n## [{model}] — {dt}\n")
        parts.append(f"- **File**: `{path.relative_to(REPO_ROOT)}`\n")
        parts.append(f"- **Weights (path)**: `{weights}`\n")
        if hub:
            parts.append(f"- **Weights (id / hub)**: `{hub}`\n")
        parts.append(f"- **Hardware**: {hw}\n")
        bk = run_backend_label(data)
        if bk:
            parts.append(f"- **Backend**: `{bk}`\n")
        parts.append(f"- **AP50**: {_fmt_cell(metric_ap50(met))}\n")
        _append_unified_eval_bullets(parts, met)
        parts.append(f"- **FPS forward**: {_fmt_cell(met.get('fps_forward'))}\n")
        parts.append(f"- **FPS predict**: {_fmt_cell(met.get('fps_predict'))}\n")
        if notes:
            parts.append(f"- **Notes**: {'; '.join(str(n) for n in notes)}\n")

    parts.append("\n---\n\n## Summary table (all runs)\n\n")
    parts.append(
        "Legend: **AR_coco** = `metrics.recall` (COCO AR maxDets=100, IoU 0.50:0.95). "
        "**grP50/grR50/grF50** = greedy micro P/R/FDR at IoU 0.50 (`precision_iou50` / "
        "`recall_iou50` / `fdr_iou50`), score threshold as in eval notes. "
        "**cAR50** = `coco_ar_iou50` (COCO AR @ IoU 0.50). Extra columns are blank if the run "
        "JSON predates unified eval.\n\n"
    )
    hdr = (
        "| Backend | Model | Date | AP25 | AP50 | AP75 | AP50-95 | AR_coco | grP50 | grR50 | grF50 | cAR50 | "
        "Infer (ms) | FPS fwd | FPS pred | MOTA | TRT |\n"
    )
    sep = "|---------|--------|------|------|------|------|---------|---------|-------|-------|-------|-------|"
    sep += "------------|---------|----------|------|-----|\n"
    parts.append(hdr + sep)

    for _, data in sorted(rows_data, key=run_sort_key, reverse=True):
        met = data.get("metrics") or {}
        tr = data.get("tracking") or {}
        tt = data.get("tensorrt") or {}
        trt = "yes" if tt.get("engine_exists") else "no"
        parts.append(
            "| "
            + " | ".join(
                [
                    run_backend_label(data),
                    str(data.get("model", "")),
                    str(data.get("date", "")),
                    _fmt_cell(met.get("AP25")),
                    _fmt_cell(metric_ap50(met)),
                    _fmt_cell(met.get("AP75")),
                    _fmt_cell(metric_ap5095(met)),
                    _fmt_cell(met.get("recall")),
                    _fmt_cell(met.get("precision_iou50")),
                    _fmt_cell(met.get("recall_iou50")),
                    _fmt_cell(met.get("fdr_iou50")),
                    _fmt_cell(met.get("coco_ar_iou50")),
                    _fmt_cell(met.get("inference_time_ms")),
                    _fmt_cell(met.get("fps_forward")),
                    _fmt_cell(met.get("fps_predict")),
                    _fmt_cell(tr.get("MOTA")),
                    trt,
                ]
            )
            + " |\n"
        )

    SUMMARY_MD.write_text("".join(parts), encoding="utf-8")


def run_benchmarks(args: argparse.Namespace) -> dict[str, Any]:
    weights = Path(args.weights).expanduser()
    if not weights.is_file():
        raise SystemExit(f"Weights not found: {weights}")

    payload = default_payload(
        model_name=args.model_name.strip(),
        weights_path=weights,
        weights_hub=args.weights_hub or "",
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        backend=(getattr(args, "backend", None) or "").strip(),
        group=getattr(args, "group", "") or "",
        detector_id=getattr(args, "detector_id", None),
        detector_label=getattr(args, "detector_label", "") or "",
    )
    modes = set()
    if args.bench_mode == "all":
        modes.update({"forward", "predict", "val"})
    else:
        modes.add(args.bench_mode)

    if "forward" in modes:
        payload["metrics"].update(
            bench_forward_fps(
                weights,
                imgsz=args.imgsz,
                warmup=args.warmup,
                iters=args.iters,
                device=args.device,
            )
        )
    if "predict" in modes:
        payload["metrics"].update(
            bench_predict_fps(
                weights,
                imgsz=args.imgsz,
                warmup=args.warmup,
                iters=args.iters,
                device=args.device,
            )
        )
    if "val" in modes:
        dy = Path(args.data_yaml).expanduser()
        if not dy.is_file():
            raise SystemExit(f"Dataset YAML not found: {dy}")
        payload["metrics"].update(
            bench_val_metrics(
                weights,
                data_yaml=dy,
                imgsz=args.imgsz,
                batch=args.batch_size,
                device=args.device,
            )
        )

    if args.bench_mode == "tracking":
        payload["notes"].append(
            "Tracking mode is not executed in bench_runner; add metrics via --merge-json."
        )

    return payload


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model-name",
        default="",
        help="Logical experiment name, e.g. yolov8n_crowdhuman (not used with --merge-json)",
    )
    p.add_argument(
        "--weights",
        default="",
        help="Path to .pt weights (not used with --merge-json)",
    )
    p.add_argument("--weights-hub", default="", help="Optional HF/YOLO hub id, e.g. yakhyo/yolov8-crowdhuman")
    p.add_argument(
        "--bench-mode",
        choices=("forward", "predict", "val", "tracking", "all"),
        default="predict",
        help="Benchmark segment to run (tracking is a stub unless merged separately)",
    )
    p.add_argument("--data-yaml", default=str(REPO_ROOT / "configs/datasets/crowdhuman_val.yaml"))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--merge-json", type=Path, default=None, help="Merge fields into an existing run JSON")
    p.add_argument(
        "--tracking-file",
        type=Path,
        default=None,
        help="JSON file merged under `tracking` (with --merge-json)",
    )
    p.add_argument(
        "--tracking-json",
        default="",
        help='Inline JSON for `tracking`, e.g. \'{"MOTA":0.68,"mot17_seq":"MOT17-02"}\'',
    )
    p.add_argument(
        "--patch-json",
        type=Path,
        default=None,
        help="With --merge-json: deep-merge arbitrary JSON into the run file (e.g. group B meta)",
    )
    p.add_argument("--group", default="", help='Experiment tag in saved JSON, e.g. "B"')
    p.add_argument("--detector-id", type=int, default=None, help="Slot id from docs/yolo_detectors_manifest.yaml")
    p.add_argument("--detector-label", default="", help="Display name for reports")
    p.add_argument(
        "--backend",
        default="ultralytics_yolo",
        help="Framework tag for summary tables (docs/benchmark_metrics_schema.md); default for Ultralytics driver",
    )
    args = p.parse_args(argv)

    if args.merge_json:
        run_path = args.merge_json.expanduser().resolve()
        if not run_path.is_file():
            raise SystemExit(f"--merge-json not found: {run_path}")
        patch: dict[str, Any] = {}
        if args.patch_json:
            ppath = args.patch_json.expanduser().resolve()
            if not ppath.is_file():
                raise SystemExit(f"--patch-json not found: {ppath}")
            patch = json.loads(ppath.read_text(encoding="utf-8"))
        elif args.tracking_json.strip():
            patch["tracking"] = json.loads(args.tracking_json)
        elif args.tracking_file:
            tpath = args.tracking_file.expanduser().resolve()
            if not tpath.is_file():
                raise SystemExit(f"--tracking-file not found: {tpath}")
            patch["tracking"] = json.loads(tpath.read_text(encoding="utf-8"))
        if not patch:
            raise SystemExit(
                "Nothing to merge: pass --patch-json, --tracking-json, or --tracking-file."
            )
        merge_run_json(run_path, patch)
        print(f"Merged into {run_path}")
        return

    if not args.model_name.strip():
        raise SystemExit("--model-name is required unless --merge-json is used.")
    if not args.weights.strip():
        raise SystemExit("--weights is required unless --merge-json is used.")

    payload = run_benchmarks(args)
    out = save_result(payload)
    print(json.dumps({"saved": str(out.relative_to(REPO_ROOT))}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
