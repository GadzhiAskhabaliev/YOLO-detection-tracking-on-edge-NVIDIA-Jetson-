#!/usr/bin/env python3
"""
Print markdown table body lines for docs/benchmark_unified_cocoeval.md (CrowdHuman val)
from one or more `results/runs/*.json` files (after unified eval merge).

Example:

  python3 scripts/group_b/print_crowdhuman_unified_md_rows.py \\
    results/runs/yolov8n_crowdhuman_2026-05-09T143848Z.json \\
    --tee-log results/logs/yolov8n_crowdhuman_unified_cocoeval_2026-05-11T203804Z.log
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


def _num(m: dict[str, Any], *keys: str) -> float | None:
    for k in keys:
        if k in m and m[k] is not None:
            return float(m[k])
    return None


def _fmt(v: float | None, nd: int = 6) -> str:
    if v is None:
        return ""
    return f"{v:.{nd}f}"


def _fps(m: dict[str, Any], k: str) -> str:
    v = m.get(k)
    if v is None:
        return "—"
    return _fmt(float(v), 3)


def _log_from_notes(notes: Any) -> str | None:
    if not isinstance(notes, list):
        return None
    for line in notes:
        if not isinstance(line, str):
            continue
        m = re.search(r"results/logs/[^\s`]+\.log", line)
        if m:
            return m.group(0)
        if "unified_cocoeval" in line and ".log" in line:
            m2 = re.search(r"`([^`]+\.log)`", line)
            if m2:
                return m2.group(1)
    return None


def row_for_run(
    run_path: Path,
    *,
    tee_log: str | None,
) -> str:
    data = json.loads(run_path.read_text(encoding="utf-8"))
    backend = str(data.get("backend", "")).strip()
    model = str(data.get("model", "")).strip()
    met = data.get("metrics") or {}
    ap50 = _num(met, "AP50", "mAP50")
    ap5095 = _num(met, "AP50-95", "mAP50-95")
    recall = _num(met, "recall")
    if ap50 is None or ap5095 is None or recall is None:
        raise SystemExit(
            f"{run_path}: missing AP50/mAP50, AP50-95/mAP50-95, or recall in metrics — run unified eval merge first."
        )

    log = (tee_log or "").strip() or _log_from_notes(data.get("notes"))
    rel_json = f"[`{run_path.name}`](../results/runs/{run_path.name})"
    if log:
        log_href = log if log.startswith("../") else f"../{log}"
        src = f"{rel_json} + log [`{Path(log).name}`]({log_href})"
    else:
        src = f"{rel_json} (add tee log path to doc if available)"

    return (
        "| "
        + " | ".join(
            [
                backend,
                model,
                _fmt(ap50),
                _fmt(ap5095),
                _fmt(recall),
                _fps(met, "fps_forward"),
                _fps(met, "fps_predict"),
                src,
            ]
        )
        + " |\n"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "run_json",
        type=Path,
        nargs="+",
        help="results/runs/*.json (order = output row order)",
    )
    p.add_argument(
        "--tee-log",
        action="append",
        default=[],
        help="Optional tee log path (repeat in same order as run_json; use ../results/logs/... in docs)",
    )
    args = p.parse_args()
    tees = list(args.tee_log)
    if len(tees) > 1 and len(tees) != len(args.run_json):
        raise SystemExit("Pass either zero, one, or N --tee-log values matching N run_json files.")
    for i, rp in enumerate(args.run_json):
        if not rp.is_file():
            raise SystemExit(f"Not found: {rp}")
        tl = None
        if len(tees) == len(args.run_json):
            tl = tees[i]
        elif len(tees) == 1:
            tl = tees[0]
        sys.stdout.write(row_for_run(rp, tee_log=tl))


if __name__ == "__main__":
    main()
