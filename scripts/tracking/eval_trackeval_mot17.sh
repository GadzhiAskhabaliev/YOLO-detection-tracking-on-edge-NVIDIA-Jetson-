#!/usr/bin/env bash
set -euo pipefail

# Evaluate MOTChallenge prediction txt with TrackEval (MOTA/IDF1/HOTA).
#
# Example:
#   MOT17_SEQ=MOT17-02-FRCNN \
#   PRED_TXT=results/tracking/yolov8_bytetrack_MOT17-02-FRCNN_*.txt \
#   bash scripts/tracking/eval_trackeval_mot17.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/tracking/path_defaults.sh"

TRACKEVAL_ROOT="${TRACKEVAL_ROOT:-${REPO_ROOT}/tools/TrackEval}"
MOT17_ROOT="${MOT17_ROOT:-${TRACKING_DATA_ROOT}/mot17}"
MOT17_SEQ="${MOT17_SEQ:-MOT17-02-FRCNN}"
TRACKER_NAME="${TRACKER_NAME:-yolov8_bytetrack}"
PRED_TXT="${PRED_TXT:-}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/results/tracking/trackeval}"

if [[ -z "${PRED_TXT}" ]]; then
  echo "PRED_TXT is required (path to MOTChallenge prediction txt)." >&2
  exit 1
fi
if [[ ! -f "${PRED_TXT}" ]]; then
  echo "Prediction file not found: ${PRED_TXT}" >&2
  exit 1
fi

if [[ ! -d "${TRACKEVAL_ROOT}" ]]; then
  mkdir -p "${REPO_ROOT}/tools"
  git clone https://github.com/JonathonLuiten/TrackEval.git "${TRACKEVAL_ROOT}"
fi

GT_SEQ_DIR="${MOT17_ROOT}/MOT17/train/${MOT17_SEQ}"
GT_TXT="${GT_SEQ_DIR}/gt/gt.txt"
SEQINFO="${GT_SEQ_DIR}/seqinfo.ini"
if [[ ! -f "${GT_TXT}" ]]; then
  echo "GT file not found: ${GT_TXT}" >&2
  exit 1
fi
if [[ ! -f "${SEQINFO}" ]]; then
  echo "seqinfo.ini not found: ${SEQINFO}" >&2
  exit 1
fi

WORK_DIR="${OUT_DIR}/work_${MOT17_SEQ}_$(date -u +%Y%m%dT%H%M%SZ)"
GT_FOLDER="${WORK_DIR}/gt_data"
TRACKERS_FOLDER="${WORK_DIR}/trackers_data"
SEQMAP_FILE="${WORK_DIR}/seqmaps/MOT17-train.txt"
mkdir -p "${GT_FOLDER}/MOT17-train/${MOT17_SEQ}/gt"
mkdir -p "${TRACKERS_FOLDER}/MOT17-train/${TRACKER_NAME}/data"
mkdir -p "$(dirname "${SEQMAP_FILE}")"
cp "${GT_TXT}" "${GT_FOLDER}/MOT17-train/${MOT17_SEQ}/gt/gt.txt"
cp "${SEQINFO}" "${GT_FOLDER}/MOT17-train/${MOT17_SEQ}/seqinfo.ini"
cp "${PRED_TXT}" "${TRACKERS_FOLDER}/MOT17-train/${TRACKER_NAME}/data/${MOT17_SEQ}.txt"
printf "name\n%s\n" "${MOT17_SEQ}" > "${SEQMAP_FILE}"

python3 - <<PY
import os
import sys
from pathlib import Path
import numpy as np

trackeval_root = Path(r"${TRACKEVAL_ROOT}").resolve()
if str(trackeval_root) not in sys.path:
    sys.path.insert(0, str(trackeval_root))

# TrackEval still references deprecated numpy aliases in some versions.
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]

try:
    import trackeval
except Exception as exc:
    raise SystemExit(
        f"Failed to import trackeval from {trackeval_root}. "
        "Ensure TrackEval is cloned and dependencies are installed."
    ) from exc

eval_config = trackeval.Evaluator.get_default_eval_config()
eval_config["USE_PARALLEL"] = False
eval_config["PRINT_CONFIG"] = True
eval_config["PRINT_RESULTS"] = True
eval_config["PLOT_CURVES"] = False

dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
dataset_config.update(
    {
        "BENCHMARK": "MOT17",
        "SPLIT_TO_EVAL": "train",
        "GT_FOLDER": str(Path(r"${GT_FOLDER}").resolve()),
        "TRACKERS_FOLDER": str(Path(r"${TRACKERS_FOLDER}").resolve()),
        "TRACKERS_TO_EVAL": [r"${TRACKER_NAME}"],
        "SEQMAP_FILE": str(Path(r"${SEQMAP_FILE}").resolve()),
        "OUTPUT_FOLDER": str(Path(r"${OUT_DIR}").resolve()),
        "DO_PREPROC": True,
        "PLOT_CURVES": False,
        "TRACKER_SUB_FOLDER": "data",
    }
)

metrics_config = {"METRICS": ["HOTA", "CLEAR", "Identity"], "THRESHOLD": 0.5}
metrics_list = []
for metric in (trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity):
    if metric.get_name() in metrics_config["METRICS"]:
        metrics_list.append(metric(metrics_config))
if not metrics_list:
    raise SystemExit("No metrics selected for TrackEval run.")

evaluator = trackeval.Evaluator(eval_config)
dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
evaluator.evaluate([dataset], metrics_list)
PY

TRACKER_OUT_DIR="${OUT_DIR}/${TRACKER_NAME}"
SUMMARY_TXT="${TRACKER_OUT_DIR}/pedestrian_summary.txt"
if [[ ! -f "${SUMMARY_TXT}" ]]; then
  SUMMARY_TXT="$(
  python3 - <<PY
from pathlib import Path
root = Path(r"${OUT_DIR}")
candidates = sorted(root.rglob("pedestrian_summary.txt"))
print(candidates[-1] if candidates else "")
PY
  )"
fi

if [[ -z "${SUMMARY_TXT}" ]]; then
  echo "TrackEval finished, but pedestrian_summary.txt not found under ${OUT_DIR}" >&2
  exit 1
fi

DETAILED_CSV="${TRACKER_OUT_DIR}/pedestrian_detailed.csv"
if [[ ! -f "${DETAILED_CSV}" ]]; then
  DETAILED_CSV="$(
  python3 - <<PY
from pathlib import Path
root = Path(r"${OUT_DIR}")
candidates = sorted(root.rglob("pedestrian_detailed.csv"))
print(candidates[-1] if candidates else "")
PY
  )"
fi

if [[ -z "${DETAILED_CSV}" ]]; then
  echo "TrackEval finished, but pedestrian_detailed.csv not found under ${OUT_DIR}" >&2
  exit 1
fi

OUT_JSON="${OUT_DIR}/${TRACKER_NAME}_${MOT17_SEQ}_metrics.json"
OUT_MD="${OUT_DIR}/${TRACKER_NAME}_${MOT17_SEQ}_summary.md"
export SUMMARY_TXT DETAILED_CSV OUT_JSON OUT_MD MOT17_SEQ TRACKER_NAME
python3 - <<'PY'
import json
import csv
import os
from pathlib import Path

summary = Path(os.environ["SUMMARY_TXT"])
detailed = Path(os.environ["DETAILED_CSV"])
out_json = Path(os.environ["OUT_JSON"])
out_md = Path(os.environ["OUT_MD"])
seq = os.environ["MOT17_SEQ"]
tracker = os.environ["TRACKER_NAME"]

with detailed.open("r", encoding="utf-8", errors="ignore") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

if not rows:
    raise SystemExit(f"TrackEval detailed parse failed: no rows in {detailed}")

metrics = None
seen_seq_labels = []
for row in rows:
    label = row.get("seq", "")
    seen_seq_labels.append(label)
    if label == seq:
        metrics = row
        break

if metrics is None:
    raise SystemExit(
        "TrackEval detailed parse failed: expected sequence row not found; "
        f"expected={seq!r}, seen={seen_seq_labels[:20]!r}"
    )

def _human_pct(key: str) -> str:
    v = metrics.get(key)
    if v is None or v == "":
        return "n/a"
    try:
        x = float(v)
    except ValueError:
        return str(v)
    return f"{x * 100:.3f}" if x <= 1.0 else f"{x:.3f}"

hota = _human_pct("HOTA___AUC")
mota = _human_pct("MOTA")
idf1 = _human_pct("IDF1")

payload = {
    "tracker": tracker,
    "sequence": seq,
    "split": "train",
    "benchmark_scope": "development_only",
    "source_summary_txt": str(summary),
    "source_detailed_csv": str(detailed),
    "metrics": metrics,
    "metrics_human": {"HOTA": hota, "MOTA": mota, "IDF1": idf1},
}
out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

md = [
    f"# TrackEval summary: {tracker} on {seq}",
    "",
    "- Split: `MOT17 train` (development benchmark; not MOTChallenge test-server result).",
    f"- Source: `{summary}`",
    f"- Detailed: `{detailed}`",
    f"- HOTA: `{hota}`",
    f"- MOTA: `{mota}`",
    f"- IDF1: `{idf1}`",
]
out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
print(json.dumps({"metrics_json": str(out_json), "summary_md": str(out_md)}, indent=2))
PY
