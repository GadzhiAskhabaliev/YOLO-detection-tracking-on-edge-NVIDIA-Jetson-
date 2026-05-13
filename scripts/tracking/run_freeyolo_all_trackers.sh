#!/usr/bin/env bash
set -euo pipefail

# Coordinated pipeline:
# 1) Prepare FreeYOLO env + two CrowdHuman weights (tiny, nano)
# 2) Dump MOT17 detections for each FreeYOLO variant
# 3) Run trackers: bytetrack,strongsort,botsort,hybridsort,deepocsort
# 4) Evaluate with TrackEval and build detector-level tracker benchmarks

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
source "${ROOT}/scripts/tracking/path_defaults.sh"

MOT17_ROOT="${MOT17_ROOT:-${TRACKING_DATA_ROOT}/mot17}"
MOT17_SEQ="${MOT17_SEQ:-MOT17-02-FRCNN}"
SEQ_IMG1="${MOT17_ROOT}/MOT17/train/${MOT17_SEQ}/img1"
TRACKERS="${TRACKERS:-bytetrack,strongsort,botsort,hybridsort,deepocsort}"

GROUP_B_ROOT="${GROUP_B_ROOT:-${TRACKING_WORK_ROOT}/group_b}"
FREEYOLO_HOME="${FREEYOLO_HOME:-${GROUP_B_ROOT}/FreeYOLO}"
FREEYOLO_VENV="${FREEYOLO_VENV:-${GROUP_B_ROOT}/venv_freeyolo}"
FREEYOLO_REV="${FREEYOLO_REV:-30ca71424c965bb61917e1a9579dabd71b55c64e}"

MODEL_DIR="${MODEL_DIR:-${TRACKING_MODEL_ROOT}}"
TINY_WEIGHT="${TINY_WEIGHT:-${MODEL_DIR}/yolo_free_tiny_ch.pth}"
NANO_WEIGHT="${NANO_WEIGHT:-${MODEL_DIR}/yolo_free_nano_ch.pth}"
TINY_URL="${TINY_URL:-https://github.com/yjh0410/FreeYOLO/releases/download/weight/yolo_free_tiny_ch.pth}"
NANO_URL="${NANO_URL:-https://github.com/yjh0410/FreeYOLO/releases/download/weight/yolo_free_nano_ch.pth}"

GT_JSON="${GT_JSON:-${MOT17_ROOT}/annotations/mot17_train_frcnn_gt.json}"
mkdir -p "${MODEL_DIR}" "${MOT17_ROOT}/annotations" "results/tracking"

[[ -d "${SEQ_IMG1}" ]] || { echo "Missing MOT17 sequence img1: ${SEQ_IMG1}" >&2; exit 1; }

if [[ ! -f "${GT_JSON}" ]]; then
  python3 scripts/mot17_gt_to_coco.py \
    --mot17-train-root "${MOT17_ROOT}/MOT17/train" \
    --det-subdir-suffix FRCNN \
    --out-json "${GT_JSON}"
fi

if [[ ! -d "${FREEYOLO_HOME}/.git" ]]; then
  git clone https://github.com/yjh0410/FreeYOLO.git "${FREEYOLO_HOME}"
fi
git -C "${FREEYOLO_HOME}" fetch origin
git -C "${FREEYOLO_HOME}" checkout "${FREEYOLO_REV}"
python3 scripts/group_b/patch_freeyolo_torch_load.py --freeyolo-home "${FREEYOLO_HOME}"
python3 scripts/group_b/patch_freeyolo_numpy_aliases.py --freeyolo-home "${FREEYOLO_HOME}"
python3 scripts/group_b/patch_freeyolo_tiny_ckpt_compat.py --freeyolo-home "${FREEYOLO_HOME}"

if [[ ! -d "${FREEYOLO_VENV}" ]]; then
  python3 -m venv "${FREEYOLO_VENV}"
fi
# shellcheck disable=SC1090
source "${FREEYOLO_VENV}/bin/activate"
FY_PY="${FREEYOLO_VENV}/bin/python"
FY_PIP="${FREEYOLO_VENV}/bin/pip"
"${FY_PIP}" install -q --upgrade pip wheel
"${FY_PIP}" install -q "numpy>=1.23,<2"
"${FY_PIP}" install -q torch torchvision --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
"${FY_PIP}" install -q opencv-python scipy matplotlib pycocotools loguru thop Pillow
"${FY_PIP}" install -q "numpy>=1.23,<2" --force-reinstall --no-deps

[[ -f "${TINY_WEIGHT}" ]] || { wget -O "${TINY_WEIGHT}.part" "${TINY_URL}" && mv "${TINY_WEIGHT}.part" "${TINY_WEIGHT}"; }
[[ -f "${NANO_WEIGHT}" ]] || { wget -O "${NANO_WEIGHT}.part" "${NANO_URL}" && mv "${NANO_WEIGHT}.part" "${NANO_WEIGHT}"; }

run_variant() {
  local variant="$1"      # yolo_free_tiny | yolo_free_nano
  local weight_path="$2"  # path to .pth
  local detector_slug="$3"  # freeyolo_tiny | freeyolo_nano
  local dt_json="results/tracking/${detector_slug}_${MOT17_SEQ}_dt.json"
  local det_root="${MOT17_ROOT}/detections/${detector_slug}_${MOT17_SEQ}"
  local det_txt="${det_root}/${MOT17_SEQ}/det.txt"

  "${FY_PY}" scripts/group_b/dump_freeyolo_mot17.py \
    --freeyolo-home "${FREEYOLO_HOME}" \
    --variant "${variant}" \
    --weights "${weight_path}" \
    --gt-json "${GT_JSON}" \
    --mot17-train-root "${MOT17_ROOT}/MOT17/train" \
    --out-coco-dt-json "${dt_json}" \
    --mot-det-root "${det_root}"

  [[ -f "${det_txt}" ]] || { echo "Missing det txt: ${det_txt}" >&2; exit 1; }

  IFS=',' read -r -a tracker_arr <<< "${TRACKERS}"
  for tracker in "${tracker_arr[@]}"; do
    tracker="$(echo "${tracker}" | xargs)"
    [[ -n "${tracker}" ]] || continue

    python3 scripts/tracking/run_boxmot_on_mot_detections.py \
      --img1-dir "${SEQ_IMG1}" \
      --det-txt "${det_txt}" \
      --tracker-type "${tracker}" \
      --tracker-label "${tracker}" \
      --detector-label "${detector_slug}" \
      --report-dir "results/tracking"

    report_path="$(
      python3 - <<PY
from pathlib import Path
files = sorted(Path("results/tracking").glob("${detector_slug}_${tracker}_${MOT17_SEQ}_*_run_report.json"), key=lambda p: p.stat().st_mtime)
if not files:
    raise SystemExit("no run report")
print(files[-1])
PY
    )"

    pred_txt="$(
      python3 - <<PY
import json
from pathlib import Path
obj = json.loads(Path("${report_path}").read_text(encoding="utf-8"))
print(obj["artifacts"]["mot_txt"])
PY
    )"

    TRACKER_NAME="${detector_slug}_${tracker}" MOT17_SEQ="${MOT17_SEQ}" PRED_TXT="${pred_txt}" \
      bash scripts/tracking/eval_trackeval_mot17.sh
  done

  python3 - <<PY
import json
from pathlib import Path

detector_slug = "${detector_slug}"
seq = "${MOT17_SEQ}"
trackers = [x.strip() for x in "${TRACKERS}".split(",") if x.strip()]
root = Path("results/tracking")

rows = []
for tracker in trackers:
    reports = sorted(root.glob(f"{detector_slug}_{tracker}_{seq}_*_run_report.json"), key=lambda p: p.stat().st_mtime)
    if not reports:
        continue
    report_path = reports[-1]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    mpath = root / "trackeval" / f"{detector_slug}_{tracker}_{seq}_metrics.json"
    metrics = {}
    if mpath.is_file():
        m = json.loads(mpath.read_text(encoding="utf-8")).get("metrics", {})
        metrics = {
            "MOTA": m.get("MOTA"),
            "IDF1": m.get("IDF1"),
            "HOTA": m.get("HOTA___AUC", m.get("HOTA")),
        }
    rows.append(
        {
            "tracker": tracker,
            "fps_e2e": report.get("fps_e2e"),
            "latency_ms_e2e": report.get("latency_ms_e2e"),
            **metrics,
            "run_report": str(report_path),
            "trackeval_json": str(mpath),
        }
    )

def score(r):
    def f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0
    return f(r.get("MOTA")) + f(r.get("IDF1")) + f(r.get("HOTA"))

ranked = sorted(rows, key=score, reverse=True)
payload = {
    "detector": detector_slug,
    "sequence": seq,
    "rows": ranked,
    "recommended_baseline": ranked[0] if ranked else {},
}
out_json = root / f"{detector_slug}_{seq}_tracker_suite_benchmark.json"
out_md = root / f"{detector_slug}_{seq}_tracker_suite_benchmark.md"
out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

lines = [
    f"# Tracker suite benchmark: {detector_slug} on {seq}",
    "",
    "| tracker | FPS e2e | latency ms | MOTA | IDF1 | HOTA |",
    "|---------|---------|------------|------|------|------|",
]
for r in ranked:
    lines.append(
        f"| {r['tracker']} | {r.get('fps_e2e')} | {r.get('latency_ms_e2e')} | "
        f"{r.get('MOTA')} | {r.get('IDF1')} | {r.get('HOTA')} |"
    )
if ranked:
    lines.append("")
    lines.append(f"Recommended baseline: `{ranked[0]['tracker']}`.")
out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({"benchmark_json": str(out_json), "benchmark_md": str(out_md)}, indent=2))
PY
}

run_variant "yolo_free_tiny" "${TINY_WEIGHT}" "freeyolo_tiny"
run_variant "yolo_free_nano" "${NANO_WEIGHT}" "freeyolo_nano"

deactivate || true
echo "Done: FreeYOLO tiny+nano tracker suite complete."
