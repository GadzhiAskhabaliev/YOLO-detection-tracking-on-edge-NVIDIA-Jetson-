# Group B: remote CrowdDet / MMDet-class runs → unified mAP

Canonical **GT** and **`image_id`** live in the CrowdHuman bridge:

- Build once with `scripts/group_b/freeyolo_prepare_crowdhuman.py` → **`CrowdHuman/annotations/val.json`** under your `--bridge-root`.
- Do **not** regenerate `val.json` with a second converter that might assign different `images[].id`.

Comparable **mAP** uses this repo’s **`scripts/eval_coco_predictions.py`** (pycocotools) on the **same** `val.json` plus a dumped detection list.

Conversion helpers and SSH-oriented wrappers live in a separate repo (no script duplication here):

- **[CV-MMdetect](https://github.com/GadzhiAskhabaliev/CV-MMdetect)** — CrowdDet JSONL → DT JSON, GT id table for debugging, `env.template.sh`, `run_eval_remote.sh`.

## Mini flow on a GPU instance

After CrowdDet `tools/test.py` produces `dump-*.json` (JSONL):

```bash
python3 /path/to/CV-MMdetect/scripts/coco_dt/crowddet_dump_jsonl_to_dt.py \
  --val-json "$VAL_JSON" \
  --crowddet-jsonl /abs/path/to/dump-XXX.json \
  --out-json /abs/path/to/predictions_crowddet_val.json \
  --score-thr 0.05
```

Then evaluate against this repo (clone path on the instance = `$STUDY_REPO`):

```bash
bash /path/to/CV-MMdetect/scripts/instance/run_eval_remote.sh \
  /abs/path/to/predictions_crowddet_val.json crowddet_rcnn_emd_refine
```

`run_eval_remote.sh` calls `scripts/eval_coco_predictions.py` with `--strict`. Copy `scripts/instance/env.template.sh` on the instance and set `VAL_JSON`, `STUDY_REPO`, `OUT_DIR`.

## Pedestron / other MMDet forks

Use the same **`val.json`** and **`eval_coco_predictions.py`**. Once you have one sample output JSON from their `tools/test.py` (or saved results), add a small converter in **CV-MMdetect** mirroring the CrowdDet pattern (`image_id`, `category_id`, xywh `bbox`, `score`).

## Protocol notes (worth logging per run)

- **`--score-thr`** used in the converter (not only a “noise filter”) — record in run **`notes`** / bench patch.
- **`pred_cls_threshold`** / NMS inside upstream `test.py` — record too if changed.

## What to paste when debugging

- Full command + stderr/stdout.
- **Absolute path** to **`VAL_JSON`** actually used.
- **`pwd`** and path to the **`dump-*.json`** (wrong dump / wrong copy of `val.json` is common).
- First lines of the dump + a couple of **`images`** entries from `val.json` if mapping fails.
- **`.png` vs `.jpg`**: CrowdDet defaults expect **`{ID}.png`** in code; bridge GT uses **`{ID}.jpg`** — alignment bugs show up as skipped IDs, partial DT, or `--strict` failures.
