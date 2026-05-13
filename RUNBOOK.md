# Jetson Docker Runbook (Safe Mode)

## Purpose
Run YOLOv8n-CrowdHuman + ByteTrack benchmarks on Jetson **strictly in Docker** with minimal risk to host system.

## Safety Policy
1. No host-level package installs (`apt`, `pip`) without explicit approval.
2. No driver/CUDA/TensorRT changes on host.
3. No writes outside approved workspace folders.
4. All experiments must be reproducible from container commands.
5. Every run must produce a short log record (config + metrics + artifact paths).

---

## 0. Pre-flight Questions (must be answered)
- Jetson model: AGX Orin / Orin NX / Xavier?
- JetPack / L4T version?
- Docker GPU mode allowed: `--gpus all` or `--runtime nvidia`?
- Allowed host workspace path and disk quota?
- Internet availability from Jetson?
- Is changing `nvpmodel` / `jetson_clocks` allowed?
- Any lab-approved base image to use?

---

## 1. Host Setup (safe directories only)

```bash
mkdir -p ~/edge/{repo,data,models,results,cache,logs}
```

Optional quick system snapshot:

```bash
uname -a | tee ~/edge/logs/host_info.txt
cat /etc/nv_tegra_release | tee -a ~/edge/logs/host_info.txt
docker --version | tee -a ~/edge/logs/host_info.txt
```

---

## 2. Launch Container

> Use lab-approved image if provided.

```bash
docker run --rm -it \
  --gpus all \
  --network host \
  --ipc host \
  -v ~/edge/repo:/workspace/repo \
  -v ~/edge/data:/workspace/data \
  -v ~/edge/models:/workspace/models \
  -v ~/edge/results:/workspace/results \
  -v ~/edge/cache:/workspace/.cache \
  -v ~/edge/logs:/workspace/logs \
  --name edge-yolo-track \
  <IMAGE>:<TAG> bash
```

If lab requires legacy runtime:
```bash
docker run --rm -it \
  --runtime nvidia \
  --network host \
  --ipc host \
  -v ~/edge/repo:/workspace/repo \
  -v ~/edge/data:/workspace/data \
  -v ~/edge/models:/workspace/models \
  -v ~/edge/results:/workspace/results \
  -v ~/edge/cache:/workspace/.cache \
  -v ~/edge/logs:/workspace/logs \
  --name edge-yolo-track \
  <IMAGE>:<TAG> bash
```

---

## 3. Inside Container: Environment Check

```bash
python3 -V
python3 -c "import torch; print('cuda=', torch.cuda.is_available(), 'torch=', torch.__version__)"
python3 -c "import cv2, numpy; print('cv2 ok, np ok')"
```

If any import fails: stop and log blocker, do not patch host.

---

## 4. Get Repository

```bash
cd /workspace/repo
# if first time:
git clone https://github.com/GadzhiAskhabaliev/real-time-people-detection-and-tracking-on-edge.git .
# otherwise:
git pull
```

Record commit:
```bash
git rev-parse --short HEAD | tee /workspace/logs/git_commit.txt
```

---

## 5. Install Project Dependencies (container only)

```bash
bash scripts/vast/install_deps.sh
pip3 install -r requirements-tracking.txt
```

If internet is blocked, switch to offline wheels (ask lab admin).

---

## 6. Data and Weights

### 6.1 CrowdHuman + MOT17
```bash
bash scripts/vast/download_crowdhuman_val.sh
bash scripts/vast/download_mot17.sh
bash scripts/vast/convert_crowdhuman_odgt.sh
bash scripts/vast/prepare_crowdhuman_yolo_layout.sh
```

### 6.2 YOLO weights
```bash
bash scripts/vast/download_yolov8n_crowdhuman.sh
```

Verify:
```bash
ls -lh /workspace/models/yolov8n_crowdhuman.pt
```

---

## 7. Dataset Layout Validation

```bash
python3 scripts/tracking/check_dataset_layout.py \
  --crowdhuman-root /workspace/data/crowdhuman \
  --mot17-root /workspace/data/mot17 \
  --out-json /workspace/results/tracking/dataset_layout_check.json
```

---

## 8. Tracking Smoke Test (short)

```bash
MOT17_ROOT=/workspace/data/mot17 \
MOT17_SEQ=MOT17-02-FRCNN \
WEIGHTS=/workspace/models/yolov8n_crowdhuman.pt \
bash scripts/tracking/run_yolov8_bytetrack_mot17.sh
```

Expected artifacts:
- `results/tracking/*_run_report.json`
- `results/tracking/*_raw_tracks.json`
- optional MOT txt export.

---

## 9. TrackEval Metrics (MOTA/IDF1/HOTA)

```bash
MOT17_ROOT=/workspace/data/mot17 \
MOT17_SEQ=MOT17-02-FRCNN \
bash scripts/tracking/eval_trackeval_mot17.sh
```

Expected artifacts:
- `results/tracking/trackeval/*_metrics.json`
- `results/tracking/trackeval/*_summary.md`

---

## 10. Benchmark Sweep (3 configs)

```bash
python3 scripts/tracking/run_tracking_benchmarks.py \
  --mot17-root /workspace/data/mot17 \
  --mot17-seq MOT17-02-FRCNN \
  --weights /workspace/models/yolov8n_crowdhuman.pt \
  --device cuda:0 \
  --imgsz 640 \
  --results-dir /workspace/results/tracking
```

Expected:
- `yolov8_bytetrack_mot17_benchmark.json`
- `yolov8_bytetrack_mot17_benchmark.md`

---

## 10.1 One-command safe run (recommended winner path)

This is the safest default for lab execution. It runs the winner setup with fail-fast guards and writes a run-note automatically.

```bash
PIN_COMMIT=c2a52ab \
MOT17_ROOT=/workspace/data/mot17 \
MODEL_DIR=/workspace/models \
REPORT_DIR=/workspace/results/tracking \
LOG_DIR=/workspace/logs \
bash scripts/tracking/run_jetson_winner_safe.sh
```

Default winner params inside the script:
- detector: `yolov8n_crowdhuman`
- tracker: `yolov8_bytetrack`
- conf: `0.35`
- iou: `0.7`

---

## 11. Run Log Template (fill after each run)

Create `/workspace/logs/run_note_<timestamp>.md` with:

- Date/time:
- Host: Jetson model + JetPack/L4T:
- Docker image tag:
- Git commit:
- Sequence:
- Params (`conf`, `iou`, `imgsz`, tracker):
- FPS e2e:
- MOTA / IDF1 / HOTA:
- Artifacts paths:
- Notes / issues:

---

## 12. Stop Conditions (do not improvise)

Stop and escalate if:
- container cannot access GPU,
- missing write permissions to workspace,
- disk quota exceeded,
- mandatory internet/download blocked,
- requires host-level changes not pre-approved.

---

## 13. Final Deliverables Checklist

- [ ] `dataset_layout_check.json`
- [ ] at least one `*_run_report.json`
- [ ] TrackEval metrics JSON + summary MD
- [ ] benchmark JSON + MD
- [ ] run logs in `/workspace/logs`
- [ ] short decision note: chosen baseline config and why

---

## 14. Notes on Scientific Validity

- MOT17 `train` evaluations are **development benchmarks**.
- Report split explicitly in text/results.
- For final claims, run multiple sequences and report average + variance.
- Keep cloud-vs-jetson comparisons separate.
