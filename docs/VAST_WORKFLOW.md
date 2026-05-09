# Repository layout vs Vast.ai instances

**Git holds code and notes**; **datasets and weights** stay on instance disk (`/workspace`) and must not be committed.

## Roles

| Location | Contents |
|----------|----------|
| **Local / GitHub** (this repo) | `scripts/`, experiment `configs/*.yaml`, `docs/`, small utilities |
| **On Vast** (`/workspace/...`) | `data/` (CrowdHuman, MOT17), downloaded `.pt`, `kagglehub` cache, run outputs |

Paths differ across machines: scripts should key off one env var, e.g. `export PROJECT_ROOT=/workspace/real-time-edge`.

## Workflow

1. Edit scripts locally → commit → push to GitHub (or keep changes local only).

2. **Once per instance**, clone the repo:
   ```bash
   cd /workspace
   git clone https://github.com/YOU/YOUR_REPO.git real-time-edge
   cd real-time-edge
   ```

3. Before each Vast session:
   ```bash
   cd /workspace/real-time-edge && git pull
   ```

4. Do not fetch datasets via git; download under `/workspace/data` per your procedure and point YAML `path:` keys at absolute paths under that tree.

5. Store benchmark artifacts under **`results/` on the instance**; pull small JSON or plots via `scp` or paste into notes — large artifacts are `.gitignore`d.

## Persistent jobs after SSH drops (tmux)

Long installs/downloads keep running if they live inside **tmux**, not in the bare SSH shell.

```bash
tmux new -s work
# inside: cd repo, bash scripts/vast/install_deps.sh, etc.
# detach (session stays alive): Ctrl+b, then d
# later: tmux attach -t work
# list: tmux ls
```

`tmux` is installed by `scripts/vast/install_deps.sh`. Or once: `apt install -y tmux`.

## Bare Ubuntu on Vast (no `nvcr.io/nvidia/pytorch` image)

If you use a minimal Linux template to avoid image conflicts, install stack from scratch:

```bash
sudo bash scripts/vast/install_deps.sh
```

PyTorch is taken from **CUDA wheels** (`TORCH_INDEX_URL`, default `cu124`). If `torch.cuda.is_available()` is false after install, check `nvidia-smi` and match the wheel index to your driver using [PyTorch install selector](https://pytorch.org/get-started/locally/).

## From chat snippets to repo scripts

While iterating, **one-off command blocks from chat** are fine. When a block stabilizes:

1. Save it under this repo, e.g. `scripts/vast/install_deps.sh`, `scripts/vast/download_crowdhuman_val.sh`.
2. Prefix with `set -euo pipefail` and a short comment describing behavior.
3. On Vast: `bash scripts/vast/install_deps.sh`

That preserves experiment history in git and avoids rerunning long pasted shells.

## Operational notes

- On NVIDIA images, prefer `python3 -m pip install ...` over bare `pip install` so packages bind to the intended interpreter.

- **CrowdHuman → YOLO conversion**: assuming a fixed `1920×1080` canvas for every image yields **wrong normalized boxes**. For credible mAP, read each image’s actual size (e.g. `cv2.imread` → `shape`) and derive `xc, yc, nw, nh` from it.

- **FPS via `model(dummy)`**: Ultralytics’ top-level forward is not always identical to the full inference path. For comparable numbers either call the inner module (`model.model(tensor)`) or `predict()` on ndarray input — pick one strategy and keep it fixed across the study.
