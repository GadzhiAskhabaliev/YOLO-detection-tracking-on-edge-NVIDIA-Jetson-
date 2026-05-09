# Edge: pedestrian detection and tracking (Jetson)

Research codebase: compare pedestrian detectors and detector–tracker stacks, then deploy on NVIDIA Jetson. Cloud (e.g. Vast.ai) for early runs; final FPS and power measurements target the board.

**No canned automation yet**: environment and code land incrementally (Vast bootstrap, benchmarks, ONNX, MOT, etc.).

## Layout

| Path | Purpose |
|------|---------|
| `docs/model_manifest.yaml` | Model inventory for experiments; extend when weights are fixed |
| `data/` | Datasets (gitignored) |
| `models/` | Checkpoints (gitignored) |
| `results/runs/` | Run logs when you start saving them |
| `src/` | Planned: C++/PyBind/TensorRT |

The repo tracks only what is actively used; the rest is added as experiments evolve.

## Repo + Vast.ai

**Scripts and configs live in git**; **CrowdHuman, MOT17, `.pt` files stay on the instance disk** (`/workspace`), not in commits (see `.gitignore`). On the machine: one-time `git clone`, then `git pull` before work. Details: [`docs/VAST_WORKFLOW.md`](docs/VAST_WORKFLOW.md).
