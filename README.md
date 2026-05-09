# Edge: pedestrian detection and tracking (Jetson)

Research codebase: compare pedestrian detectors and detector–tracker stacks, then deploy on NVIDIA Jetson. Cloud (e.g. Vast.ai) for early runs; final FPS and power measurements target the board.

## Layout

| Path | Purpose |
|------|---------|
| `scripts/vast/` | Cloud bootstrap: deps, datasets, CrowdHuman→YOLO, bench FPS, val — see [`scripts/vast/README.md`](scripts/vast/README.md) |
| `configs/datasets/` | Dataset YAML for Ultralytics (e.g. CrowdHuman val) |
| `docs/model_manifest.yaml` | Model inventory for experiments |
| `data/` | Local datasets placeholder (gitignored) |
| `models/` | Local checkpoints (gitignored) |
| `results/runs/` | Run logs |
| `src/` | Planned: C++/PyBind/TensorRT |

## Repo + Vast.ai

**Scripts and configs live in git**; **CrowdHuman, MOT17, `.pt` files stay on the instance disk** (`/workspace`), not in commits (see `.gitignore`). On the machine: one-time `git clone`, then `git pull` before work. Details: [`docs/VAST_WORKFLOW.md`](docs/VAST_WORKFLOW.md).
