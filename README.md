# Edge: pedestrian detection and tracking (Jetson)

Research codebase: compare pedestrian detectors and detector–tracker stacks, then deploy on NVIDIA Jetson. Cloud (e.g. Vast.ai) for early runs; final FPS and power measurements target the board.

## Layout

| Path | Purpose |
|------|---------|
| `scripts/vast/` | Cloud bootstrap: deps, datasets, CrowdHuman→YOLO, bench FPS, val — see [`scripts/vast/README.md`](scripts/vast/README.md) |
| `configs/datasets/` | Dataset YAML for Ultralytics (e.g. CrowdHuman val) |
| `docs/model_manifest.yaml` | Model inventory for experiments |
| `docs/group_b_pedestrian_detectors.yaml` | Группа B: CrowdDet, Pedestron, YOLOv8n-CH, FreeYOLO, PeopleNet |
| `docs/GROUP_B_BENCHMARKS.md` | Как прогонять группу B и строить графики |
| `data/` | Local datasets placeholder (gitignored) |
| `models/` | Local checkpoints (gitignored) |
| `results/runs/` | Benchmark run JSON |
| `scripts/bench_runner.py` | Unified bench + README / summary refresh |
| `scripts/generate_comparison_table.py` | mAP50 vs FPS comparison → `results/model_comparison.md` |
| `src/` | Planned: C++/PyBind/TensorRT |

## Benchmark Results

После прогона `scripts/bench_runner.py` таблица ниже обновляется автоматически (между HTML-комментариями). Детальный лог: [`results/benchmark_summary.md`](results/benchmark_summary.md).

<!-- TABLE_START -->

*(нет сохранённых прогонов — выполните `bench_runner.py` на машине с GPU / данными)*

<!-- TABLE_END -->

### Команды

Полный прогон (forward + predict + val на CrowdHuman YAML из конфига):

```bash
python3 scripts/bench_runner.py \
  --model-name yolov8n_crowdhuman \
  --weights /workspace/models/yolov8n_crowdhuman.pt \
  --weights-hub yakhyo/yolov8-crowdhuman \
  --bench-mode all \
  --data-yaml configs/datasets/crowdhuman_val.yaml
```

Только predict FPS через отдельный скрипт, но с записью в тот же формат `results/runs/`:

```bash
python3 scripts/vast/bench_yolo_fps.py \
  --weights /workspace/models/yolov8n_crowdhuman.pt \
  --record-model-name yolov8n_crowdhuman \
  --weights-hub yakhyo/yolov8-crowdhuman
```

Таблица сравнения моделей (mAP50 vs FPS):

```bash
python3 scripts/generate_comparison_table.py
```

Добавить метрики трекинга к уже сохранённому JSON (`--model-name` и `--weights` не нужны):

```bash
python3 scripts/bench_runner.py \
  --merge-json results/runs/yolov8n_crowdhuman_2026-05-09T120000Z.json \
  --tracking-json '{"mot17_seq":"MOT17-02","MOTA":0.68,"HOTA":0.52,"IDF1":0.61}'
```

## Repo + Vast.ai

**Scripts and configs live in git**; **CrowdHuman, MOT17, `.pt` files stay on the instance disk** (`/workspace`), not in commits (see `.gitignore`). On the machine: one-time `git clone`, then `git pull` before work. Details: [`docs/VAST_WORKFLOW.md`](docs/VAST_WORKFLOW.md).
