# Группа B: прогоны и отчёты

Манифест моделей: [`group_b_pedestrian_detectors.yaml`](group_b_pedestrian_detectors.yaml).

## Автоматический прогон из этого репозитория

### Один скрипт (YOLOv8n + FreeYOLO + памятки по остальным)

```bash
bash scripts/run_group_b_benchmarks.sh
```

- **№6 YOLOv8n-CrowdHuman** — `bench_runner.py` (forward / predict / val).
- **№7 FreeYOLO** — отдельный venv под `/workspace/group_b`, `eval.py` на CrowdHuman val, затем запись в `results/runs/` (долго на полном val).
- **№4 CrowdDet, №5 Pedestron, №8 PeopleNet** — в этом прогоне выводятся шаги и ссылки; полный eval только в своих окружениях (Docker / mmcv / NGC).

Отключить всё кроме YOLOv8: `GROUP_B_EXTRA_MODELS=0 bash scripts/run_group_b_benchmarks.sh`  
Отключить только FreeYOLO: `GROUP_B_FREEYOLO=0 bash scripts/group_b/run_remaining_models.sh`

### Только YOLOv8 (как раньше)

Через `scripts/bench_runner.py` — см. корневой `README.md`; для группы B используйте `--group B --detector-id 6`.

Переменные окружения (опционально):

| Переменная | По умолчанию |
|------------|----------------|
| `MODEL_DIR` | `/workspace/models` |
| `DATA_YAML` | `configs/datasets/crowdhuman_val.yaml` |

## CrowdDet, Pedestron, PeopleNet (в основном вне этого репо)

У них **другие кодовые базы** (Docker / MMDetection / TAO). После своего eval сведите результат в JSON. Общий алгоритм:

1. Прогоните детекцию и замер FPS **в их окружении** на том же железе (или зафиксируйте GPU в JSON).
2. Сведите метрики в один объект в формате `results/runs/*.json` (поля `metrics`, `tracking`, `hardware`, …).
3. Пометьте прогон меткой группы (**`group`: `"B"`**, **`detector_id`**: `4`…`8`), чтобы `plot_group_b_results.py` сопоставил строку с манифестом.

Через CLI (после того как JSON уже создан — вручную или другим пайплайном):

```bash
python3 scripts/bench_runner.py \
  --merge-json results/runs/peoplenet_resnet34_2026.json \
  --patch-json docs/templates/group_b_meta_patch.example.json
```

Либо сохраните свой JSON-патч и передайте его как `--patch-json` (поля `group`, `detector_id`, `detector_label` и при необходимости `metrics`, `notes`).

## Визуализация группы B

После появления JSON в `results/runs/`:

```bash
pip install matplotlib pyyaml   # если ещё не стоят (см. install_deps.sh)
python3 scripts/plot_group_b_results.py
```

Выход:

- `results/figures/group_b_scatter_map_fps.png` — mAP50 vs FPS (predict или forward).
- `results/figures/group_b_map50_bars.png` — столбцы mAP50 по слотам группы B.
- `results/group_b_report.md` — таблица + статус «есть прогон / нет».

Сравнение по **одному и тому же валидационному протоколу** для всех RCNN/YOLOX моделей на CrowdHuman возможно только после **единого eval-скрипта** (отдельная задача); пока в отчёте явно смотрите поле `notes` и источник метрик.
