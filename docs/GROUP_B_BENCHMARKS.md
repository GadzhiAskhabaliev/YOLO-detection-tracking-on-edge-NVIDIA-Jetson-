# Группа B: прогоны и отчёты

Манифест моделей: [`group_b_pedestrian_detectors.yaml`](group_b_pedestrian_detectors.yaml).

### Куда попадают метрики и логи

- **Сводка по всем прогонам** строится из `results/runs/*.json` → **`results/benchmark_summary.md`** и таблица в **`README.md`** (через `bench_runner.save_result()`).
- **Сырой лог FreeYOLO** (`eval.py`) сохраняется в **`results/logs/freeyolo_<variant>_<UTC>.log`** (можно коммитить в git).
- Для **`bench_runner`** / полного `run_group_b_benchmarks.sh` при желании сохраняйте весь stdout одной командой, например:
  ```bash
  bash scripts/run_group_b_benchmarks.sh 2>&1 | tee "results/logs/group_b_run_$(date -u +%Y%m%dT%H%M%SZ).log"
  ```

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

#### Другой размер FreeYOLO на том же CrowdHuman val (tiny / large / …)

Пример **Tiny** (отдельный `.pt`, отдельная строка в таблице `freeyolo_ch_tiny`):

```bash
FREEYOLO_VARIANT=yolo_free_tiny \
FREEYOLO_WEIGHT_URL=https://github.com/yjh0410/FreeYOLO/releases/download/weight/yolo_free_tiny_ch.pth \
FREEYOLO_WEIGHT_PATH="${MODEL_DIR:-/workspace/models}/yolo_free_tiny_ch.pth" \
bash scripts/group_b/run_freeyolo_crowdhuman.sh
```

**Large:** `-v yolo_free_large`, файл `yolo_free_large_ch.pth` (URL в README FreeYOLO releases).

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

### FreeYOLO и NumPy 2.x / `np.int`

Скрипт делает два слоя защиты: **`numpy<2` в venv** и **патч исходников** (`patch_freeyolo_numpy_aliases.py`: `np.int`→`int` и т.д.), если всё же подтянулся NumPy 2.x.

Ручная переустановка только NumPy 1.x:

```bash
source /workspace/group_b/venv_freeyolo/bin/activate
pip install "numpy>=1.23,<2" --force-reinstall --no-deps
```

### FreeYOLO и PyTorch 2.6+

Если `eval.py` падает с `WeightsUnpickler error` / `weights_only`, после `git pull` скрипт `run_freeyolo_crowdhuman.sh` сам патчит `FreeYOLO/utils/misc.py`. На уже склонированном дереве без обновления репо:

```bash
python3 scripts/group_b/patch_freeyolo_torch_load.py --freeyolo-home /workspace/group_b/FreeYOLO
```

### Согласование метрик (FreeYOLO и любые другие бэкенды)

Общий контракт полей JSON для **всех** моделей в сводке (MMDet, FairMOT, YOLO-семейство, FreeYOLO, ONNX/MMDeploy и т.д.): **[`BENCHMARK_METRICS_SCHEMA.md`](BENCHMARK_METRICS_SCHEMA.md)**.

После `eval.py` скрипт **`freeyolo_save_run.py`** вызывает **`scripts/group_b/freeyolo_speed_bench.py`** (нужен `--freeyolo-home` или `FREEYOLO_HOME`), чтобы заполнить канонические **`fps_forward`** / **`fps_predict`** / **`inference_time_ms`**. У FreeYOLO: forward — проход с `no_decode=True`; predict — `ValTransforms` + полный decode/NMS (`no_decode=False`). **`eval_throughput_fps`** — только wall-clock всего `eval.py` (включая COCOeval на CPU), для честного FPS не использовать как основной столбец.

Если из лога доступна строка COCO **Average Recall (AR)** (`maxDets=100`), она может быть записана в **`recall`** — в `notes` обязательно указать, что это AR, а не «mean recall» из другого отчёта val.

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

Сопоставимое качество на одном сплите: выгрузить bbox-предикты под тот же `val.json`, что и GT, и прогнать **`scripts/eval_coco_predictions.py`** (см. **[`BENCHMARK_METRICS_SCHEMA.md`](BENCHMARK_METRICS_SCHEMA.md)**). В сводке всегда смотрите `notes` и источник FPS.

---

## Замыкание исследования: что уже есть и что осталось

**Уже зафиксировано в `results/runs/` (можно строить таблицу и графики):**

| Слот | Модель | JSON |
|------|--------|------|
| 6 | YOLOv8n-CrowdHuman | `yolov8n_crowdhuman_*.json` |
| 7 | FreeYOLO (variants) | `freeyolo_ch_tiny_*.json`, `freeyolo_yolox_mot17_*.json` |

**Нет прогона в репо — финальные mAP/FPS появятся только после вашего шага в их кодовой базе:**

| Слот | Модель | Действие |
|------|--------|----------|
| 4 | CrowdDet | Инференс на том же CrowdHuman val → dt-json → `eval_coco_predictions.py` → `bench_runner --merge-json` |
| 5 | Pedestron | То же |
| 8 | PeopleNet (TAO) | Инференс / экспорт под вашим протоколом → dt-json или канон `metrics` + FPS → merge |

**MMDet (FCOS / SSD / YOLOX и т.д., если сравниваете отдельно от слотов B):** не в манифесте YAML, но тот же контракт — предикты + **`eval_coco_predictions.py`** + свой `results/runs/<slug>.json`.

### Чеклист до «итоговых резов» по каждой недостающей модели

1. Зафиксировать **версию** фреймворка и **команду** воспроизведения (commit / Docker image / pip freeze фрагмент).
2. Убедиться, что GT — **тот же** `annotations/val.json` (те же `image_id`), что для уже замеренных моделей (bridge CrowdHuman → COCO).
3. Прогнать инференс, сохранить **список** `{image_id, category_id, bbox [xywh], score}` (строго под канон из **`BENCHMARK_METRICS_SCHEMA.md`**).
4. Локально в этом репо:  
   `python3 scripts/eval_coco_predictions.py --gt-json … --dt-json … --strict --out-patch-json /tmp/p.json`
5. Создать или дополнить **`results/runs/<bench_slug>_….json`**, затем  
   `python3 scripts/bench_runner.py --merge-json … --patch-json /tmp/p.json`  
   и при необходимости вторым патчем дописать **FPS** (`fps_predict`, `inference_time_ms`, …).
6. Для группы B: в патче **`group`: `"B"`**, **`detector_id`**: `4`, `5` или `8` по [`group_b_pedestrian_detectors.yaml`](group_b_pedestrian_detectors.yaml).
7. `python3 scripts/plot_group_b_results.py` — обновить **`results/group_b_report.md`** и фигуры.

Пока пункты 1–7 для слота не выполнены, в таблице по этой модели будет пробел — это ожидаемо: репозиторий даёт **единую математику и формат**, но не ставит чужие стеки.
