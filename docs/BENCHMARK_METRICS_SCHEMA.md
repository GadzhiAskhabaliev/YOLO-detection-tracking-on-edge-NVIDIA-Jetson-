# Единый контракт метрик (`results/runs/*.json`)

Все модели (MMDetection, FairMOT, YOLO-семейство, FreeYOLO, PeopleNet, будущий ONNX/MMDeploy и т.д.) попадают в **`results/benchmark_summary.md`** и README только если JSON прогона следует **одним и тем же ключам** ниже. Конкретный фреймворк указывается в **`backend`** — это не «истина от Ultralytics», а общая таблица для сравнения при **явных определениях** метрик и в **`notes`**.

## Верхний уровень JSON

| Поле | Обязательно | Описание |
|------|-------------|----------|
| `model` | да | Логическое имя строки в таблице (например `fcos_r50_caffe_fpn_gn_ch`, `fairmot_dla34`, `yolov8s_crowdhuman_proxy`). |
| `date` | да | UTC ISO8601, задаётся при сохранении. |
| `weights` | да | Абсолютный или репо-путь к чекпойнту на машине прогона. |
| `weights_hub` | нет | Идентификатор в registry / Model Zoo / URL релиза (что удобнее воспроизвести). |
| `hardware` | да | Строка GPU/CPU (например из `nvidia-smi`). |
| `backend` | **рекомендуется** | Короткий тег: `mmdet`, `fairmot`, `freeyolo`, `ultralytics_yolo`, `tao_peoplenet`, `onnx_runtime`, … |
| `batch_size`, `imgsz` | да | Как в прогоне; для честного FPS обычно `batch_size=1`. |
| `group`, `detector_id`, `detector_label` | нет | Слоты отчётов (например группа B). |
| `metrics` | да | Словарь числовых метрик (см. ниже). |
| `tracking` | нет | MOTA, IDF1, HOTA и т.д. для трекеров / joint. |
| `tensorrt` | нет | `engine_exists`, опционально `fps_fp16`. |
| `notes` | нет | Обязательно сюда: датасет, сплит, пороги, **как именно посчитаны** precision/recall/FPS, если они отличаются от канона. |

## Ключи `metrics` (одна строка таблицы — одни имена полей)

Имена **фиксированы**, чтобы FCOS/SSD/YOLOX/FairMOT/YOLOv8s-CH/FreeYOLO попадали в одну сводку.

| Ключ | Смысл (канон) |
|------|----------------|
| `mAP50` | AP при IoU=0.50 на выбранном **фиксированном** валидационном сплите (часто CrowdHuman val в этом репо). Другой датасет — написать в `notes`. |
| `mAP50-95` | AP при IoU=0.50:0.95 там же; если фреймворк не даёт — оставить пустым и пояснить. |
| `precision` | Средняя precision по классу/боксам **в том протоколе, который вы зафиксировали** (например из отчёта val MMDet). Не смешивать с «AP@0.5» без пометки. |
| `recall` | Аналогично recall из того же протокола. Если заполняете **COCO AR** (Average Recall) — явно напишите в `notes` строку из summarize и диапазон IoU / maxDets. |
| `inference_time_ms` | Среднее время **полного инференса одного кадра** в миллисекундах: препроцесс + модель + постпроцесс (decode/NMS), в том же режиме, что и `fps_predict`. |
| `fps_forward` | Пропускная способность **узкой «нейросетевой» части**: вход уже в форме тензора после типичного препроцесса, **без** тяжёлого постпроцесса на CPU там, где это разумно отделить (MMDet: обычно `model.test_step` / замер только backbone+neck+head до merge_heads — зафиксируйте в `notes`; FreeYOLO: режим `no_decode`; у простых обёрток допускается совпадение с predict, если разделить нельзя — указать в `notes`). |
| `fps_predict` | Сквозной путь как в проде/демо: препроцесс детектора + инференс + декод/NMS на dummy или репрезентативном изображении, **batch=1**, то же железо. |
| `forward_time_ms_mean` | \(`1000 / fps_forward`\), если считаете удобным дублировать явным числом (не обязательно). |
| `inference_time_ms_predict` | \(`1000 / fps_predict`\); если задано `inference_time_ms`, они должны быть согласованы в пределах округления. |

Дополнительные поля (`eval_throughput_fps`, `eval_wall_seconds`, …) не участвуют в главном сравнении «кадр/сек детектора», но могут хранить **end-to-end** время скрипта eval — всегда с пояснением в `notes`.

## По типам моделей

### MMDetection (FCOS, SSD, YOLOX, …)

1. Один и тот же конфиг + checkpoint из Model Zoo; в `weights_hub` — строка вида `fcos_r50_caffe_fpn_gn-head_1x_coco` или полный идентификатор.
2. Качество на CrowdHuman: отдельный config/test или конвертация датасета под CocoDataset — зафиксировать в `notes`.
3. FPS: предпочтительно общий скрипт-обёртка (warmup + N итераций, `torch.cuda.synchronize()`), заполняющая **`fps_predict`** и **`inference_time_ms`**. **`fps_forward`** — по согласованному правилу из `notes` (например только `extract_feat` + heads до `get_bboxes`).
4. `backend`: `mmdet`.

### ONNX / MMDeploy / TensorRT

Пока в таблице основной путь — `.pth` / native. После конвертации: новый прогон с тем же каноном; в `notes` — версия MMDeploy, opset, precision (FP16/INT8). `backend`: `onnx_runtime` / `tensorrt` и т.д.

### FairMOT и joint «детектор+трекер»

- Детекционные метрики на CrowdHuman (если есть отдельная голова/прогон): те же ключи `metrics`.
- Трекинг: **`tracking`** (MOTA, IDF1, HOTA, …), датасет и сплит в `notes`.
- FPS joint: либо отдельное поле в `notes`, либо договориться и записать как **`fps_predict`** для полного пайплайна *с трекингом*, явно написав это — иначе не сравнивать с чистыми детекторами.

### YOLOv8s-CrowdHuman как прокси для joint

Использовать те же ключи; в `notes`: «proxy для joint, без трекера». `backend`: например `ultralytics_yolo`.

## Шаблон для ручного merge

См. [`docs/templates/benchmark_run_canonical.example.json`](templates/benchmark_run_canonical.example.json).

После редактирования JSON:

```bash
python3 scripts/bench_runner.py --merge-json results/runs/<your_run>.json --patch-json docs/templates/your_patch.json
```

Перегенерация **`benchmark_summary.md`** и таблицы в README выполняется автоматически внутри `save_result` / `merge_run_json`.
