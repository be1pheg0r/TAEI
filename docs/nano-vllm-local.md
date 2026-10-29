# Локальный сбор Qwen3-0.6B на одной GPU

Сетка для модели времени nano-vLLM на RTX 4070 Laptop (8 ГБ, TP=1, eager).
Код: [`src/taei/nano_dataset/local_grid.py`](../src/taei/nano_dataset/local_grid.py),
запуск: [`experiments/nano_vllm_qwen3_0_6b_local/`](../experiments/nano_vllm_qwen3_0_6b_local/).
Формат попытки тот же, что на сервере, поэтому `taei.nano_time.data.load` читает
её без изменений.

## Окружение

Locked-extra `inference` собран под CUDA 13, а ему нужен драйвер ≥ 580. На
драйвере 535 ставятся те же версии torch и flash-attn, собранные под CUDA 12.6:

```bash
uv venv .venv-nano --python 3.12
uv pip install --python .venv-nano --index-strategy unsafe-best-match -r requirements-nano-local.txt
uv pip install --python .venv-nano --no-deps -e .
.venv-nano/bin/python experiments/nano_vllm_qwen3_0_6b_local/run.py mode=download
```

## Машина

```bash
sudo systemctl stop ollama
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sudo nvidia-smi -lgc 2100,2100
```

Частота SM фиксируется ниже минимума, до которого она проседает без фиксации,
с запасом ~5%. Его даёт фаза `stress`: на этом ноутбуке под нагрузкой частота
держится на 2205–2400 МГц и упирается в ограничение по мощности. Раннер
закрепляет worker'ы на P-ядрах (`dataset.cpu_affinity`).

Даже на фиксированной частоте прошивка ноутбука иногда на десятки секунд
срезает мощность GPU до ~40 Вт, и частота падает до 500–1500 МГц.
Телеметрия снимается раз в 0,25 с. Если в попытке частота хоть раз оказалась
ниже `dataset.locked_sm_clock_mhz`, попытка получает статус `throttled`, и
загрузчик её не берёт. После такой попытки worker ждёт 30 с, а в конце фазы
она перезапускается (до `throttle_reruns` раз).

После сбора вернуть настройки:

```bash
sudo nvidia-smi -rgc
sudo systemctl start ollama
echo powersave | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

## Фазы

| Фаза | Что | Зачем |
|---|---|---|
| `smoke` | один реперный прогон | число KV-блоков → `dataset.kv_blocks` |
| `stress` | самые тяжёлые точки, частота не зафиксирована | выбор частоты |
| `A` | REF, S1 (prefill), S2 (decode), S0 (серверная матрица) | подгонка |
| `B` | S3 (смеси по доле prefill), S4 (поток Пуассона) | проверка |
| `C_smoke` | одна точка S5 (N = 4, прогрев 5, замер 20) | проверка закрытого цикла |
| `C` | REF, S5 (закрытый цикл по конкурентности) | перенос |
| `D_smoke` | REF, по точке S6 и S7, короткий S8 с лимитом окна 5 с | проверка новых полей и лимита |
| `D` | S6, S7 (decode), S8 (закрытый цикл на W1–W4), добор S5 | train и проверка |

```bash
P=.venv-nano/bin/python
$P experiments/nano_vllm_qwen3_0_6b_local/run.py mode=run dataset.phase=A dataset.kv_blocks=186
uv run --extra analysis python experiments/nano_vllm_qwen3_0_6b_local/fit_phase_a.py <runs фазы A>
$P experiments/nano_vllm_qwen3_0_6b_local/run.py mode=run dataset.phase=B dataset.kv_blocks=186 \
  'dataset.coef={a: ..., b: ..., c: ..., d0: ..., d1: ..., d2: ...}'
$P experiments/nano_vllm_qwen3_0_6b_local/run.py mode=run dataset.phase=C dataset.kv_blocks=186 \
  dataset.telemetry_interval_s=1.0 dataset.compact_cache_events=true \
  hydra.run.dir=data/qwen-3-06b-local/phase_C
# Фаза D: план целиком, затем блоки по одному в порядке приоритета
# (S6, S7, S8-W1, S8-W2, S8-W3, S8-W4, S5); коэффициенты — фазы A (как в B).
D="dataset.phase=D dataset.kv_blocks=186 dataset.telemetry_interval_s=1.0 \
  dataset.compact_cache_events=true dataset.machine_checks=true dataset.ref_drift_max=0.03 \
  dataset.engine_reference=data/qwen-3-06b-local/phase_C/plan.json \
  dataset.output_root=data/qwen-3-06b-local/phase_D/runs"
$P experiments/nano_vllm_qwen3_0_6b_local/run.py mode=plan $D 'dataset.coef={...}' \
  hydra.run.dir=data/qwen-3-06b-local/phase_D
$P experiments/nano_vllm_qwen3_0_6b_local/run.py mode=run $D 'dataset.coef={...}' \
  dataset.block=S6 hydra.run.dir=data/qwen-3-06b-local/phase_D/S6
```

### Фаза C: S5, закрытый цикл

- **Что такое закрытый цикл.** В системе всё время ровно N запросов:
  - сначала подаются N запросов;
  - после каждого завершения подаётся следующий из пула;
  - после 20 прогревочных и 200 измеренных завершений (60 при N = 1) подача прекращается, хвост дорабатывает.
- **Пул запросов.** Один пул на зерно (42, 43, 44) — смесь S4. Workload точки — первые N + 219 запросов пула, поэтому точки отличаются только N.
- **Отметки в трассе:**
  - у запроса `closed_loop_role` (warmup / measure / tail);
  - в `runs.json → closed_loop` — окно замера (от завершения 20-го до завершения 220-го).
- **Сетка.**
  - upstream_dynamic без кэша и с кэшем: N = 1, 2, 4, 6, 8, 12, 16, 24, 32;
  - fixed_batch без кэша: N = 2–16.

  Не снимались fixed_batch с кэшем и N = 48: бюджет 2,5 ч, оценка до запуска 2,05 ч.
- **Оценка времени.** `local_grid.s5_estimate_seconds` — по задержке одиночного запроса и потолку из S4. Если оценка больше бюджета, план сначала убирает N = 32.
- **REF.** Каждые 6 попыток (около 10 минут), плюс пять в начале и один в конце.
- **Диск.** `compact_cache_events` после каждой порции попыток:
  - пишет `kv_blocks.jsonl` — по шагу число занятых блоков KV;
  - сжимает `cache_events.jsonl` в `.gz`. Загрузчик модели времени его не читает.

### Фаза D: decode до полного KV, общий префикс, закрытый цикл на новых нагрузках

- **S6** (train) — B запросов разного контента разом, кэш выключен, O = 40:
  4 decode-шага прогрева и 35 в замер.
  - Сетка: B от 1 до 96 × Σℓ 8k–44k и «max» — наибольшая Σℓ, при которой все
    блоки с ростом на O влезают в 0,95 KV.
  - Подсерия mixed: длины от 0,25·ℓ̄ до 1,75·ℓ̄; 44k не влезает из-за неполных
    блоков и заменяется наибольшей Σℓ той же формы.
- **S7** (train) — то же с общим префиксом: кэш включён, R 0,25–0,9, 1 или 4
  семьи, донор на семью. При R = 0 — пара «кэш вкл / выкл».
- **S8** (только проверка) — закрытый цикл как S5.
  - Замер — 200 завершений или 3 мин окна, но не меньше 60. Неподанный остаток
    пула в трассу не пишется, в `closed_loop` — `submitted` и фактический
    `measure`.
  - Нагрузки W1–W4 (`local_grid._s8_pool`). Сетка N — по колену KV: не меньше
    4 размеров до него и 2 после.
- **Добор S5** — те же пулы `_s5_pool`, продлённые до 320 запросов.
- **Новые поля шага:**
  - `context_tokens_sum`;
  - `kv_unique_blocks` — блоки батча, общий префикс один раз;
  - `kv_used_blocks`, `kv_capacity_blocks`, `kv_fill` после шага;
  - `preempted_count`.
- **Контроль сбора:**
  - перед каждой порцией — нет чужих процессов на GPU, питание от сети, профиль
    `performance`;
  - REF закрывает порцию; дрейф больше `ref_drift_max` останавливает сбор;
  - перезапуск с `dataset.resume_after=<run_order последнего хорошего REF>`;
  - после блока — `block-check.json`.

## Устройство

- **Процессы.** Один worker-процесс загружает модель один раз, прогревает её
  (prefill на 16k токенов и полный decode-батч) и проводит до
  `attempts_per_process` попыток. Перед каждой попыткой сбрасываются блоки и
  кэш, а генератор случайных чисел заново инициализируется зерном. После
  упавшей попытки остаток списка уходит в новый процесс.
- **Ёмкость KV.** Точка, которой не хватает KV (общие префиксы плюс все запросы
  разом больше `kv_fill` кэша), выбрасывается ещё в плане и попадает в
  `plan.json → dropped`. В S4 очередь и вытеснение допустимы.
- **Неполный вывод.** Попытка, где хотя бы один запрос выдал не все запрошенные
  токены, помечается failed.
- **Состояние машины.** В `runs.json → machine_state` пишутся диапазон частоты
  SM и маски причин троттлинга.

## Сбор 2026-10-28

Условия: `gpu_memory_utilization` 0.95 (186 блоков KV), `max_num_seqs` 256,
частота SM 2100 МГц.

| Фаза | Попыток | Prefill-шагов | Decode-шагов | Запросов | Объём |
|---|---|---|---|---|---|
| A (REF 48, S1 840, S2 234, S0 48) | 1170 | 1 545 | 86 726 | 8 418 | 3,7 ГБ |
| B (REF 49, S3 1014, S4 84) | 1147 | 13 170 | 602 098 | 37 123 | 14 ГБ |
| C (REF 18, S5 72) | 90 | 10 589 | 388 421 | 15 990 | 4,3 ГБ |

- **Итог.** Все попытки success, ни одной `throttled`, потерянных токенов
  нет. Частота SM во всех попытках ровно 2100 МГц.
- **Фаза C.** Сбор занял 1 ч 57 мин при оценке 2,05 ч. Все 90 попыток
  success, троттлинга нет. REF 6,187 с, дрейф от фаз A/B 0,1%. Ровно N
  запросов в системе — не меньше 99,9% окна замера во всех точках.
- **Шум и дрейф.** Время реперного прогона: 6,181 с в фазе A и 6,186 с в
  фазе B; std/mean 0,7%.
- **Выбросы decode-шагов.** Изредка (≈ 0,03–0,06% шагов) decode-шаг длится
  0,13–0,48 с вместо ~10–30 мс. Самые длинные (0,35–0,48 с) — первый
  decode-шаг прогона сразу после prefill при B = 1; так же ведут себя
  prefill-шаги S1 с R ≈ 1, где считается один токен. Остальные выбросы стоят в
  случайном месте длинных прогонов. Причина не выяснена: первое похоже на
  холодный запуск ядра для маленькой формы, второе — на сборку мусора Python в
  сборщике трассы.
