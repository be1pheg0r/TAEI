<div align="center">

# TAEI

**Воспроизводимые эксперименты · nano-vLLM · анализ журналов**

![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Hydra](https://img.shields.io/badge/config-Hydra-008080)
![15 tests](https://img.shields.io/badge/tests-15-4C956C)

[Быстрый старт](#быстрый-старт) · [Эксперименты](#эксперименты) · [Log Explorer](#log-explorer) · [Проверки](#проверки)

</div>

Каждый эксперимент — обычный Python-сценарий и Hydra-конфигурация в отдельной папке. Общий runtime управляет запуском, метриками и журналами; Log Explorer помогает исследовать полученные архивы.

## Быстрый старт

Требуются **Python 3.12** и **uv**. Из корня репозитория:

```bash
uv sync --locked
uv run --locked python experiments/smoke/run.py
```

Smoke работает без GPU, весов модели и сетевых запросов. Результаты находятся в `outputs/qwen3_8b_dense_smoke/<запуск>/`: `config.yaml`, `results.json`, `run.log`, `hydra.log`, `logs.zip`. Архив содержит журналы, а не полный набор артефактов.

## Эксперименты

| Сценарий | Назначение |
|---|---|
| [`smoke`](experiments/smoke/) / [`dummy`](experiments/dummy/) | Быстрая CPU-проверка и пример переопределения метрик |
| [`nano_vllm_dataset`](experiments/nano_vllm_dataset/) | Парные нагрузки для исследования prefix cache и планировщиков |
| [`Qwen3-0.6B`](experiments/nano_vllm_qwen3_0_6b/) / [`Qwen3-8B`](experiments/nano_vllm_qwen3_8b/) | Конфигурации отдельных моделей |
| [`long_context`](experiments/nano_vllm_qwen3_8b_long_context/) | Длинный контекст, диагностика и request scaling |

Построить план без загрузки модели и inference:

```bash
uv run --locked python experiments/nano_vllm_qwen3_8b/run.py mode=plan
```

Настройки находятся рядом со сценарием в `config.yaml`; Hydra принимает `ключ=значение`, `--cfg job` и `-m`. Для реального inference нужны совместимые GPU, локальная модель и зависимости `inference`:

```bash
uv sync --locked --extra inference
uv run --locked --extra inference python experiments/nano_vllm_dataset/run.py mode=run download.enabled=false dataset.run_limit=1
```

Выберите подходящие `dataset.model_path` и аппаратный профиль в конфигурации. При `download.enabled=true` отсутствующая модель может скачиваться. `mode=preflight` отключён и не проверяет готовность среды. Для фонового запуска добавьте `--detach`; состояние доступно через `python -m taei.launch status <launch.json>`.

## Log Explorer

Отдельное приложение: **FastAPI + React + ECharts**. Принимает ZIP/RAR и показывает метрики, таблицы, графики и журналы; API — `POST /api/analyze`, проверка процесса — `GET /api/health`.

```bash
docker compose -f tools/log-explorer/compose.yaml up -d --build
```

[Compose](tools/log-explorer/compose.yaml) ожидает внешнюю сеть `outline-stack_edge`; для своей установки адаптируйте её имя. Сервис слушает `127.0.0.1:8091`; HTTPS настраивается отдельно. Данные выдаются как предпросмотр, а не полная выгрузка архива. Не загружайте секреты и персональные данные.

## Разработка

Скопируйте [`experiments/_template/`](experiments/_template/), измените `config.yaml` и реализуйте процедуру в `run.py`. Общие компоненты размещаются в [`src/taei/`](src/taei/); параметры процедуры — в YAML. Процедура возвращает JSON-сериализуемые метрики, runtime сохраняет итог один раз.

### Логирование

Используйте `taei.logging`: `configure_logging(config=cfg)` вызывается после композиции Hydra, `logging.markup=false` отключает Rich-разметку. Приоритет уровня: явный `level` → `logging.level` → `TAEI_LOG_LEVEL` → `INFO`. Для прогресса используйте `experiment_progress`; жизненный цикл эксперимента остаётся в runtime, секреты не попадают в конфигурационные дампы и журналы. Захват начинается при входе в runtime; архивирование выполняется при штатном завершении, перехваченной ошибке или прерывании, но не гарантируется после `SIGKILL`.

## Проверки

**15 приоритетных сценариев:** 12 для runtime, CLI, нагрузок, артефактов и QC; 3 для API, точности данных и защиты загрузок Log Explorer.

```bash
uv run --locked pytest tests -q
uv run --locked ruff check .
uv run --locked isort --check-only .
```

<details>
<summary><strong>Три API-теста Log Explorer — отдельное окружение</strong></summary>

```bash
uv venv tools/log-explorer/backend/.venv --python 3.12
uv pip install --python tools/log-explorer/backend/.venv/bin/python -r tools/log-explorer/backend/requirements.txt pytest httpx httpx2
cd tools/log-explorer/backend
.venv/bin/python -m pytest tests -q
```

Зависимости backend не входят в корневой `uv.lock` и задаются диапазонами версий.

</details>

`uv.lock` фиксирует зависимости ядра, nano-vLLM закреплён на Git revision. Тесты не требуют GPU или сетевого скачивания и не заменяют проверку реального inference и браузерного интерфейса. Seed не гарантирует побитовую идентичность между разными GPU и драйверами.
