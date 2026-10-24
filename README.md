# TAEI

TAEI организует эксперимент как отдельную папку: рядом лежат его Hydra-конфигурация и обычный Python-скрипт процедуры. Общие компоненты — настройка логирования, метрики и жизненный цикл запуска — находятся в `src/taei/`. Эксперимент не обязан быть классом или заранее обобщённым компонентом.

## Быстрый старт

```bash
uv sync
uv run python experiments/smoke/run.py
uv run python experiments/dummy/run.py metrics.score=0.9
```

`smoke` — синтетическая проверка без загрузки модели, GPU, сети и ClearML; это не бенчмарк Qwen или nano-vLLM. Чтобы включить ClearML для запуска (нужны настроенные учётные данные и доступность сервиса):

```bash
uv run python experiments/smoke/run.py clearml.enabled=true
```

## Устройство

```text
experiments/
├── _template/
│   ├── config.yaml
│   └── run.py
├── dummy/
│   ├── config.yaml
│   └── run.py
└── smoke/
    ├── config.yaml
    └── run.py
src/taei/
├── conf/
│   ├── base.yaml
│   ├── engine/nano_vllm.yaml
│   └── model/qwen3_8b_dense.yaml
├── logging.py
├── metrics.py
└── runtime.py
```

В `experiments/<имя>/run.py` находится уникальная процедура; `config.yaml` хранит её параметры и комментарии к ним. Переиспользуемые настройки и код относятся в `src/taei/`, а не копируются между экспериментами. Не нужно обобщать одноразовую процедуру или вводить обязательный класс `Experiment`.

Скрипт запуска использует Hydra с `config_path='.'`, `config_name='config'`. Локальный YAML наследует общие настройки через `defaults: [base, _self_]`; уникальные шаги процедуры остаются в Python. Общая конфигурация содержит группы `engine` и `model`; их наличие само по себе не загружает модель и не запускает inference.

## Новый эксперимент

```bash
cp -r experiments/_template experiments/my_experiment
```

Отредактируйте `experiments/my_experiment/config.yaml` — имя, описание, параметры и комментарии — и реализуйте процедуру в `run.py`. Запускайте из корня проекта:

```bash
uv run python experiments/my_experiment/run.py
```

Hydra принимает обычные переопределения и multirun, например:

```bash
uv run python experiments/dummy/run.py metrics.score=0.9
uv run python experiments/dummy/run.py --cfg job
uv run python experiments/dummy/run.py -m metrics.score=0.7,0.9
```

Каждый запуск сохраняет разрешённый `config.yaml`, непрерывный `run.log` и архив `logs.zip` в `outputs/<experiment.name>/<каталог-запуска>/`. После успешного выполнения эксперимента там появляется `results.json`; ошибка выполнения или `KeyboardInterrupt` до записи результата не создаёт его, но ошибка поздней финализации ClearML может оставить уже записанный файл. Multirun-каталоги Hydra располагаются внутри каталога эксперимента; Hydra не меняет текущую рабочую директорию (`chdir=false`). Архив можно забрать по SFTP без доступа к ClearML; подробности и границы захвата — в [правилах логирования](docs/logging.md).

## Практический запуск эксперимента

Общий шаблон: из корня проекта запускайте `experiments/<имя>/run.py`; сначала можно посмотреть итоговую Hydra-конфигурацию (`--cfg job`) и выполнить безопасный режим планирования (`mode=plan`). Параметры переопределяются аргументами `ключ=значение`; группы профилей выбираются через синтаксис Hydra `profile@_global_=имя`.

### nano-vLLM на RTX A4000, TP=2

Для этого варианта предусмотрен отдельный профиль `rtx_a4000_tp2`: он задаёт TP=2 и `enforce_eager=true`, а результаты пишет в отдельное дерево `outputs/nano_vllm_dataset_rtx_a4000_tp2/`. Это аппаратный производный профиль, а не исходный пилот TP=4. Проверьте композицию и план без запуска инференса:

```bash
uv run --locked --extra inference python experiments/nano_vllm_dataset/run.py profile@_global_=rtx_a4000_tp2 --cfg job
uv run --locked --extra inference python experiments/nano_vllm_dataset/run.py profile@_global_=rtx_a4000_tp2 mode=plan
```

На машине выполнения установите inference-зависимости и заранее выберите две совместимые RTX A4000. Например, если именно эти устройства соответствуют индексам 1 и 2 в перечислении PCI_BUS_ID:

```bash
uv sync --locked --extra inference
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2 uv run --locked --extra inference python experiments/nano_vllm_dataset/run.py profile@_global_=rtx_a4000_tp2 mode=preflight
```

Индексы `1,2` — только пример для конкретной машины, не универсальный выбор GPU. Preflight проверяет готовность среды и локальной модели, но сам по себе не загружает модель и не доказывает успешный инференс. Если модельных файлов нет, отдельная подготовка может скачать настроенный snapshot из Hugging Face; это сетевой этап вне измеряемых запусков:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2 uv run --locked --extra inference python experiments/nano_vllm_dataset/run.py profile@_global_=rtx_a4000_tp2 mode=download
```

После подготовки повторите `mode=preflight`. Для проверки одного реального запуска задайте `dataset.run_limit=1`; для полного плана — `dataset.run_limit=24`. Команда полного запуска с выбранными выше GPU:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2 uv run --locked --extra inference python experiments/nano_vllm_dataset/run.py profile@_global_=rtx_a4000_tp2 mode=run dataset.run_limit=24
```

Каждая попытка ограничена таймаутом 1800 секунд; при ошибке выполнения серия останавливается. Артефакты попыток и сводка записываются под `outputs/nano_vllm_dataset_rtx_a4000_tp2/`. Успешное завершение генерации отдельно от статуса приёмки QC: наличие выполненного запуска не гарантирует, что датасет принят. Отчёт калибровки накладных расходов можно передать через `dataset.calibration_report=путь`; он необязателен для команды запуска.

См. [конфигурацию и запуск](docs/configuration.md) и [правила логирования](docs/logging.md).
