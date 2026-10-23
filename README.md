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

См. [конфигурацию и запуск](docs/configuration.md) и [правила логирования](docs/logging.md).