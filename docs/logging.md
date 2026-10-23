# Logging guidance for agents

Use `taei.logging.configure_logging` for application logging. Do not configure a
second Loguru sink or create a competing project logging wrapper.

- Configure once at the application entrypoint, after Hydra has composed its
  `DictConfig` and before emitting application logs.
- Keep logging options in Hydra configuration under `logging`; pass the full
  composed config to `configure_logging(config=cfg)`. This package deliberately
  does not depend on Hydra, so unit tests and non-Hydra scripts can pass ordinary
  mappings or use `configure_logging(level=...)`.
- Use `rich_console()` for Rich output and `progress()` as a context-managed
  Rich progress display. Set `logging.markup: false` when log messages are not
  trusted or should be rendered literally.
- An active ClearML task receives Loguru messages automatically, even if a
  logging flag is false. If the composed config has `clearml.enabled: true` or
  `logging.clearml: true`, a forwarding sink is installed and resolves the
  current task as messages are emitted. Create/activate ClearML tasks in the
  experiment layer; this helper does not call `Task.init`.
- Never print or include ClearML credentials in logs. This helper does not
  read credential values, initialize ClearML sessions, or print credentials.

Example Hydra configuration:

```yaml
logging:
  level: INFO
  markup: true
  clearml: false

clearml:
  enabled: false
```

Example entrypoint and progress display:

```python
from taei.logging import configure_logging, progress


def run(cfg):  # cfg is the fully composed Hydra DictConfig
    configure_logging(config=cfg)
    with progress() as bar:
        task_id = bar.add_task("Working", total=100)
        # Advance with bar.update(task_id, advance=...) as work completes.
```

The package's existing `configure_logging(level="DEBUG")` call remains
supported. An explicit `level` argument overrides the Hydra setting, which in
turn overrides `TAEI_LOG_LEVEL` and the default `INFO` level.
