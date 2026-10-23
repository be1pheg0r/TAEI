# Agent instructions

For project logging, follow [`docs/logging.md`](docs/logging.md) and use the
reusable API in `src/taei/logging.py`. Configure logging from the composed Hydra
config at the application entrypoint. Keep experiment/task lifecycle management
outside the logging helper, and never expose ClearML credentials in logs.
