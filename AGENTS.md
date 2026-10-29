# Agent instructions

For project logging, follow [`README.md#логирование`](README.md#логирование) and use the
reusable API in `src/taei/logging.py`. Configure logging from the composed Hydra
config at the application entrypoint. Keep experiment/task lifecycle management
outside the logging helper, and never expose service credentials in logs.
