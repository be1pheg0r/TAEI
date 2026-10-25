"""Настройка Loguru, форматирования Rich, захвата вывода и передачи логов ClearML.

``configure_logging`` вызывается точкой входа приложения после сборки Hydra-
конфигурации, но модуль не требует Hydra как runtime-зависимость. Он читает
параметры уровня и разметки, создает консоль и индикатор прогресса, подключает
уже активную задачу ClearML без ее инициализации и не раскрывает учетные данные.
Контекстный менеджер ``capture_output`` дублирует stdout/stderr в UTF-8 файл,
включая доступные записи через файловые дескрипторы, и восстанавливает потоки
после завершения, в том числе при исключении.
"""

from __future__ import annotations

import codecs
import contextvars
import os
import select
import sys
import threading
import time
from collections.abc import Mapping
from contextlib import contextmanager, suppress
from io import TextIOBase
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Column

_CAPTURED_TTY: contextvars.ContextVar[tuple[bool, bool] | None] = (
    contextvars.ContextVar("taei_captured_tty", default=None)
)


class _TeeTextStream(TextIOBase):
    """Text-stream fallback for environments whose standard streams lack FDs."""

    def __init__(self, original: Any, log_file: Any, lock: threading.Lock) -> None:
        self._original = original
        self._log_file = log_file
        self._lock = lock
        self._active = True

    def deactivate(self) -> None:
        with self._lock:
            self._active = False

    @property
    def encoding(self) -> str:
        return getattr(self._original, "encoding", None) or "utf-8"

    @property
    def errors(self) -> str:
        return getattr(self._original, "errors", None) or "strict"

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        self._original.write(text)
        with self._lock:
            if self._active:
                self._log_file.write(text)
                self._log_file.flush()
        return len(text)

    def flush(self) -> None:
        self._original.flush()
        with self._lock:
            if self._active:
                self._log_file.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._original, "isatty", lambda: False)())


@contextmanager
def capture_output(log_path: str | Path):
    """Tee stdout and stderr (including native writes) to a UTF-8 text file.

    File-descriptor capture is used when both standard streams expose usable
    descriptors. In notebook/test environments without descriptors, Python
    stream writes are captured through a text-stream fallback. Relay workers
    drain data available at teardown and stop even if a child retains a pipe;
    writes made by processes after capture exits are not captured.
    """
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    log_file = path.open("w", encoding="utf-8", buffering=1)
    lock = threading.Lock()
    original_streams = (sys.stdout, sys.stderr)
    captured_tty = (
        bool(getattr(original_streams[0], "isatty", lambda: False)()),
        bool(getattr(original_streams[1], "isatty", lambda: False)()),
    )
    tty_token = _CAPTURED_TTY.set(captured_tty)
    fds: tuple[int, int] | None = None
    saved_fds: list[int] = []
    pipe_fds: list[tuple[int, int]] = []
    threads: list[threading.Thread] = []
    fallback_streams: tuple[_TeeTextStream, _TeeTextStream] | None = None
    try:
        try:
            fds = (sys.stdout.fileno(), sys.stderr.fileno())
            saved_fds = [os.dup(fd) for fd in fds]
            pipe_fds = [os.pipe(), os.pipe()]
            for fd, (_, write_fd) in zip(fds, pipe_fds, strict=True):
                os.dup2(write_fd, fd)
                os.close(write_fd)
        except (AttributeError, OSError, ValueError):
            for fd, saved_fd in zip(fds or (), saved_fds, strict=False):
                os.dup2(saved_fd, fd)
                os.close(saved_fd)
            for pipe_pair in pipe_fds:
                for pipe_fd in pipe_pair:
                    with suppress(OSError):
                        os.close(pipe_fd)
            saved_fds.clear()
            pipe_fds.clear()
            fallback_streams = (
                _TeeTextStream(original_streams[0], log_file, lock),
                _TeeTextStream(original_streams[1], log_file, lock),
            )
            sys.stdout, sys.stderr = fallback_streams
            try:
                yield
            finally:
                sys.stdout, sys.stderr = original_streams
                for stream in fallback_streams:
                    stream.deactivate()
            return

        stop_relay = threading.Event()
        relay_errors: list[tuple[str, BaseException]] = []

        def write_all(fd: int, data: bytes) -> None:
            remaining = memoryview(data)
            while remaining:
                written = os.write(fd, remaining)
                if written == 0:
                    raise OSError("zero-length write to capture console")
                remaining = remaining[written:]

        def relay(read_fd: int, original_fd: int) -> None:
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            poller = select.poll()
            poller.register(read_fd, select.POLLIN | select.POLLHUP | select.POLLERR)
            try:
                while True:
                    ready = poller.poll(50)
                    if not ready and not stop_relay.is_set():
                        continue
                    try:
                        chunk = os.read(read_fd, 8192)
                    except BlockingIOError:
                        if stop_relay.is_set():
                            break
                        continue
                    if not chunk:
                        break
                    try:
                        write_all(original_fd, chunk)
                    except OSError as exc:
                        relay_errors.append(("console", exc))
                    text = decoder.decode(chunk)
                    if text:
                        with lock:
                            try:
                                log_file.write(text)
                                log_file.flush()
                            except BaseException as exc:
                                relay_errors.append(("disk", exc))
                                break
                tail = decoder.decode(b"", final=True)
                if tail:
                    with lock:
                        try:
                            log_file.write(tail)
                            log_file.flush()
                        except BaseException as exc:
                            relay_errors.append(("disk", exc))
            finally:
                os.close(read_fd)

        for read_fd, _ in pipe_fds:
            os.set_blocking(read_fd, False)

        for (read_fd, _), saved_fd in zip(pipe_fds, saved_fds, strict=True):
            thread = threading.Thread(
                target=relay,
                args=(read_fd, saved_fd),
                name="taei-output-relay",
            )
            thread.start()
            threads.append(thread)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            for fd, saved_fd in zip(fds, saved_fds, strict=True):
                os.dup2(saved_fd, fd)
            stop_relay.set()
            for thread in threads:
                thread.join()
            for saved_fd in saved_fds:
                os.close(saved_fd)
            disk_errors = [exc for kind, exc in relay_errors if kind == "disk"]
            console_errors = [exc for kind, exc in relay_errors if kind == "console"]
            if sys.exc_info()[0] is None:
                if disk_errors:
                    raise disk_errors[0]
                if console_errors:
                    raise console_errors[0]
    finally:
        _CAPTURED_TTY.reset(tty_token)
        log_file.flush()
        log_file.close()


def _current_clearml_task() -> Any | None:
    """Return the active ClearML task without initializing or authenticating one."""
    try:
        from clearml import Task

        return Task.current_task()
    except Exception:
        return None


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    try:
        return value if hasattr(value, "get") else {}
    except Exception:
        return {}


def _logging_options(config: Any) -> tuple[Mapping[str, Any], bool]:
    root = _as_mapping(config)
    nested = _as_mapping(root.get("logging", root))
    clearml_config = _as_mapping(root.get("clearml", {}))
    clearml_enabled = bool(nested.get("clearml", False)) or bool(
        clearml_config.get("enabled", False)
    )
    return nested, clearml_enabled


def _terminal_ui_enabled(*, stderr: bool = True) -> bool:
    if "NO_COLOR" in os.environ or os.getenv("TERM", "").lower() == "dumb":
        return False
    captured = _CAPTURED_TTY.get()
    if captured is not None:
        return captured[1 if stderr else 0]
    stream = sys.stderr if stderr else sys.stdout
    return bool(getattr(stream, "isatty", lambda: False)())


def rich_console(*, markup: bool = True, stderr: bool = True) -> Console:
    """Create the Rich console used for styled log output."""
    return Console(
        stderr=stderr,
        markup=markup,
        highlight=False,
        force_terminal=_terminal_ui_enabled(stderr=stderr),
    )


def progress(**kwargs: Any) -> Progress:
    """Create a Rich progress bar, suitable for use as a context manager."""
    return Progress(console=rich_console(), **kwargs)


class _ExperimentProgress:
    """Completed-work progress with live TTY and readable detached snapshots."""

    def __init__(self, total: int, label: str) -> None:
        if total < 0:
            raise ValueError("progress total must not be negative")
        self.total = total
        self.label = label
        self.completed = 0
        self._started_at = 0.0
        self._last_stage: str | None = None
        self._console = rich_console()
        self._live: Progress | None = None
        self._task_id: Any = None
        self._detached = os.getenv("TAEI_DETACHED_PROGRESS") == "1"
        self._stop_refresh = threading.Event()
        self._refresh_thread: threading.Thread | None = None
        self._state_lock = threading.RLock()

    @staticmethod
    def _duration(seconds: float) -> str:
        minutes, seconds = divmod(max(0, int(seconds)), 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}"

    def _print_detached(self, outcome: str | None = None) -> None:
        with self._state_lock:
            elapsed = max(0.0, time.monotonic() - self._started_at)
            fraction = self.completed / self.total if self.total else 1.0
            filled = int(fraction * 24)
            bar = "#" * filled + "-" * (24 - filled)
            remaining = (
                elapsed * (self.total - self.completed) / self.completed
                if self.completed
                else None
            )
            eta = self._duration(remaining) if remaining is not None else "unknown"
            detail = outcome or self._last_stage or "starting"
            self._console.print(
                f"{self.label} [{bar}] {self.completed}/{self.total} "
                f"({fraction:.1%}) elapsed={self._duration(elapsed)} "
                f"ETA~{eta} | {detail}",
                markup=False,
                highlight=False,
                soft_wrap=True,
            )

    def _refresh_detached(self) -> None:
        while not self._stop_refresh.wait(30.0):
            self._print_detached()

    def __enter__(self) -> _ExperimentProgress:
        self._started_at = time.monotonic()
        if self._console.is_terminal:
            self._live = Progress(
                SpinnerColumn(),
                TextColumn(
                    "{task.description}",
                    table_column=Column(
                        max_width=32, no_wrap=True, overflow="ellipsis"
                    ),
                ),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self._console,
                transient=False,
            )
            self._live.start()
            self._task_id = self._live.add_task(self.label, total=self.total)
        elif self._detached:
            self._print_detached()
            self._refresh_thread = threading.Thread(
                target=self._refresh_detached,
                name="taei-detached-progress",
                daemon=True,
            )
            self._refresh_thread.start()
        else:
            self._console.print(f"{self.label} started: 0/{self.total}")
        return self

    def stage(self, description: str) -> None:
        with self._state_lock:
            if description == self._last_stage:
                return
            self._last_stage = description
        text = f"{self.label}: {description} ({self.completed}/{self.total} completed)"
        if self._live is not None:
            self._live.update(self._task_id, description=text)
        elif self._detached:
            self._print_detached()
        else:
            self._console.print(text)

    def advance(self, outcome: str) -> None:
        with self._state_lock:
            self.completed = min(self.completed + 1, self.total)
            if self._detached:
                self._last_stage = f"completed ({outcome})"
        if self._live is not None:
            self._live.update(self._task_id, advance=1)
            self._live.update(
                self._task_id,
                description=(
                    f"{self.label}: {self.completed}/{self.total} completed ({outcome})"
                ),
            )
        elif self._detached:
            self._print_detached()
        else:
            self._console.print(
                f"{self.label}: {self.completed}/{self.total} completed ({outcome})"
            )

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._stop_refresh.set()
        if self._refresh_thread is not None:
            self._refresh_thread.join()
        status = (
            "finished"
            if exc is None
            else "interrupted"
            if isinstance(exc, KeyboardInterrupt)
            else "failed"
        )
        elapsed = time.monotonic() - self._started_at
        if self._live is not None:
            if exc is not None:
                self._live.update(self._task_id, description=f"{self.label}: {status}")
            self._live.stop()
            self._console.print(
                f"{self.label} {status}: {self.completed}/{self.total} completed "
                f"in {elapsed:.1f}s"
            )
        elif self._detached:
            self._print_detached(status)
        else:
            self._console.print(
                f"{self.label} {status}: {self.completed}/{self.total} completed "
                f"in {elapsed:.1f}s"
            )


def experiment_progress(total: int, label: str) -> _ExperimentProgress:
    """Render completed work live on TTY or periodically in detached stderr logs."""
    return _ExperimentProgress(total, label)


def configure_logging(
    *,
    level: str | None = None,
    env_file: str | Path | None = None,
    config: Any | None = None,
) -> None:
    """Configure Rich console logging and ClearML forwarding.

    ``config`` may be a composed Hydra ``DictConfig`` or an equivalent mapping
    with ``logging.level``, ``logging.markup`` and optional ``logging.clearml``
    keys. A root ``clearml.enabled`` setting is also honored. An already active
    ClearML task is always wired, regardless of config switches. This function
    never calls ``Task.init`` or logs ClearML credentials.
    """
    if env_file is not None:
        load_dotenv(Path(env_file), override=False)

    options, clearml_enabled = _logging_options(config)
    chosen_level = level or options.get("level") or os.getenv("TAEI_LOG_LEVEL", "INFO")
    console = rich_console(markup=bool(options.get("markup", True)))
    task = _current_clearml_task()

    logger.remove()
    logger.add(
        lambda message: console.print(message, end=""),
        level=str(chosen_level).upper(),
        format="{level: <8} | {message}",
        colorize=False,
        enqueue=False,
        catch=True,
        backtrace=False,
        diagnose=False,
    )

    if task is not None or clearml_enabled:

        def forward_to_clearml(message: Any) -> None:
            active_task = task or _current_clearml_task()
            if active_task is None:
                return
            record = message.record
            active_task.get_logger().report_text(
                record["message"], level=record["level"].name
            )

        logger.add(
            forward_to_clearml,
            level=str(chosen_level).upper(),
            format="{message}",
            enqueue=False,
            catch=True,
            backtrace=False,
            diagnose=False,
        )
