"""Проверяет настройку журналирования, Rich-инструменты и передачу логов в ClearML.

Тесты проверяют вывод сообщений с обычной и Hydra-конфигурацией, консоль Rich
с разметкой, создание progress-индикатора и передачу сообщений уже активной
задаче ClearML без управления ее жизненным циклом из помощника логирования.
"""

import pytest
from loguru import logger
from rich.progress import Progress

from taei.logging import configure_logging, progress, rich_console


def test_configure_logging_emits_message(capsys) -> None:
    configure_logging(level="INFO")

    logger.info("taei logging works")

    assert "taei logging works" in capsys.readouterr().err


def test_configure_logging_accepts_composed_hydra_config(capsys) -> None:
    configure_logging(config={"logging": {"level": "WARNING", "markup": True}})

    logger.info("hidden")
    logger.warning("[bold]visible[/bold]")

    output = capsys.readouterr().err
    assert "hidden" not in output
    assert "visible" in output


def test_rich_console_enables_markup() -> None:
    console = rich_console(markup=True)

    assert console._markup is True


def test_progress_returns_rich_progress() -> None:
    assert isinstance(progress(), Progress)


def test_rich_console_preserves_tty_visibility_during_fd_capture(tmp_path, monkeypatch):
    import sys

    from taei.logging import capture_output

    original = sys.stderr
    monkeypatch.setattr(original, "isatty", lambda: True)
    with capture_output(tmp_path / "run.log"):
        console = rich_console()

    assert console.is_terminal is True


def test_experiment_progress_plain_mode_reports_only_truthful_boundaries(capsys):
    from taei.logging import experiment_progress

    with experiment_progress(2, "Dataset") as progress_bar:
        progress_bar.stage("Waiting for attempt 1/2")
        progress_bar.advance("success")
        progress_bar.stage("Waiting for attempt 2/2")
        progress_bar.advance("failed")

    output = capsys.readouterr().err
    assert "Dataset started: 0/2" in output
    assert "Dataset: 1/2 completed (success)" in output
    assert "Dataset: 2/2 completed (failed)" in output
    assert "100%" not in output
    assert len(output.splitlines()) == 6


def test_experiment_progress_reports_runtime_error_as_failure(capsys):
    from taei.logging import experiment_progress

    with (
        pytest.raises(RuntimeError, match="attempt error"),
        experiment_progress(1, "Dataset"),
    ):
        raise RuntimeError("attempt error")

    assert "Dataset failed: 0/1 completed" in capsys.readouterr().err


@pytest.mark.parametrize(("variable", "value"), [("NO_COLOR", "1"), ("TERM", "dumb")])
def test_non_color_terminal_settings_disable_live_mode(monkeypatch, variable, value):
    import io

    from taei import logging as taei_logging

    class Terminal(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.setattr(taei_logging.sys, "stderr", Terminal())
    monkeypatch.setenv(variable, value)

    assert taei_logging.rich_console().is_terminal is False


def test_clearml_connected_task_receives_log_messages(monkeypatch) -> None:
    from taei import logging as taei_logging

    reported = []

    class FakeTaskLogger:
        def report_text(self, text, level=None):
            reported.append((text, level))

    class FakeTask:
        def get_logger(self):
            return FakeTaskLogger()

    monkeypatch.setattr(taei_logging, "_current_clearml_task", lambda: FakeTask())
    configure_logging(config={"logging": {"level": "INFO", "clearml": False}})
    logger.info("forwarded safely")

    assert any("forwarded safely" in text for text, _ in reported)
