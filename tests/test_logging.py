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
