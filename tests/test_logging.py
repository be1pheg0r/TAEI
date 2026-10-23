from loguru import logger

from taei.logging import configure_logging


def test_configure_logging_emits_message(capsys) -> None:
    configure_logging(level="INFO")

    logger.info("taei logging works")

    assert "taei logging works" in capsys.readouterr().err
