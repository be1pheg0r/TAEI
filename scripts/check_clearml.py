"""Проверяет аутентификацию ClearML по учетным данным из корневого ``.env``.

Скрипт читает только обязательные параметры подключения, сообщает о недостающих
значениях без раскрытия секретов и проверяет сессию ClearML. Код завершения
различает успешную проверку, отсутствие конфигурации и ошибку соединения либо
аутентификации; задачи эксперимента он не создает.
"""

from pathlib import Path

from clearml.backend_api.session import Session
from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_SETTINGS = (
    "CLEARML_API_HOST",
    "CLEARML_API_ACCESS_KEY",
    "CLEARML_API_SECRET_KEY",
)


def main() -> int:
    """Return 0 if ClearML authenticates, 1 on connection/auth failure, 2 if unset."""
    settings = dotenv_values(PROJECT_ROOT / ".env")
    missing = [name for name in REQUIRED_SETTINGS if not settings.get(name)]
    if missing:
        print("Missing .env settings: " + ", ".join(missing))
        return 2

    try:
        Session(
            host=settings["CLEARML_API_HOST"],
            api_key=settings["CLEARML_API_ACCESS_KEY"],
            secret_key=settings["CLEARML_API_SECRET_KEY"],
        )
    except Exception as exc:
        print(
            f"ClearML check failed ({type(exc).__name__}); "
            "verify credentials and API connectivity."
        )
        return 1

    print("ClearML authentication succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
