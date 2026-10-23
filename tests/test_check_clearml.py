from taei.check_clearml import main


def test_check_clearml_authenticates_without_printing_credentials(
    monkeypatch, capsys
) -> None:
    from taei import check_clearml

    monkeypatch.setattr(
        check_clearml,
        "dotenv_values",
        lambda _: {
            "CLEARML_API_HOST": "https://api.example.test",
            "CLEARML_API_ACCESS_KEY": "access-secret-value",
            "CLEARML_API_SECRET_KEY": "private-secret-value",
        },
    )
    calls = {}

    def fake_session(**kwargs):
        calls.update(kwargs)

    monkeypatch.setattr(check_clearml, "Session", fake_session)

    result = main()
    output = capsys.readouterr().out

    assert result == 0
    assert calls == {
        "host": "https://api.example.test",
        "api_key": "access-secret-value",
        "secret_key": "private-secret-value",
    }
    assert "succeeded" in output
    assert "access-secret-value" not in output
    assert "private-secret-value" not in output


def test_check_clearml_reports_missing_credentials_without_connecting(
    monkeypatch, capsys
) -> None:
    from taei import check_clearml

    monkeypatch.setattr(check_clearml, "dotenv_values", lambda _: {})
    monkeypatch.setattr(
        check_clearml,
        "Session",
        lambda **_: (_ for _ in ()).throw(AssertionError("should not connect")),
    )

    result = main()
    output = capsys.readouterr().out

    assert result == 2
    assert "CLEARML_API_ACCESS_KEY" in output
