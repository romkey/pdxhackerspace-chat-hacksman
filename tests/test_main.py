from pathlib import Path

import pytest
import sentry_sdk
import uvicorn
from fastapi import FastAPI

from chat_hacksman import __main__ as cli
from chat_hacksman import __version__


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out == f"chat-hacksman {__version__}\n"


def test_bad_config_exits_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("APP_PORT", "eighty")
    with pytest.raises(SystemExit) as exit_info:
        cli.main([])
    assert exit_info.value.code == 2
    assert "APP_PORT must be a whole number" in capsys.readouterr().err


def test_runs_uvicorn_with_configured_address(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CHAT_HACKSMAN_DB_PATH", str(tmp_path / "db.sqlite"))
    monkeypatch.setenv("APP_HOST", "127.0.0.2")
    monkeypatch.setenv("APP_PORT", "9123")
    monkeypatch.delenv("SENTRY_DSN", raising=False)
    calls: list[tuple[FastAPI, str, int]] = []

    def fake_run(app: FastAPI, *, host: str, port: int, log_config: object) -> None:
        calls.append((app, host, port))

    monkeypatch.setattr(uvicorn, "run", fake_run)

    assert cli.main([]) == 0
    ((app, host, port),) = calls
    assert isinstance(app, FastAPI)
    assert (host, port) == ("127.0.0.2", 9123)
    assert (tmp_path / "db.sqlite").exists()


def test_configure_sentry(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, str]] = []

    def fake_init(*, dsn: str, release: str) -> None:
        calls.append({"dsn": dsn, "release": release})

    monkeypatch.setattr(sentry_sdk, "init", fake_init)

    assert cli.configure_sentry(None) is False
    assert calls == []
    assert cli.configure_sentry("https://public@example.ingest.sentry.io/1") is True
    assert calls == [
        {
            "dsn": "https://public@example.ingest.sentry.io/1",
            "release": f"chat-hacksman@{__version__}",
        }
    ]
