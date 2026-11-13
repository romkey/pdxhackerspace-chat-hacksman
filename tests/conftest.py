import dataclasses
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from chat_hacksman.config import Settings
from chat_hacksman.server import create_app

Handler = Callable[[httpx.Request], httpx.Response]
MakeClient = Callable[..., TestClient]


@pytest.fixture
def make_client(tmp_path: Path) -> Iterator[MakeClient]:
    """Builds a TestClient for an app with its own database; keyword
    arguments override Settings fields."""
    clients: list[TestClient] = []

    def make(**overrides: Any) -> TestClient:  # Any: Settings fields have mixed types.
        settings = dataclasses.replace(Settings(db_path=tmp_path / "chat_hacksman.db"), **overrides)
        client = TestClient(create_app(settings))
        clients.append(client)
        return client

    yield make
    for client in clients:
        client.close()


@pytest.fixture
def mock_http(monkeypatch: pytest.MonkeyPatch) -> Callable[[Handler], None]:
    """Routes every httpx.AsyncClient the app creates to a handler function."""
    real_async_client = httpx.AsyncClient

    def install(handler: Handler) -> None:
        def factory(**kwargs: Any) -> httpx.AsyncClient:  # Any: httpx's own kwargs.
            kwargs.pop("transport", None)
            return real_async_client(transport=httpx.MockTransport(handler), **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", factory)

    return install
