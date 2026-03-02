from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

os.environ["CHAT_HACKSMAN_DB_PATH"] = str(Path("data/test_chat_hacksman.db"))

import app.main as main_module  # noqa: E402
from app.llm import LlmError  # noqa: E402
from app.main import app, llm_service, rag_service  # noqa: E402
from app.models import ContextChunk  # noqa: E402


def test_settings_round_trip() -> None:
    with TestClient(app) as client:
        updated = {
            "provider": "ollama",
            "llm_base_url": "http://localhost:11434",
            "model": "llama3.2:latest",
            "system_prompt": "Custom prompt",
            "tweaks": {
                "max_tokens": 512,
                "temperature": 0.1,
                "top_p": 0.9,
                "num_ctx": 4096,
                "repeat_penalty": 1.1,
                "seed": None,
            },
        }
        put_resp = client.put("/api/settings", json=updated)
        assert put_resp.status_code == 200
        get_resp = client.get("/api/settings")
        assert get_resp.status_code == 200
        assert get_resp.json()["system_prompt"] == "Custom prompt"


def test_chat_endpoint_records_history(monkeypatch) -> None:
    async def fake_retrieve(self, _: str, enabled_collections=None):  # noqa: ANN001
        del self
        del enabled_collections
        return [
            ContextChunk(
                collection="docs",
                score=0.9,
                text="Safety glasses are required.",
                metadata={},
            )
        ]

    async def fake_chat(self, *, settings, question, context):  # noqa: ANN001
        del self
        del settings, question, context
        return SimpleNamespace(
            answer="Wear safety glasses.",
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            llm_latency_ms=120.0,
            tokens_per_second=41.6,
            provider_metrics={"provider": "ollama"},
        )

    monkeypatch.setattr(type(rag_service), "retrieve", fake_retrieve)
    monkeypatch.setattr(type(llm_service), "chat", fake_chat)

    with TestClient(app) as client:
        response = client.post(
            "/api/chat",
            json={"question": "What PPE do I need?", "use_rag": True},
        )
        assert response.status_code == 200
        body = response.json()
        assert "safety glasses" in body["answer"].lower()
        assert body["metrics"]["completion_tokens"] == 5

        history_resp = client.get("/api/history?limit=1")
        assert history_resp.status_code == 200
        rows = history_resp.json()
        assert rows
        assert rows[0]["question"] == "What PPE do I need?"


def test_delete_history_endpoint(monkeypatch) -> None:
    async def fake_retrieve(self, _: str, enabled_collections=None):  # noqa: ANN001
        del self, enabled_collections
        return []

    async def fake_chat(self, *, settings, question, context):  # noqa: ANN001
        del self, settings, question, context
        return SimpleNamespace(
            answer="ok",
            input_tokens=1,
            output_tokens=1,
            total_tokens=2,
            llm_latency_ms=1.0,
            tokens_per_second=1000.0,
            provider_metrics={},
        )

    monkeypatch.setattr(type(rag_service), "retrieve", fake_retrieve)
    monkeypatch.setattr(type(llm_service), "chat", fake_chat)

    with TestClient(app) as client:
        chat_resp = client.post("/api/chat", json={"question": "ping", "use_rag": False})
        assert chat_resp.status_code == 200

        clear_resp = client.delete("/api/history")
        assert clear_resp.status_code == 200
        assert clear_resp.json()["deleted_rows"] >= 1

        history_resp = client.get("/api/history?limit=10")
        assert history_resp.status_code == 200
        assert history_resp.json() == []


def test_delete_latest_history_endpoint(monkeypatch) -> None:
    async def fake_retrieve(self, _: str, enabled_collections=None):  # noqa: ANN001
        del self, enabled_collections
        return []

    async def fake_chat(self, *, settings, question, context):  # noqa: ANN001
        del self, settings, question, context
        return SimpleNamespace(
            answer="ok",
            input_tokens=1,
            output_tokens=1,
            total_tokens=2,
            llm_latency_ms=1.0,
            tokens_per_second=1000.0,
            provider_metrics={},
        )

    monkeypatch.setattr(type(rag_service), "retrieve", fake_retrieve)
    monkeypatch.setattr(type(llm_service), "chat", fake_chat)

    with TestClient(app) as client:
        client.post("/api/chat", json={"question": "first", "use_rag": False})
        client.post("/api/chat", json={"question": "second", "use_rag": False})

        delete_resp = client.delete("/api/history/latest")
        assert delete_resp.status_code == 200
        assert delete_resp.json()["deleted_rows"] == 1

        history_resp = client.get("/api/history?limit=10")
        assert history_resp.status_code == 200
        rows = history_resp.json()
        assert rows
        assert rows[0]["question"] == "first"


def test_temporary_chat_not_saved_to_history(monkeypatch) -> None:
    async def fake_retrieve(self, _: str, enabled_collections=None):  # noqa: ANN001
        del self, enabled_collections
        return []

    async def fake_chat(self, *, settings, question, context):  # noqa: ANN001
        del self, settings, question, context
        return SimpleNamespace(
            answer="temp",
            input_tokens=2,
            output_tokens=2,
            total_tokens=4,
            llm_latency_ms=2.0,
            tokens_per_second=500.0,
            provider_metrics={},
        )

    monkeypatch.setattr(type(rag_service), "retrieve", fake_retrieve)
    monkeypatch.setattr(type(llm_service), "chat", fake_chat)

    with TestClient(app) as client:
        client.delete("/api/history")

        chat_resp = client.post(
            "/api/chat",
            json={"question": "private", "use_rag": False, "temporary_chat": True},
        )
        assert chat_resp.status_code == 200
        assert chat_resp.json()["metrics"]["history_saved"] is False

        history_resp = client.get("/api/history?limit=10")
        assert history_resp.status_code == 200
        assert history_resp.json() == []


def test_chat_endpoint_returns_502_when_llm_fails(monkeypatch) -> None:
    async def fake_retrieve(self, _: str, enabled_collections=None):  # noqa: ANN001
        del self, enabled_collections
        return []

    async def fake_chat(self, *, settings, question, context):  # noqa: ANN001
        del self, settings, question, context
        raise LlmError("model transport failed")

    monkeypatch.setattr(type(rag_service), "retrieve", fake_retrieve)
    monkeypatch.setattr(type(llm_service), "chat", fake_chat)

    with TestClient(app) as client:
        response = client.post("/api/chat", json={"question": "ping", "use_rag": False})
        assert response.status_code == 502
        assert response.json()["detail"] == "model transport failed"


def test_models_endpoint_returns_ollama_model_list(monkeypatch) -> None:
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "models": [
                    {"name": "llama3.2:latest"},
                    {"name": "nomic-embed-text"},
                ]
            }

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return False

        async def get(self, url):
            assert url.endswith("/api/tags")
            return FakeResponse()

    monkeypatch.setattr(main_module.httpx, "AsyncClient", lambda timeout: FakeAsyncClient())

    with TestClient(app) as client:
        response = client.get("/api/models?provider=ollama&base_url=http://localhost:11434")
        assert response.status_code == 200
        body = response.json()
        assert body["provider"] == "ollama"
        assert "llama3.2:latest" in body["models"]


def test_meta_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/api/meta")
        assert response.status_code == 200
        body = response.json()
        assert body["app_name"] == "Chat Hacksman"
        assert isinstance(body["version"], str)
        assert isinstance(body["repo_url"], str)


def test_pull_model_endpoint(monkeypatch) -> None:
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"status": "success"}

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return False

        async def post(self, url, json):
            assert url.endswith("/api/pull")
            assert json["name"] == "llama3.2:latest"
            assert json["stream"] is False
            return FakeResponse()

    monkeypatch.setattr(main_module.httpx, "AsyncClient", lambda timeout: FakeAsyncClient())

    with TestClient(app) as client:
        response = client.post(
            "/api/models/pull",
            json={
                "provider": "ollama",
                "base_url": "http://localhost:11434",
                "name": "llama3.2:latest",
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["provider"] == "ollama"
        assert body["pulled_model"] == "llama3.2:latest"
