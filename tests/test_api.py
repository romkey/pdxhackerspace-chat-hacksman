from __future__ import annotations

import os
from pathlib import Path

from fastapi.testclient import TestClient

os.environ["CHAT_HACKSMAN_DB_PATH"] = str(Path("data/test_chat_hacksman.db"))

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
        return "Wear safety glasses."

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

        history_resp = client.get("/api/history?limit=1")
        assert history_resp.status_code == 200
        rows = history_resp.json()
        assert rows
        assert rows[0]["question"] == "What PPE do I need?"
