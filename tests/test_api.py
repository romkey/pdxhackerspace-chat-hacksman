import asyncio
import base64
from collections.abc import Callable

# Any: request bodies are JSON.
from typing import Any, cast

import httpx
import pytest
from fastapi import FastAPI

from chat_hacksman import __version__
from chat_hacksman import server as server_module
from chat_hacksman.llm import LlmError, LlmResult, LlmService
from chat_hacksman.models import AppSettings, ContextChunk
from chat_hacksman.rag import RagService
from tests.conftest import Handler, MakeClient

MockHttp = Callable[[Handler], None]


def settings_payload(system_prompt: str, model: str = "") -> dict[str, Any]:
    return {
        "model": model,
        "system_prompt": system_prompt,
        "tweaks": {"max_tokens": 512, "temperature": 0.1, "top_p": 0.9, "seed": None},
    }


def llm_result(answer: str = "ok") -> LlmResult:
    return LlmResult(
        answer=answer,
        model="test-model",
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        llm_latency_ms=120.0,
        tokens_per_second=41.6,
        provider_metrics={"finish_reason": "stop"},
    )


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> list[list[ContextChunk]]:
    """The LLM answers "ok"; returns the context each call was given."""
    calls: list[list[ContextChunk]] = []

    async def chat(
        self: LlmService, *, settings: AppSettings, question: str, context: list[ContextChunk]
    ) -> LlmResult:
        calls.append(context)
        return llm_result()

    monkeypatch.setattr(LlmService, "chat", chat)
    return calls


def fake_retrieve(
    monkeypatch: pytest.MonkeyPatch, chunks: list[ContextChunk] | None = None
) -> None:
    async def retrieve(
        self: RagService, query: str, enabled_collections: list[str] | None = None
    ) -> list[ContextChunk]:
        return list(chunks or [])

    monkeypatch.setattr(RagService, "retrieve", retrieve)


def test_healthz_reports_version(make_client: MakeClient) -> None:
    with make_client() as client:
        response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}


def test_settings_round_trip(make_client: MakeClient) -> None:
    with make_client() as client:
        put_resp = client.put(
            "/api/settings", json=settings_payload("Custom prompt", model="qwen3")
        )
        assert put_resp.status_code == 200
        get_resp = client.get("/api/settings")
    assert get_resp.status_code == 200
    assert get_resp.json()["system_prompt"] == "Custom prompt"
    assert get_resp.json()["model"] == "qwen3"


def test_settings_ignore_old_provider_fields(make_client: MakeClient) -> None:
    payload = settings_payload("p") | {"provider": "ollama", "llm_base_url": "http://x:11434"}
    with make_client() as client:
        saved = client.put("/api/settings", json=payload).json()
    assert "provider" not in saved
    assert "llm_base_url" not in saved


def test_settings_default_to_configured_collections(make_client: MakeClient) -> None:
    with make_client(rag_collections=("docs", "events")) as client:
        body = client.get("/api/rag/collections").json()
        assert body == {
            "available_collections": ["docs", "events"],
            "enabled_collections": ["docs", "events"],
        }
        payload = settings_payload("p")
        payload["enabled_rag_collections"] = ["events", "unknown"]
        saved = client.put("/api/settings", json=payload).json()
    assert saved["enabled_rag_collections"] == ["events"]


def test_llm_status_reports_transition_times(
    make_client: MakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    outcomes = [True, True, False]
    seen: list[tuple[str, str | None]] = []

    async def fake_check(url: str, api_key: str | None) -> bool:
        seen.append((url, api_key))
        return outcomes.pop(0)

    monkeypatch.setattr(server_module, "_check_llm_availability", fake_check)
    monkeypatch.setattr(server_module, "_resolve_host_ip", lambda url: "127.0.0.1")

    with make_client(llm_url="http://llm.local/v1", llm_api_key="key") as client:
        first = client.get("/api/llm/status").json()
        second = client.get("/api/llm/status").json()
        third = client.get("/api/llm/status").json()
    assert seen[0] == ("http://llm.local/v1", "key")
    assert first["url"] == "http://llm.local/v1"
    assert (first["available"], second["available"], third["available"]) == (True, True, False)
    assert second["last_changed_at"] == first["last_changed_at"]
    assert third["last_changed_at"] != first["last_changed_at"]


def test_check_llm_availability(mock_http: MockHttp) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "down.local":
            raise httpx.ConnectError("refused", request=request)
        assert request.url.path == "/v1/models"
        assert request.headers["Authorization"] == "Bearer key"
        return httpx.Response(401)

    mock_http(handler)
    check = server_module._check_llm_availability
    assert asyncio.run(check("http://up.local/v1", "key")) is True
    assert asyncio.run(check("http://down.local/v1", "key")) is False


def test_chat_endpoint_records_history(
    make_client: MakeClient, monkeypatch: pytest.MonkeyPatch, fake_llm: list[list[ContextChunk]]
) -> None:
    fake_retrieve(
        monkeypatch,
        [ContextChunk(collection="docs", score=0.9, text="Safety glasses are required.")],
    )

    with make_client(rag_collections=("docs",)) as client:
        response = client.post(
            "/api/chat", json={"question": "What PPE do I need?", "use_rag": True}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["metrics"]["completion_tokens"] == 5
        assert body["context"][0]["text"] == "Safety glasses are required."
        assert isinstance(body["history_id"], int)

        rows = client.get("/api/history?limit=1").json()
    assert rows[0]["question"] == "What PPE do I need?"
    assert isinstance(rows[0]["prompt_id"], int)
    assert rows[0]["rag_collections"] == ["docs"]


def test_chat_adds_fallback_events_for_schedule_questions(
    make_client: MakeClient, monkeypatch: pytest.MonkeyPatch, fake_llm: list[list[ContextChunk]]
) -> None:
    fake_retrieve(monkeypatch)
    occurrence = {
        "occurs_at_unix": 4_102_444_800,  # 2100-01-01
        "title": "Laser training",
        "occurs_at": "2100-01-01T00:00:00Z",
        "open_to": "members",
        "duration": 60,
    }
    with make_client(rag_collections=("events",)) as client:
        services: server_module.Services = cast(FastAPI, client.app).state.services
        services.topics._cache_occurrences = [occurrence]
        response = client.post(
            "/api/chat", json={"question": "what events are upcoming", "use_rag": True}
        )
    assert response.status_code == 200
    (context,) = fake_llm
    assert [chunk.text for chunk in context] == [
        "Laser training at 2100-01-01T00:00:00Z (open_to=members)"
    ]


def test_history_links_saved_chats_to_prompt_ids(
    make_client: MakeClient, monkeypatch: pytest.MonkeyPatch, fake_llm: list[list[ContextChunk]]
) -> None:
    fake_retrieve(monkeypatch)
    with make_client() as client:
        assert client.post("/api/chat", json={"question": "first"}).status_code == 200
        assert client.post("/api/chat", json={"question": "second"}).status_code == 200
        rows = client.get("/api/history?limit=2").json()
    assert len(rows) == 2
    assert isinstance(rows[0]["prompt_id"], int)
    assert rows[0]["prompt_id"] == rows[1]["prompt_id"]


def test_delete_history_endpoint(
    make_client: MakeClient, monkeypatch: pytest.MonkeyPatch, fake_llm: list[list[ContextChunk]]
) -> None:
    fake_retrieve(monkeypatch)
    with make_client() as client:
        assert client.post("/api/chat", json={"question": "ping"}).status_code == 200
        clear_resp = client.delete("/api/history")
        assert clear_resp.status_code == 200
        assert clear_resp.json()["deleted_rows"] == 1
        assert client.get("/api/history?limit=10").json() == []


def test_delete_latest_history_endpoint(
    make_client: MakeClient, monkeypatch: pytest.MonkeyPatch, fake_llm: list[list[ContextChunk]]
) -> None:
    fake_retrieve(monkeypatch)
    with make_client() as client:
        client.post("/api/chat", json={"question": "first"})
        client.post("/api/chat", json={"question": "second"})
        delete_resp = client.delete("/api/history/latest")
        assert delete_resp.json()["deleted_rows"] == 1
        rows = client.get("/api/history?limit=10").json()
    assert [row["question"] for row in rows] == ["first"]


def test_temporary_chat_not_saved_to_history(
    make_client: MakeClient, monkeypatch: pytest.MonkeyPatch, fake_llm: list[list[ContextChunk]]
) -> None:
    fake_retrieve(monkeypatch)
    with make_client() as client:
        chat_resp = client.post(
            "/api/chat", json={"question": "private", "use_rag": False, "temporary_chat": True}
        )
        assert chat_resp.status_code == 200
        assert chat_resp.json()["metrics"]["history_saved"] is False
        assert client.get("/api/history?limit=10").json() == []


def test_chat_endpoint_returns_502_when_llm_fails(
    make_client: MakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_retrieve(monkeypatch)

    async def failing_chat(
        self: LlmService, *, settings: AppSettings, question: str, context: list[ContextChunk]
    ) -> LlmResult:
        raise LlmError("model transport failed")

    monkeypatch.setattr(LlmService, "chat", failing_chat)

    with make_client() as client:
        response = client.post("/api/chat", json={"question": "ping", "use_rag": False})
    assert response.status_code == 502
    assert response.json()["detail"] == "model transport failed"


def test_optional_basic_auth_protects_routes(make_client: MakeClient) -> None:
    token = base64.b64encode(b"alice:wonderland").decode("ascii")
    headers = {"Authorization": f"Basic {token}"}
    wrong = base64.b64encode(b"alice:nope").decode("ascii")

    password = "wonderland"  # noqa: S105 - a dummy credential for this test
    with make_client(basic_auth_username="alice", basic_auth_password=password) as client:
        unauthorized = client.get("/api/settings")
        assert unauthorized.status_code == 401
        assert unauthorized.headers.get("www-authenticate") == 'Basic realm="Chat Hacksman"'
        assert client.get("/").status_code == 401
        assert client.get("/static/app.js").status_code == 401
        assert client.get("/api/settings", headers={"Authorization": "Bearer x"}).status_code == 401
        assert (
            client.get("/api/settings", headers={"Authorization": f"Basic {wrong}"}).status_code
            == 401
        )
        assert client.get("/api/settings", headers=headers).status_code == 200
        assert client.get("/health").status_code == 200
        assert client.get("/healthz").status_code == 200


def test_root_serves_the_ui(make_client: MakeClient) -> None:
    with make_client() as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_feedback_endpoint_records_and_returns_feedback(
    make_client: MakeClient, monkeypatch: pytest.MonkeyPatch, fake_llm: list[list[ContextChunk]]
) -> None:
    fake_retrieve(monkeypatch)
    with make_client() as client:
        history_id = client.post("/api/chat", json={"question": "ping"}).json()["history_id"]
        feedback_resp = client.post(
            "/api/feedback",
            json={
                "rating": "up",
                "text": "helpful response",
                "history_id": history_id,
                "question": "ping",
                "answer": "ok",
                "provider": "openai",
                "model": "test-model",
            },
        )
        assert feedback_resp.status_code == 200
        assert isinstance(feedback_resp.json()["id"], int)
        rows = client.get("/api/feedback?limit=10").json()
    assert rows[0]["rating"] == "up"
    assert rows[0]["text"] == "helpful response"


def test_models_endpoint_lists_server_models(make_client: MakeClient, mock_http: MockHttp) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={"object": "list", "data": [{"id": "zeta"}, {"id": "Alpha"}, {"x": 1}]},
        )

    mock_http(handler)
    with make_client(
        llm_url="http://llm.local/v1", llm_api_key="llm-secret", llm_model="zeta"
    ) as client:
        response = client.get("/api/models")
    assert response.json() == {"default_model": "zeta", "models": ["Alpha", "zeta"], "error": None}
    assert str(seen[0].url) == "http://llm.local/v1/models"
    assert seen[0].headers["Authorization"] == "Bearer llm-secret"


def test_models_endpoint_reports_errors(make_client: MakeClient, mock_http: MockHttp) -> None:
    mock_http(lambda request: httpx.Response(500))
    with make_client() as client:
        failed = client.get("/api/models").json()
        assert failed["models"] == []
        assert "500" in failed["error"]

        mock_http(lambda request: httpx.Response(200, json={"models": "nope"}))
        odd = client.get("/api/models").json()
    assert "no data list" in odd["error"]


def test_meta_endpoint(make_client: MakeClient) -> None:
    with make_client(repo_url="https://example.com/repo") as client:
        body = client.get("/api/meta").json()
    assert body == {
        "app_name": "Chat Hacksman",
        "version": __version__,
        "repo_url": "https://example.com/repo",
    }


def test_topics_endpoint_stores_feed_records(make_client: MakeClient, mock_http: MockHttp) -> None:
    feed = {
        "interests": ["Lasers"],
        "training_topics": ["CNC"],
        "events": [
            {
                "id": 1,
                "slug": "laser-night",
                "title": "Laser night",
                "start_time": "2100-01-01T00:00:00Z",
                "duration": 60,
            }
        ],
    }
    mock_http(lambda request: httpx.Response(200, json=feed))
    with make_client() as client:
        topics = client.get("/api/topics")
        assert topics.status_code == 200
        assert "Lasers" in topics.json()["all_topics"]
        events = client.get("/api/events").json()
        assert client.get("/api/occurrences").json() == []
    assert [event["slug"] for event in events] == ["laser-night"]


def test_topics_endpoint_returns_502_when_feed_fails(
    make_client: MakeClient, mock_http: MockHttp
) -> None:
    mock_http(lambda request: httpx.Response(503))
    with make_client() as client:
        response = client.get("/api/topics")
    assert response.status_code == 502
