import asyncio
import json
from collections.abc import AsyncIterator, Callable

# Any: request bodies are JSON.
from typing import Any, cast

import httpx
import pytest
from fastapi import FastAPI

from chat_hacksman import __version__
from chat_hacksman import server as server_module
from chat_hacksman.llm import LlmError, LlmResult, LlmService, StreamState
from chat_hacksman.models import AppSettings, ChatTurn, ContextChunk
from chat_hacksman.rag import RagService
from chat_hacksman.timing import StageTimer
from tests.conftest import Handler, MakeClient

MockHttp = Callable[[Handler], None]


def settings_payload(system_prompt: str, model: str = "") -> dict[str, Any]:
    return {
        "model": model,
        "system_prompt": system_prompt,
        "tweaks": {"max_tokens": 512, "temperature": 0.1, "top_p": 0.9, "seed": None},
    }


def llm_result(answer: str = "ok [1]") -> LlmResult:
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
        self: LlmService,
        *,
        settings: AppSettings,
        question: str,
        context: list[ContextChunk],
        history: list[ChatTurn] | None = None,
    ) -> LlmResult:
        calls.append(context)
        return llm_result()

    monkeypatch.setattr(LlmService, "chat", chat)
    return calls


FOUND = [
    ContextChunk(
        collection="hs_member",
        score=0.9,
        text="Safety glasses are required.",
        metadata={"source": "policy", "doc_id": "ppe", "topics": ["safety"], "authority": 5},
    )
]


def fake_retrieve(
    monkeypatch: pytest.MonkeyPatch, chunks: list[ContextChunk] | None = None
) -> list[str]:
    """Retrieval finds `chunks` (FOUND by default); returns the queries searched."""
    queries: list[str] = []
    found = FOUND if chunks is None else chunks

    async def retrieve(
        self: RagService,
        query: str,
        enabled_collections: list[str] | None = None,
        timer: StageTimer | None = None,
    ) -> list[ContextChunk]:
        queries.append(query)
        if timer is not None:
            timer.add("qdrant", 5.0)
            timer.add("rerank", 7.0)
        return list(found)

    monkeypatch.setattr(RagService, "retrieve", retrieve)
    return queries


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
    fake_retrieve(monkeypatch)

    with make_client() as client:
        response = client.post("/api/chat", json={"question": "What PPE do I need?"})
        assert response.status_code == 200
        body = response.json()
        assert body["metrics"]["completion_tokens"] == 5
        assert body["context"][0]["text"] == "Safety glasses are required."
        assert (body["citations"], body["answered"]) == ([1], True)
        assert body["search_query"] == "What PPE do I need?"
        assert isinstance(body["history_id"], int)

        rows = client.get("/api/history?limit=1").json()
    assert rows[0]["question"] == "What PPE do I need?"
    assert isinstance(rows[0]["prompt_id"], int)
    assert rows[0]["rag_collections"] == ["hs_member"]
    assert (rows[0]["doc_ids"], rows[0]["topics"], rows[0]["answered"]) == (
        ["policy:ppe"],
        ["safety"],
        True,
    )


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
    fake_retrieve(monkeypatch, [])
    with make_client() as client:
        services: server_module.Services = cast(FastAPI, client.app).state.services
        services.topics._cache_occurrences = [occurrence]
        response = client.post("/api/chat", json={"question": "what events are upcoming"})
    assert response.status_code == 200
    (context,) = fake_llm
    assert [chunk.text for chunk in context] == [
        "Laser training at 2100-01-01T00:00:00Z (open_to=members)"
    ]
    assert context[0].metadata["source"] == "event"


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
        chat_resp = client.post("/api/chat", json={"question": "private", "temporary_chat": True})
        assert chat_resp.status_code == 200
        assert chat_resp.json()["metrics"]["history_saved"] is False
        assert client.get("/api/history?limit=10").json() == []


def test_chat_endpoint_returns_502_when_llm_fails(
    make_client: MakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_retrieve(monkeypatch)

    async def failing_chat(
        self: LlmService,
        *,
        settings: AppSettings,
        question: str,
        context: list[ContextChunk],
        history: list[ChatTurn] | None = None,
    ) -> LlmResult:
        raise LlmError("model transport failed")

    monkeypatch.setattr(LlmService, "chat", failing_chat)

    with make_client() as client:
        response = client.post("/api/chat", json={"question": "ping"})
    assert response.status_code == 502
    body = response.json()
    assert body["detail"] == "model transport failed"
    assert list(body["timings"])[-1] == "total"
    assert body["timings"]["qdrant"] == 5.0
    assert "llm" in body["timings"]
    assert body["timing_stats"]["errors"] == 1


def test_root_serves_the_ui(make_client: MakeClient) -> None:
    with make_client() as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    # Scripts carry a hash of their contents, so browsers never run a stale one.
    version = server_module.asset_version()
    assert f'/static/app.js?v={version}"' in response.text
    assert f'/static/markdown.js?v={version}"' in response.text
    assert "?v=dev" not in response.text


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


def test_no_sources_gives_the_no_answer_message_without_the_model(
    make_client: MakeClient, monkeypatch: pytest.MonkeyPatch, fake_llm: list[list[ContextChunk]]
) -> None:
    fake_retrieve(monkeypatch, [])
    with make_client(no_answer_message="Ask in #general.") as client:
        body = client.post("/api/chat", json={"question": "Where is the moon rock?"}).json()
        streamed = client.post(
            "/api/chat", json={"question": "Where is the moon rock?", "stream": True}
        )
        rows = client.get("/api/history?limit=1").json()
    assert fake_llm == []
    assert (body["answer"], body["answered"], body["context"]) == ("Ask in #general.", False, [])
    (event,) = [json.loads(line) for line in streamed.text.splitlines()]
    assert (event["type"], event["answer"]) == ("done", "Ask in #general.")
    assert rows[0]["answered"] is False


def test_answer_without_citations_is_not_answered(
    make_client: MakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_retrieve(monkeypatch)

    async def chat(self: LlmService, **kwargs: Any) -> LlmResult:  # Any: chat's keywords.
        return llm_result("I think so.")

    monkeypatch.setattr(LlmService, "chat", chat)
    with make_client() as client:
        body = client.post("/api/chat", json={"question": "q"}).json()
    assert (body["citations"], body["answered"]) == ([], False)


def test_history_is_rewritten_into_the_search_query_and_sent_to_the_model(
    make_client: MakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    queries = fake_retrieve(monkeypatch)
    seen: list[list[ChatTurn]] = []

    async def rewrite_query(
        self: LlmService, *, settings: AppSettings, question: str, history: list[ChatTurn]
    ) -> str:
        return f"{history[-1].content} / {question}"

    async def chat(
        self: LlmService,
        *,
        settings: AppSettings,
        question: str,
        context: list[ContextChunk],
        history: list[ChatTurn] | None = None,
    ) -> LlmResult:
        seen.append(history or [])
        return llm_result()

    monkeypatch.setattr(LlmService, "rewrite_query", rewrite_query)
    monkeypatch.setattr(LlmService, "chat", chat)
    history = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "older answer"},
        {"role": "user", "content": "the laser"},
    ]
    with make_client(chat_history_turns=2) as client:
        body = client.post("/api/chat", json={"question": "training?", "history": history}).json()
    assert queries == ["the laser / training?"]
    assert body["search_query"] == "the laser / training?"
    assert [turn.content for turn in seen[0]] == ["older answer", "the laser"]


def test_streamed_chat_sends_deltas_then_the_response(
    make_client: MakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_retrieve(monkeypatch)

    async def chat_stream(
        self: LlmService, *, state: StreamState, **kwargs: Any
    ) -> AsyncIterator[str]:  # Any: chat_stream's keywords.
        for piece in ("Wear ", "glasses [1]."):
            state.parts.append(piece)
            yield piece
        state.result = llm_result("".join(state.parts))

    monkeypatch.setattr(LlmService, "chat_stream", chat_stream)
    with make_client() as client:
        response = client.post("/api/chat", json={"question": "PPE?", "stream": True})
        rows = client.get("/api/history?limit=1").json()
    assert response.headers["content-type"].startswith("application/x-ndjson")
    events = [json.loads(line) for line in response.text.splitlines()]
    assert [event["type"] for event in events] == ["delta", "delta", "done"]
    assert events[-1]["answer"] == "Wear glasses [1]."
    assert events[-1]["citations"] == [1]
    assert rows[0]["answer"] == "Wear glasses [1]."


def test_streamed_chat_reports_model_errors(
    make_client: MakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_retrieve(monkeypatch)

    async def chat_stream(
        self: LlmService, *, state: StreamState, **kwargs: Any
    ) -> AsyncIterator[str]:  # Any: chat_stream's keywords.
        raise LlmError("model down")
        yield ""

    monkeypatch.setattr(LlmService, "chat_stream", chat_stream)
    with make_client() as client:
        response = client.post("/api/chat", json={"question": "PPE?", "stream": True})
        assert client.get("/api/history?limit=1").json() == []
    events = [json.loads(line) for line in response.text.splitlines()]
    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert events[0]["detail"] == "model down"
    assert {"rewrite", "qdrant", "rerank", "llm", "total"} <= set(events[0]["timings"])
    assert events[0]["timing_stats"]["chats"] == 1
    assert events[0]["timing_stats"]["errors"] == 1


def test_reports_list_unanswered_questions_and_negative_feedback(
    make_client: MakeClient, monkeypatch: pytest.MonkeyPatch, fake_llm: list[list[ContextChunk]]
) -> None:
    with make_client() as client:
        fake_retrieve(monkeypatch, [])
        client.post("/api/chat", json={"question": "Where is the moon rock?"})
        client.post("/api/chat", json={"question": "Who has the van keys?"})
        fake_retrieve(monkeypatch)
        answered = client.post("/api/chat", json={"question": "PPE?"}).json()
        client.post(
            "/api/feedback",
            json={
                "rating": "down",
                "text": "out of date",
                "history_id": answered["history_id"],
                "question": "PPE?",
                "answer": answered["answer"],
            },
        )
        client.post("/api/feedback", json={"rating": "up", "question": "x"})
        unanswered = client.get("/api/reports/unanswered?days=7").json()
        negative = client.get("/api/reports/negative-feedback").json()
    assert unanswered["total"] == 2
    (group,) = unanswered["groups"]
    assert group["topic"] == "unclassified"
    assert [item["question"] for item in group["questions"]] == [
        "Where is the moon rock?",
        "Who has the van keys?",
    ]
    (item,) = negative["items"]
    assert (item["note"], item["question"], item["doc_ids"]) == (
        "out of date",
        "PPE?",
        ["policy:ppe"],
    )


def test_unanswered_report_groups_by_topic(
    make_client: MakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_retrieve(monkeypatch)

    async def chat(self: LlmService, **kwargs: Any) -> LlmResult:  # Any: chat's keywords.
        return llm_result("No citation.")

    monkeypatch.setattr(LlmService, "chat", chat)
    with make_client() as client:
        client.post("/api/chat", json={"question": "a"})
        client.post("/api/chat", json={"question": "b"})
        report = client.get("/api/reports/unanswered").json()
    assert [(group["topic"], len(group["questions"])) for group in report["groups"]] == [
        ("safety", 2)
    ]


def test_chat_reports_stage_timings_and_running_averages(
    make_client: MakeClient, monkeypatch: pytest.MonkeyPatch, fake_llm: list[list[ContextChunk]]
) -> None:
    fake_retrieve(monkeypatch)
    with make_client() as client:
        first = client.post("/api/chat", json={"question": "PPE?"}).json()
        second = client.post("/api/chat", json={"question": "PPE?"}).json()
        stats = client.get("/api/timings").json()
        reset = client.delete("/api/timings").json()
        after = client.get("/api/timings").json()
        saved = client.get("/api/history?limit=2").json()

    timings = first["timings"]
    assert list(timings) == ["rewrite", "qdrant", "rerank", "llm", "finish", "total"]
    assert timings["qdrant"] == 5.0
    assert timings["total"] >= timings["llm"]
    assert first["timing_stats"]["chats"] == 1
    assert second["timing_stats"]["chats"] == 2

    assert stats["chats"] == 2
    assert stats["errors"] == 0
    by_stage = {row["stage"]: row for row in stats["stages"]}
    assert [row["stage"] for row in stats["stages"]] == list(timings)
    assert by_stage["qdrant"] == {
        "stage": "qdrant",
        "count": 2,
        "average_ms": 5.0,
        "min_ms": 5.0,
        "max_ms": 5.0,
        "last_ms": 5.0,
    }
    assert reset == after == {"chats": 0, "errors": 0, "since": None, "stages": []}
    # A saved chat keeps the timings it was answered with.
    assert saved[0]["timings"] == second["timings"]
    assert saved[1]["timings"] == timings
    assert "timings" not in saved[0]["config_snapshot"]["last_metrics"]


def test_streamed_chat_done_event_carries_timings(
    make_client: MakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_retrieve(monkeypatch)

    async def chat_stream(
        self: LlmService, *, state: StreamState, **kwargs: Any
    ) -> AsyncIterator[str]:  # Any: chat_stream's keywords.
        yield "ok [1]"
        state.parts.append("ok [1]")
        state.result = llm_result()
        state.result.first_token_ms = 12.5
        state.result.phases = {"prompt": 2.5, "thinking": 10.0, "writing": 30.0}

    monkeypatch.setattr(LlmService, "chat_stream", chat_stream)
    with make_client() as client:
        response = client.post("/api/chat", json={"question": "PPE?", "stream": True})
    done = [json.loads(line) for line in response.text.splitlines()][-1]
    assert done["type"] == "done"
    assert done["timings"]["first_token"] == 12.5
    assert [name for name in done["timings"] if name.startswith("llm")] == [
        "llm",
        "llm:prompt",
        "llm:thinking",
        "llm:writing",
    ]
    assert done["timings"]["llm:thinking"] == 10.0
    assert list(done["timings"])[-1] == "total"
    assert done["timing_stats"]["chats"] == 1
