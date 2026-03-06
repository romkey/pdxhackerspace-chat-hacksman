from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.rag import RagService, _extract_chunk_text


def _service() -> RagService:
    return RagService(
        qdrant_url="http://localhost:6333",
        qdrant_api_key=None,
        collections=["wiki"],
        embedding_url="http://localhost:11434",
        embedding_model="nomic-embed-text",
        top_k=2,
    )


def test_rag_retrieve_works_with_legacy_search_api(monkeypatch) -> None:
    service = _service()

    async def fake_embed_query(self, _: str):  # noqa: ANN001
        del self
        return [0.1, 0.2, 0.3]

    class FakeClient:
        def search(self, *, collection_name, query_vector, with_payload, limit):
            del query_vector, with_payload, limit
            return [
                SimpleNamespace(
                    payload={"text": f"Result from {collection_name}"},
                    score=0.9,
                )
            ]

    monkeypatch.setattr(type(service), "_embed_query", fake_embed_query)
    monkeypatch.setattr(type(service), "_make_qdrant_client", lambda self: FakeClient())

    results = asyncio.run(service.retrieve("test query", enabled_collections=["wiki"]))
    assert len(results) == 1
    assert results[0].collection == "wiki"
    assert "Result from wiki" in results[0].text


def test_rag_retrieve_works_with_query_points_api(monkeypatch) -> None:
    service = _service()

    async def fake_embed_query(self, _: str):  # noqa: ANN001
        del self
        return [0.1, 0.2, 0.3]

    class FakeClient:
        def query_points(self, *, collection_name, query, with_payload, limit):
            del query, with_payload, limit
            return SimpleNamespace(
                points=[
                    SimpleNamespace(
                        payload={"text": f"Point from {collection_name}"},
                        score=0.8,
                    )
                ]
            )

    monkeypatch.setattr(type(service), "_embed_query", fake_embed_query)
    monkeypatch.setattr(type(service), "_make_qdrant_client", lambda self: FakeClient())

    results = asyncio.run(service.retrieve("test query", enabled_collections=["wiki"]))
    assert len(results) == 1
    assert results[0].collection == "wiki"
    assert "Point from wiki" in results[0].text


def test_rag_retrieve_skips_collection_when_vector_dim_mismatch(monkeypatch) -> None:
    service = _service()

    async def fake_embed_query(self, _: str):  # noqa: ANN001
        del self
        return [0.1, 0.2, 0.3, 0.4]

    class FakeClient:
        def get_collection(self, *, collection_name):
            del collection_name
            return SimpleNamespace(
                config=SimpleNamespace(
                    params=SimpleNamespace(vectors=SimpleNamespace(size=3))
                )
            )

        def query_points(self, *, collection_name, query, with_payload, limit):
            del collection_name, query, with_payload, limit
            raise AssertionError("query_points should not be called on dimension mismatch")

    monkeypatch.setattr(type(service), "_embed_query", fake_embed_query)
    monkeypatch.setattr(type(service), "_make_qdrant_client", lambda self: FakeClient())

    results = asyncio.run(service.retrieve("test query", enabled_collections=["wiki"]))
    assert results == []


def test_extract_chunk_text_prefers_chunk_text() -> None:
    payload = {"text": "fallback", "chunk_text": "ebook body"}
    assert _extract_chunk_text(payload) == "ebook body"


def test_extract_chunk_text_prefers_slack_summary_for_thread_summary() -> None:
    payload = {
        "doc_type": "thread_summary",
        "channel_name": "ops",
        "summary": "Summary of the thread",
        "text": "Raw thread message",
    }
    assert _extract_chunk_text(payload) == "Summary of the thread"


def test_extract_chunk_text_does_not_treat_doc_type_alone_as_slack() -> None:
    payload = {
        "doc_type": "event_summary",
        "chunk_text": "Canonical event text",
        "summary": "This should not override chunk_text",
    }
    assert _extract_chunk_text(payload) == "Canonical event text"


def test_embed_query_uses_modern_embed_endpoint_first(monkeypatch) -> None:
    service = _service()

    async def fake_modern(self, query):  # noqa: ANN001
        del self, query
        return [0.5, 0.6]

    async def fake_legacy(self, query):  # noqa: ANN001
        del self, query
        raise AssertionError("legacy endpoint should not be called when modern succeeds")

    monkeypatch.setattr(type(service), "_embed_query_modern", fake_modern)
    monkeypatch.setattr(type(service), "_embed_query_legacy", fake_legacy)

    vector = asyncio.run(service._embed_query("hello"))
    assert vector == [0.5, 0.6]


def test_embed_query_falls_back_to_legacy_endpoint(monkeypatch) -> None:
    service = _service()

    async def fake_modern(self, query):  # noqa: ANN001
        del self, query
        return None

    async def fake_legacy(self, query):  # noqa: ANN001
        del self, query
        return [0.7, 0.8]

    monkeypatch.setattr(type(service), "_embed_query_modern", fake_modern)
    monkeypatch.setattr(type(service), "_embed_query_legacy", fake_legacy)

    vector = asyncio.run(service._embed_query("hello"))
    assert vector == [0.7, 0.8]


def test_embed_query_truncates_to_embedding_context_length(monkeypatch) -> None:
    service = _service()
    service.embedding_context_length = 3

    captured: dict[str, str] = {}

    async def fake_modern(self, query):  # noqa: ANN001
        del self
        captured["query"] = query
        return [0.5, 0.6]

    monkeypatch.setattr(type(service), "_embed_query_modern", fake_modern)
    monkeypatch.setattr(type(service), "_embed_query_legacy", fake_modern)

    vector = asyncio.run(service._embed_query("one two three four five"))
    assert vector == [0.5, 0.6]
    assert captured["query"] == "one two three"


def test_embed_query_truncates_long_single_token_input(monkeypatch) -> None:
    service = _service()
    service.embedding_context_length = 2

    captured: dict[str, str] = {}

    async def fake_modern(self, query):  # noqa: ANN001
        del self
        captured["query"] = query
        return [0.5, 0.6]

    monkeypatch.setattr(type(service), "_embed_query_modern", fake_modern)
    monkeypatch.setattr(type(service), "_embed_query_legacy", fake_modern)

    vector = asyncio.run(service._embed_query("x" * 1000))
    assert vector == [0.5, 0.6]
    assert len(captured["query"]) == 128


def test_rag_retrieve_applies_min_score(monkeypatch) -> None:
    service = _service()
    service.min_score = 0.5

    async def fake_embed_query(self, _: str):  # noqa: ANN001
        del self
        return [0.1, 0.2, 0.3]

    class FakeClient:
        def query_points(self, *, collection_name, query, query_filter, with_payload, limit):
            del collection_name, query, query_filter, with_payload, limit
            return SimpleNamespace(
                points=[
                    SimpleNamespace(payload={"text": "low score"}, score=0.2),
                    SimpleNamespace(payload={"text": "high score"}, score=0.9),
                ]
            )

    monkeypatch.setattr(type(service), "_embed_query", fake_embed_query)
    monkeypatch.setattr(type(service), "_make_qdrant_client", lambda self: FakeClient())

    results = asyncio.run(service.retrieve("test query", enabled_collections=["wiki"]))
    assert len(results) == 1
    assert results[0].text == "high score"


def test_calibre_filter_always_includes_library_summary() -> None:
    service = _service()
    query_filter = service._build_collection_filter("calibre_books", "find me books on electronics")
    assert query_filter is not None
    assert query_filter.must
    condition = query_filter.must[0]
    match_any = getattr(condition, "match", None)
    values = getattr(match_any, "any", [])
    assert "library_summary" in values


def test_slack_filter_targets_channel_and_workspace_for_channel_queries() -> None:
    service = _service()
    query_filter = service._build_collection_filter("slack", "what channels should I join")
    assert query_filter is not None
    assert query_filter.must
    condition = query_filter.must[0]
    assert getattr(condition, "key", "") == "doc_type"
    match_any = getattr(condition, "match", None)
    values = getattr(match_any, "any", [])
    assert "channel_summary" in values
    assert "workspace_summary" in values


def test_slack_filter_returns_none_when_no_intent_words() -> None:
    service = _service()
    query_filter = service._build_collection_filter("slack", "compressor maintenance schedule")
    assert query_filter is None


def test_events_filter_future_uses_start_time_range() -> None:
    service = _service()
    query_filter = service._build_collection_filter("events", "what upcoming events are next")
    assert query_filter is not None
    assert query_filter.must
    condition = query_filter.must[0]
    assert getattr(condition, "key", "") == "start_time"
    value_range = getattr(condition, "range", None)
    assert getattr(value_range, "gte", None) is not None


def test_events_filter_prefers_event_summary_for_schedule_queries() -> None:
    service = _service()
    query_filter = service._build_collection_filter("events", "what is the regular schedule")
    assert query_filter is not None
    keys = [getattr(item, "key", "") for item in query_filter.must]
    assert "record_type" in keys
    record_condition = next(
        item for item in query_filter.must if getattr(item, "key", "") == "record_type"
    )
    values = getattr(getattr(record_condition, "match", None), "any", [])
    assert "event_summary" in values


def test_events_filter_prefers_occurrence_for_specific_past_query() -> None:
    service = _service()
    query_filter = service._build_collection_filter("events", "what happened that tuesday")
    assert query_filter is not None
    keys = [getattr(item, "key", "") for item in query_filter.must]
    assert "record_type" in keys
    record_condition = next(
        item for item in query_filter.must if getattr(item, "key", "") == "record_type"
    )
    values = getattr(getattr(record_condition, "match", None), "any", [])
    assert "occurrence" in values


def test_events_filter_supports_hyphenated_collection_names() -> None:
    service = _service()
    query_filter = service._build_collection_filter(
        "events-community", "what upcoming events are next"
    )
    assert query_filter is not None


def test_rag_expands_search_limit_when_min_score_enabled() -> None:
    service = _service()
    service.top_k = 3
    service.min_score = 0.1
    captured: dict[str, int] = {}

    class FakeClient:
        def query_points(self, *, collection_name, query, query_filter, with_payload, limit):
            del collection_name, query, query_filter, with_payload
            captured["limit"] = limit
            return SimpleNamespace(points=[])

    service._search_collection(
        client=FakeClient(),
        collection="wiki",
        vector=[0.1, 0.2],
        query_filter=None,
    )
    assert captured["limit"] == 6


def test_rag_uses_collection_specific_top_k_for_events() -> None:
    service = _service()
    service.top_k = 2
    service.top_k_events = 5
    captured: dict[str, int] = {}

    class FakeClient:
        def query_points(self, *, collection_name, query, query_filter, with_payload, limit):
            del collection_name, query, query_filter, with_payload
            captured["limit"] = limit
            return SimpleNamespace(points=[])

    service._search_collection(
        client=FakeClient(),
        collection="events",
        vector=[0.1, 0.2],
        query_filter=None,
    )
    assert captured["limit"] == 5


def test_rag_reuses_single_qdrant_client(monkeypatch) -> None:
    service = _service()
    calls = {"count": 0}

    async def fake_embed_query(self, _: str):  # noqa: ANN001
        del self
        return [0.1, 0.2, 0.3]

    class FakeClient:
        def query_points(self, *, collection_name, query, query_filter, with_payload, limit):
            del collection_name, query, query_filter, with_payload, limit
            return SimpleNamespace(points=[SimpleNamespace(payload={"text": "ok"}, score=0.9)])

    def fake_make_client(self):  # noqa: ANN001
        del self
        calls["count"] += 1
        return FakeClient()

    monkeypatch.setattr(type(service), "_embed_query", fake_embed_query)
    monkeypatch.setattr(type(service), "_make_qdrant_client", fake_make_client)

    first = asyncio.run(service.retrieve("one", enabled_collections=["wiki"]))
    second = asyncio.run(service.retrieve("two", enabled_collections=["wiki"]))
    assert first and second
    assert calls["count"] == 1


def test_rag_final_limit_sums_collection_specific_limits(monkeypatch) -> None:
    service = _service()
    service.collections = ["events", "slack_general"]
    service.top_k = 2
    service.top_k_events = 3
    service.top_k_slack = 3

    async def fake_embed_query(self, _: str):  # noqa: ANN001
        del self
        return [0.1, 0.2, 0.3]

    class FakeClient:
        def query_points(self, *, collection_name, query, query_filter, with_payload, limit):
            del query, query_filter, with_payload, limit
            return SimpleNamespace(
                points=[
                    SimpleNamespace(payload={"text": f"{collection_name}-1"}, score=0.91),
                    SimpleNamespace(payload={"text": f"{collection_name}-2"}, score=0.90),
                ]
            )

    monkeypatch.setattr(type(service), "_embed_query", fake_embed_query)
    monkeypatch.setattr(type(service), "_make_qdrant_client", lambda self: FakeClient())

    results = asyncio.run(
        service.retrieve("test query", enabled_collections=["events", "slack_general"])
    )
    assert len(results) == 4
