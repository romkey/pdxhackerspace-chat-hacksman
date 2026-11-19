import asyncio
import json
from collections.abc import Callable
from datetime import UTC, datetime
from types import SimpleNamespace

# Any: fake Qdrant client calls take the real client's keyword arguments.
from typing import Any

import httpx
import pytest
from qdrant_client.http.models import (
    Distance,
    Document,
    Filter,
    FusionQuery,
    VectorParams,
)

from chat_hacksman import rag as rag_module
from chat_hacksman.models import ContextChunk
from chat_hacksman.rag import RagService, select_chunks
from chat_hacksman.rerank import Reranker
from tests.conftest import Handler


def _service(**overrides: Any) -> RagService:  # Any: RagService fields have mixed types.
    service = RagService(
        qdrant_url="http://localhost:6333",
        qdrant_api_key=None,
        collections=["hs_member", "hs_catalog"],
        embedding_url="http://localhost:11434",
        embedding_model="nomic-embed-text",
        candidates=50,
        top_n=8,
    )
    for name, value in overrides.items():
        setattr(service, name, value)
    return service


def _hit(text: str, score: float, **payload: object) -> SimpleNamespace:
    return SimpleNamespace(payload={"text": text, "tier": "member", **payload}, score=score)


class FakeClient:
    """Records query_points calls; answers each collection with its hits."""

    def __init__(
        self,
        hits: dict[str, list[SimpleNamespace]],
        dims: dict[str, int] | None = None,
        fail_hybrid: bool = False,
    ) -> None:
        self.hits = hits
        self.dims = dims or {}
        self.fail_hybrid = fail_hybrid
        self.calls: list[dict[str, Any]] = []

    def get_collection(self, *, collection_name: str) -> SimpleNamespace:
        size = self.dims.get(collection_name, 3)
        vectors = {"dense": VectorParams(size=size, distance=Distance.COSINE)}
        return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=vectors)))

    def query_points(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        if self.fail_hybrid and "prefetch" in kwargs:
            raise RuntimeError("no BM25 inference here")
        return SimpleNamespace(points=list(self.hits.get(kwargs["collection_name"], [])))


def _install(monkeypatch: pytest.MonkeyPatch, service: RagService, client: FakeClient) -> None:
    async def fake_embed_query(self: RagService, _: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr(RagService, "_embed_query", fake_embed_query)
    monkeypatch.setattr(RagService, "_make_qdrant_client", lambda self: client)


def test_hybrid_search_fuses_dense_and_sparse(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _service()
    client = FakeClient({"hs_member": [_hit("Badge in at the front door.", 0.03)]})
    _install(monkeypatch, service, client)

    results = asyncio.run(service.retrieve("how do I get in?"))

    assert [chunk.text for chunk in results] == ["Badge in at the front door."]
    assert [call["collection_name"] for call in client.calls] == ["hs_member", "hs_catalog"]
    call = client.calls[0]
    assert isinstance(call["query"], FusionQuery)
    assert call["limit"] == 50
    dense, sparse = call["prefetch"]
    assert (dense.using, dense.query, dense.limit) == ("dense", [0.1, 0.2, 0.3], 50)
    assert sparse.using == "sparse"
    assert sparse.query == Document(text="how do I get in?", model="qdrant/bm25")
    assert dense.filter == sparse.filter == service.base_filter(_now_of(dense.filter))


def _now_of(query_filter: Filter) -> datetime:
    moment = query_filter.model_dump()["must"][2]["should"][1]["range"]["gt"]
    assert isinstance(moment, datetime)
    return moment


def test_dense_only_search_without_sparse_vector(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _service(sparse_vector="", min_score=0.5)
    client = FakeClient({"hs_member": [_hit("kept", 0.8), _hit("weak", 0.4)]})
    _install(monkeypatch, service, client)

    results = asyncio.run(service.retrieve("q", enabled_collections=["hs_member"]))

    assert [chunk.text for chunk in results] == ["kept"]
    (call,) = client.calls
    assert "prefetch" not in call
    assert (call["query"], call["using"]) == ([0.1, 0.2, 0.3], "dense")


def test_hybrid_failure_falls_back_to_dense(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _service()
    client = FakeClient({"hs_member": [_hit("found", 0.9)]}, fail_hybrid=True)
    _install(monkeypatch, service, client)

    results = asyncio.run(service.retrieve("q", enabled_collections=["hs_member"]))

    assert [chunk.text for chunk in results] == ["found"]
    assert ["prefetch" in call for call in client.calls] == [True, False]


def test_base_filter_excludes_other_tiers_superseded_and_stale() -> None:
    moment = datetime(2026, 11, 17, tzinfo=UTC)
    dumped = _service().base_filter(moment).model_dump(exclude_none=True)
    tier, superseded, stale = dumped["must"]
    assert tier == {"key": "tier", "match": {"any": ["public", "member"]}}
    assert superseded == {"is_empty": {"key": "superseded_by"}}
    assert stale == {
        "should": [
            {"is_empty": {"key": "stale_after"}},
            {"key": "stale_after", "range": {"gt": moment}},
        ]
    }


def test_retrieve_skips_unknown_collections_and_dimension_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    client = FakeClient(
        {"hs_member": [_hit("a", 0.9)], "hs_catalog": [_hit("b", 0.9)]}, dims={"hs_catalog": 768}
    )
    _install(monkeypatch, service, client)

    results = asyncio.run(
        service.retrieve("q", enabled_collections=["hs_member", "hs_catalog", "x"])
    )

    assert [chunk.collection for chunk in results] == ["hs_member"]
    assert [call["collection_name"] for call in client.calls] == ["hs_member"]
    assert asyncio.run(service.retrieve("q", enabled_collections=["x"])) == []


class NotFoundError(Exception):
    status_code = 404


def test_missing_collections_are_skipped_then_looked_for_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    client = FakeClient({"hs_member": [_hit("a", 0.9)]})
    lookups: list[str] = []
    real_get_collection = client.get_collection

    def get_collection(*, collection_name: str) -> SimpleNamespace:
        lookups.append(collection_name)
        if collection_name == "hs_catalog":
            raise NotFoundError("Not found: Collection `hs_catalog` doesn't exist!")
        return real_get_collection(collection_name=collection_name)

    client.get_collection = get_collection  # type: ignore[method-assign]
    _install(monkeypatch, service, client)
    clock = [1000.0]
    monkeypatch.setattr(rag_module, "_monotonic", lambda: clock[0])

    assert [chunk.text for chunk in asyncio.run(service.retrieve("q"))] == ["a"]
    assert [chunk.text for chunk in asyncio.run(service.retrieve("q"))] == ["a"]
    assert lookups == ["hs_member", "hs_catalog"]
    assert [call["collection_name"] for call in client.calls] == ["hs_member", "hs_member"]

    clock[0] += rag_module.MISSING_COLLECTION_RECHECK_SECONDS
    asyncio.run(service.retrieve("q"))
    assert lookups == ["hs_member", "hs_catalog", "hs_catalog"]


def test_collection_deleted_after_its_lookup_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _service()
    client = FakeClient({"hs_member": [_hit("a", 0.9)]})
    catalog_queries: list[dict[str, Any]] = []

    def query_points(**kwargs: Any) -> SimpleNamespace:
        if kwargs["collection_name"] == "hs_catalog":
            catalog_queries.append(kwargs)
            raise NotFoundError("gone")
        return FakeClient.query_points(client, **kwargs)

    client.query_points = query_points  # type: ignore[method-assign]
    _install(monkeypatch, service, client)
    assert [chunk.text for chunk in asyncio.run(service.retrieve("q"))] == ["a"]
    assert [chunk.text for chunk in asyncio.run(service.retrieve("q"))] == ["a"]
    # One hybrid query: not retried as dense search, and not searched again.
    assert len(catalog_queries) == 1
    assert "prefetch" in catalog_queries[0]


def test_retrieve_drops_points_without_text_and_survives_search_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(sparse_vector="")
    client = FakeClient({"hs_member": [SimpleNamespace(payload={"title": "no text"}, score=0.9)]})

    def query_points(**kwargs: Any) -> SimpleNamespace:
        if kwargs["collection_name"] == "hs_catalog":
            raise RuntimeError("down")
        return FakeClient.query_points(client, **kwargs)

    client.query_points = query_points  # type: ignore[method-assign]
    _install(monkeypatch, service, client)
    assert asyncio.run(service.retrieve("q")) == []


def test_retrieve_returns_nothing_when_embedding_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    async def no_embedding(self: RagService, _: str) -> None:
        return None

    monkeypatch.setattr(RagService, "_embed_query", no_embedding)
    assert asyncio.run(_service().retrieve("q")) == []


def test_reranks_candidates_from_every_collection_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[list[str]] = []

    async def rerank(self: Reranker, query: str, documents: list[str]) -> list[float]:
        sent.append(documents)
        return [0.1 if "member" in doc else 0.9 for doc in documents]

    monkeypatch.setattr(Reranker, "rerank", rerank)
    service = _service(
        reranker=Reranker(url="http://rerank.local", model="m"), top_n=1, rerank_min_score=0.05
    )
    client = FakeClient(
        {
            "hs_member": [_hit("member text", 0.05, title="Wiki", heading_path="Access")],
            "hs_catalog": [_hit("catalog text", 0.01)],
        }
    )
    _install(monkeypatch, service, client)

    results = asyncio.run(service.retrieve("q"))

    assert sent == [["Wiki > Access\nmember text", "catalog text"]]
    assert [(chunk.text, chunk.rerank_score) for chunk in results] == [("catalog text", 0.9)]


def test_rerank_min_score_can_leave_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    async def rerank(self: Reranker, query: str, documents: list[str]) -> list[float]:
        return [-5.0 for _ in documents]

    monkeypatch.setattr(Reranker, "rerank", rerank)
    service = _service(
        reranker=Reranker(url="http://rerank.local", model="m"), rerank_min_score=0.0
    )
    _install(monkeypatch, service, FakeClient({"hs_member": [_hit("weak", 0.02)]}))
    assert asyncio.run(service.retrieve("q")) == []


def test_reranker_failure_keeps_search_order(monkeypatch: pytest.MonkeyPatch) -> None:
    async def rerank(self: Reranker, query: str, documents: list[str]) -> None:
        return None

    monkeypatch.setattr(Reranker, "rerank", rerank)
    service = _service(reranker=Reranker(url="http://rerank.local", model="m"))
    _install(
        monkeypatch, service, FakeClient({"hs_member": [_hit("second", 0.02), _hit("first", 0.03)]})
    )
    assert [chunk.text for chunk in asyncio.run(service.retrieve("q"))] == ["first", "second"]


def _chunk(text: str, rerank_score: float, authority: int | None = None) -> ContextChunk:
    metadata: dict[str, object] = {} if authority is None else {"authority": authority}
    return ContextChunk(
        collection="hs_member", score=0.0, text=text, rerank_score=rerank_score, metadata=metadata
    )


def test_authority_breaks_near_ties() -> None:
    chunks = [_chunk("slack fact", 0.81, 1), _chunk("wiki page", 0.80, 4), _chunk("low", 0.0, 5)]
    assert [chunk.text for chunk in select_chunks(chunks, 1, 0.1)] == ["wiki page"]
    # Without the authority weight, relevance alone decides.
    assert [chunk.text for chunk in select_chunks(chunks, 1, 0.0)] == ["slack fact"]


def test_authority_does_not_beat_much_better_relevance() -> None:
    chunks = [_chunk("manual", 0.9, 2), _chunk("sign", 0.3, 5), _chunk("other", 0.0, 1)]
    assert select_chunks(chunks, 1, 0.1)[0].text == "manual"


def test_relevant_policy_chunk_is_always_included() -> None:
    chunks = [
        _chunk("a", 1.0),
        _chunk("b", 0.95),
        _chunk("c", 0.9),
        _chunk("policy", 0.6, 5),
        _chunk("irrelevant policy", 0.1, 5),
        _chunk("floor", 0.0),
    ]
    assert [chunk.text for chunk in select_chunks(chunks, 3, 0.0)] == ["a", "b", "policy"]


def test_irrelevant_policy_chunk_is_not_forced_in() -> None:
    chunks = [_chunk("a", 1.0), _chunk("b", 0.9), _chunk("policy", 0.2, 5), _chunk("z", 0.0)]
    assert [chunk.text for chunk in select_chunks(chunks, 2, 0.0)] == ["a", "b"]


def test_select_chunks_edge_cases() -> None:
    assert select_chunks([], 8, 0.1) == []
    same = [_chunk("x", 0.5, 7), _chunk("y", 0.5, True)]
    # Equal relevance: authority (capped at 5; a bool isn't one) decides.
    assert [chunk.text for chunk in select_chunks(same, 2, 0.1)] == ["x", "y"]


def test_reuses_one_qdrant_client(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _service()
    made: list[FakeClient] = []

    def make(self: RagService) -> FakeClient:
        made.append(FakeClient({}))
        return made[-1]

    async def fake_embed_query(self: RagService, _: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr(RagService, "_embed_query", fake_embed_query)
    monkeypatch.setattr(RagService, "_make_qdrant_client", make)
    asyncio.run(service.retrieve("one"))
    asyncio.run(service.retrieve("two"))
    assert len(made) == 1


def test_qdrant_client_sends_sparse_queries_to_the_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    made: list[dict[str, Any]] = []
    monkeypatch.setattr(rag_module, "QdrantClient", lambda **kwargs: made.append(kwargs))
    _service()._make_qdrant_client()
    _service(sparse_vector="")._make_qdrant_client()
    assert [kwargs["cloud_inference"] for kwargs in made] == [True, False]


def test_vector_dim_for_unnamed_and_named_vectors() -> None:
    def info(vectors: object) -> SimpleNamespace:
        return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=vectors)))

    unnamed = VectorParams(size=768, distance=Distance.COSINE)
    assert _service(dense_vector="")._extract_vector_dim(info(unnamed)) == 768
    assert _service()._extract_vector_dim(info(unnamed)) is None
    assert _service()._extract_vector_dim(info({"other": unnamed})) is None
    assert _service()._extract_vector_dim(info(None)) is None


def _embedding_response(vector: list[float]) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": vector}],
            "model": "nomic-ai/modernbert-embed-base",
        },
    )


def test_embed_query_calls_openai_embeddings_endpoint(
    mock_http: Callable[[Handler], None],
) -> None:
    service = _service()
    service.embedding_url = "http://embed.local/v1"
    service.embedding_api_key = "dummy-embedding-key"
    service.embedding_query_prefix = "search_query: "
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _embedding_response([0.5, 0.6])

    mock_http(handler)

    assert asyncio.run(service._embed_query("hello")) == [0.5, 0.6]
    (request,) = seen
    assert request.method == "POST"
    assert str(request.url) == "http://embed.local/v1/embeddings"
    assert request.headers["Authorization"] == "Bearer dummy-embedding-key"
    assert json.loads(request.content) == {
        "model": "nomic-embed-text",
        "input": ["search_query: hello"],
        "encoding_format": "float",
    }


def test_embed_query_sends_no_auth_header_without_api_key(
    mock_http: Callable[[Handler], None],
) -> None:
    service = _service()
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _embedding_response([1.0])

    mock_http(handler)
    assert asyncio.run(service._embed_query("hello")) == [1.0]
    assert "Authorization" not in seen[0].headers
    assert json.loads(seen[0].content)["input"] == ["hello"]


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500, text="boom"),
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"data": []}),
        httpx.Response(200, json={"data": [{"embedding": []}]}),
        httpx.Response(200, json=[1, 2]),
    ],
)
def test_embed_query_returns_none_on_bad_responses(
    mock_http: Callable[[Handler], None], response: httpx.Response
) -> None:
    mock_http(lambda request: response)
    assert asyncio.run(_service()._embed_query("hello")) is None


def test_embed_query_returns_none_when_server_is_down(
    mock_http: Callable[[Handler], None],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    mock_http(handler)
    assert asyncio.run(_service()._embed_query("hello")) is None


def _capture_embedding_input(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    captured: dict[str, str] = {}

    async def fake_request(self: RagService, text: str) -> list[float]:
        captured["text"] = text
        return [0.5, 0.6]

    monkeypatch.setattr(RagService, "_request_embedding", fake_request)
    return captured


def test_embed_query_truncates_to_embedding_context_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    service.embedding_context_length = 3
    captured = _capture_embedding_input(monkeypatch)

    assert asyncio.run(service._embed_query("one two three four five")) == [0.5, 0.6]
    assert captured["text"] == "one two three"


def test_embed_query_keeps_prefix_whole_when_truncating(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _service()
    service.embedding_context_length = 4
    service.embedding_query_prefix = "query: "
    captured = _capture_embedding_input(monkeypatch)

    asyncio.run(service._embed_query("one two three four five"))
    assert captured["text"] == "query: one two three"


def test_embed_query_truncates_long_single_token_input(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _service()
    service.embedding_context_length = 2
    captured = _capture_embedding_input(monkeypatch)

    asyncio.run(service._embed_query("x" * 1000))
    assert len(captured["text"]) == 128
