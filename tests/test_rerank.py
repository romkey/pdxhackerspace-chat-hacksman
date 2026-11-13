import asyncio
import json
from collections.abc import Callable
from types import SimpleNamespace

import httpx
import pytest

from chat_hacksman.config import Settings
from chat_hacksman.models import ContextChunk
from chat_hacksman.rag import RagService
from chat_hacksman.rerank import Reranker, truncate_words
from chat_hacksman.server import build_reranker
from tests.conftest import Handler

MockHttp = Callable[[Handler], None]


def jina_response(scores: dict[int, float]) -> httpx.Response:
    """A Jina/Cohere-style response, sorted by score like real servers."""
    results = [
        {"index": index, "relevance_score": score}
        for index, score in sorted(scores.items(), key=lambda item: -item[1])
    ]
    return httpx.Response(200, json={"model": "m", "results": results})


def test_jina_request_and_scores_in_document_order(mock_http: MockHttp) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return jina_response({0: 0.1, 1: 0.9, 2: 0.5})

    mock_http(handler)
    reranker = Reranker(url="http://rerank.local/v1/", model="BAAI/bge-reranker-v2-m3")
    scores = asyncio.run(reranker.rerank(" What is a laser? ", ["a", "b", "c"]))

    assert scores == [0.1, 0.9, 0.5]
    (request,) = seen
    assert str(request.url) == "http://rerank.local/v1/rerank"
    assert "Authorization" not in request.headers
    assert json.loads(request.content) == {
        "model": "BAAI/bge-reranker-v2-m3",
        "query": "What is a laser?",
        "documents": ["a", "b", "c"],
        "top_n": 3,
    }


def test_tei_request_and_scores(mock_http: MockHttp) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=[{"index": 1, "score": 0.8}, {"index": 0, "score": 0.2}])

    mock_http(handler)
    reranker = Reranker(url="http://tei.local", model="ignored", api="tei", api_key="dummy-key")
    assert asyncio.run(reranker.rerank("q", ["x", "y"])) == [0.2, 0.8]
    assert seen[0].headers["Authorization"] == "Bearer dummy-key"
    assert json.loads(seen[0].content) == {"query": "q", "texts": ["x", "y"], "truncate": True}


def test_no_documents_needs_no_request(mock_http: MockHttp) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    mock_http(handler)
    assert asyncio.run(Reranker(url="http://r.local", model="m").rerank("q", [])) == []


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500, text="boom"),
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"results": "nope"}),
        httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.5}]}),
        httpx.Response(200, json={"results": [{"index": 5, "relevance_score": 0.5}]}),
        httpx.Response(200, json={"results": [{"index": 0, "relevance_score": "high"}]}),
        httpx.Response(200, json={"results": [1, 2]}),
        # A TEI-style answer when RERANK_API is jina.
        httpx.Response(200, json=[{"index": 0, "score": 0.5}, {"index": 1, "score": 0.4}]),
    ],
)
def test_bad_responses_return_none(mock_http: MockHttp, response: httpx.Response) -> None:
    mock_http(lambda request: response)
    assert asyncio.run(Reranker(url="http://r.local", model="m").rerank("q", ["a", "b"])) is None


def test_unreachable_server_returns_none(mock_http: MockHttp) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    mock_http(handler)
    assert asyncio.run(Reranker(url="http://r.local", model="m").rerank("q", ["a"])) is None


def test_documents_are_cut_to_fit_the_context(mock_http: MockHttp) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return jina_response({0: 0.5})

    mock_http(handler)
    reranker = Reranker(url="http://r.local", model="m", context_length=8)
    asyncio.run(reranker.rerank("three word query", ["one two three four five six seven"]))
    body = json.loads(seen[0].content)
    # The query gets at most a quarter of the context; the document the rest.
    assert body["query"] == "three word"
    assert body["documents"] == ["one two three four five six"]


def test_truncate_words() -> None:
    assert truncate_words("a b c d", 2) == "a b"
    assert truncate_words("short text", 10) == "short text"
    assert truncate_words("x" * 1000, 2) == "x" * 128


def _service(reranker: Reranker | None, **kwargs: object) -> RagService:
    service = RagService(
        qdrant_url="http://localhost:6333",
        qdrant_api_key=None,
        collections=["wiki", "events"],
        embedding_url="http://localhost:8080/v1",
        embedding_model="m",
        top_k=2,
        top_k_events=1,
        reranker=reranker,
    )
    for key, value in kwargs.items():
        setattr(service, key, value)
    return service


class FakeQdrant:
    """Returns four hits per collection, best vector score first."""

    def __init__(self) -> None:
        self.limits: dict[str, int] = {}

    def query_points(
        self,
        *,
        collection_name: str,
        query: object,
        query_filter: object,
        with_payload: bool,
        limit: int,
    ) -> SimpleNamespace:
        self.limits[collection_name] = limit
        return SimpleNamespace(
            points=[
                SimpleNamespace(payload={"text": f"{collection_name}-{n}"}, score=0.9 - n / 10)
                for n in range(4)
            ]
        )


def _retrieve(
    monkeypatch: pytest.MonkeyPatch, service: RagService, scores: dict[str, float] | None
) -> tuple[list[ContextChunk], FakeQdrant, list[list[str]]]:
    """Runs retrieve() with fake embedding, Qdrant and reranker. `scores`
    maps chunk text to rerank score; None makes the reranker fail."""
    qdrant = FakeQdrant()
    reranked: list[list[str]] = []

    async def fake_embed(self: RagService, query: str) -> list[float]:
        return [0.1, 0.2]

    async def fake_rerank(self: Reranker, query: str, documents: list[str]) -> list[float] | None:
        reranked.append(documents)
        return None if scores is None else [scores[document] for document in documents]

    monkeypatch.setattr(RagService, "_embed_query", fake_embed)
    monkeypatch.setattr(RagService, "_make_qdrant_client", lambda self: qdrant)
    monkeypatch.setattr(Reranker, "rerank", fake_rerank)
    chunks = asyncio.run(service.retrieve("laser safety"))
    return chunks, qdrant, reranked


ALL_SCORES = {f"{c}-{n}": 0.0 for c in ("wiki", "events") for n in range(4)} | {
    "wiki-3": 0.95,
    "events-2": 0.9,
    "wiki-0": 0.2,
}


def test_reranking_reorders_across_collections(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _service(Reranker(url="http://r.local", model="m"), rerank_candidates=4)
    chunks, qdrant, reranked = _retrieve(monkeypatch, service, ALL_SCORES)

    # Each collection is searched for rerank_candidates, not its top_k.
    assert qdrant.limits == {"wiki": 4, "events": 4}
    assert len(reranked[0]) == 8
    # Kept: the sum of top_k (2 + 1), by rerank score.
    assert [chunk.text for chunk in chunks] == ["wiki-3", "events-2", "wiki-0"]
    assert [chunk.rerank_score for chunk in chunks] == [0.95, 0.9, 0.2]
    assert chunks[0].score == pytest.approx(0.6)


def test_rerank_top_n_and_min_score(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _service(
        Reranker(url="http://r.local", model="m"),
        rerank_candidates=4,
        rerank_top_n=5,
        rerank_min_score=0.5,
    )
    chunks, _, _ = _retrieve(monkeypatch, service, ALL_SCORES)
    assert [chunk.text for chunk in chunks] == ["wiki-3", "events-2"]


def test_candidates_never_fewer_than_top_k(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _service(Reranker(url="http://r.local", model="m"), rerank_candidates=1)
    _, qdrant, _ = _retrieve(monkeypatch, service, ALL_SCORES)
    assert qdrant.limits == {"wiki": 2, "events": 1}


def test_failed_reranking_falls_back_to_vector_order(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _service(Reranker(url="http://r.local", model="m"), rerank_candidates=4)
    chunks, _, reranked = _retrieve(monkeypatch, service, None)
    assert reranked
    assert [chunk.text for chunk in chunks] == ["wiki-0", "events-0", "wiki-1"]
    assert all(chunk.rerank_score is None for chunk in chunks)


def test_no_reranker_keeps_vector_search(monkeypatch: pytest.MonkeyPatch) -> None:
    chunks, qdrant, reranked = _retrieve(monkeypatch, _service(None), ALL_SCORES)
    assert reranked == []
    assert qdrant.limits == {"wiki": 2, "events": 1}
    assert [chunk.text for chunk in chunks] == ["wiki-0", "events-0", "wiki-1"]


def test_build_reranker_from_settings() -> None:
    assert build_reranker(Settings()) is None
    reranker = build_reranker(
        Settings(rerank_url="http://tei.local", rerank_api="tei", rerank_api_key="k")
    )
    assert reranker == Reranker(
        url="http://tei.local",
        model="BAAI/bge-reranker-v2-m3",
        api="tei",
        api_key="k",
        timeout_seconds=30.0,
        context_length=8192,
    )
