import asyncio
import json
from collections.abc import Callable

import httpx
import pytest

from chat_hacksman.config import Settings
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
