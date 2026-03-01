from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.rag import RagService


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
