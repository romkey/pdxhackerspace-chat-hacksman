from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import httpx
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import FieldCondition, Filter, MatchAny

from app.models import ContextChunk

logger = logging.getLogger(__name__)


def _extract_chunk_text(payload: dict[str, Any]) -> str:
    for key in ("chunk_text", "text", "content", "chunk", "document", "body"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


@dataclass(slots=True)
class RagService:
    qdrant_url: str
    qdrant_api_key: str | None
    collections: list[str]
    embedding_url: str
    embedding_model: str
    top_k: int
    embedding_timeout_seconds: float = 30.0
    min_score: float = 0.0
    _collection_vector_dims: dict[str, int | None] = field(default_factory=dict)
    _client: QdrantClient | None = None

    def _make_qdrant_client(self) -> QdrantClient:
        return QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key, timeout=5.0)

    def _get_qdrant_client(self) -> QdrantClient:
        if self._client is None:
            self._client = self._make_qdrant_client()
        return self._client

    async def _embed_query_modern(self, query: str) -> list[float] | None:
        async with httpx.AsyncClient(timeout=self.embedding_timeout_seconds) as client:
            response = await client.post(
                f"{self.embedding_url.rstrip('/')}/api/embed",
                json={"model": self.embedding_model, "input": [query]},
            )
            response.raise_for_status()
            data = response.json()
        embeddings = data.get("embeddings")
        if isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], list):
            logger.info("Embedding generated via /api/embed with %d dimensions", len(embeddings[0]))
            return [float(x) for x in embeddings[0]]
        return None

    async def _embed_query_legacy(self, query: str) -> list[float] | None:
        async with httpx.AsyncClient(timeout=self.embedding_timeout_seconds) as client:
            response = await client.post(
                f"{self.embedding_url.rstrip('/')}/api/embeddings",
                json={"model": self.embedding_model, "prompt": query},
            )
            response.raise_for_status()
            data = response.json()
        embedding = data.get("embedding")
        if isinstance(embedding, list) and embedding:
            logger.info(
                "Embedding generated via /api/embeddings with %d dimensions", len(embedding)
            )
            return [float(x) for x in embedding]
        return None

    async def _embed_query(self, query: str) -> list[float] | None:
        try:
            logger.info(
                "Creating embedding for query using model=%s at %s",
                self.embedding_model,
                self.embedding_url,
            )
            modern_embedding: list[float] | None = None
            try:
                modern_embedding = await self._embed_query_modern(query)
            except Exception as exc:
                logger.warning("Modern embed endpoint /api/embed failed: %s", exc)
            if modern_embedding is not None:
                return modern_embedding
            logger.warning("Trying legacy embed endpoint /api/embeddings")
            legacy_embedding = await self._embed_query_legacy(query)
            if legacy_embedding is not None:
                return legacy_embedding
            logger.warning("Embedding response had no usable vector payload")
        except Exception as exc:
            logger.exception("Embedding request failed: %s", exc)
            return None
        return None

    def _search_collection(
        self,
        *,
        client: QdrantClient,
        collection: str,
        vector: list[float],
        query_filter: Filter | None = None,
    ) -> list[Any]:
        if hasattr(client, "search"):
            try:
                hits = client.search(
                    collection_name=collection,
                    query_vector=vector,
                    query_filter=query_filter,
                    with_payload=True,
                    limit=self.top_k,
                )
            except TypeError:
                try:
                    hits = client.search(
                        collection_name=collection,
                        query_vector=vector,
                        filter=query_filter,
                        with_payload=True,
                        limit=self.top_k,
                    )
                except TypeError:
                    hits = client.search(
                        collection_name=collection,
                        query_vector=vector,
                        with_payload=True,
                        limit=self.top_k,
                    )
            return list(hits)

        if hasattr(client, "query_points"):
            try:
                response = client.query_points(
                    collection_name=collection,
                    query=vector,
                    query_filter=query_filter,
                    with_payload=True,
                    limit=self.top_k,
                )
            except TypeError:
                try:
                    response = client.query_points(
                        collection_name=collection,
                        query=vector,
                        filter=query_filter,
                        with_payload=True,
                        limit=self.top_k,
                    )
                except TypeError:
                    response = client.query_points(
                        collection_name=collection,
                        query=vector,
                        with_payload=True,
                        limit=self.top_k,
                    )
            points = getattr(response, "points", response)
            return list(points) if points is not None else []

        raise AttributeError(
            "Qdrant client has neither 'search' nor 'query_points'; unsupported version."
        )

    def _calibre_chunk_types_for_query(self, query: str) -> list[str]:
        normalized = query.strip().lower()
        catalog_words = {"books", "book", "library", "catalog", "available", "have", "collection"}
        passage_words = {"quote", "passage", "chapter", "excerpt", "text", "say", "where"}
        tokens = set(normalized.split())

        if tokens & catalog_words:
            return ["book_metadata", "library_summary"]
        if tokens & passage_words:
            return ["content", "description", "library_summary"]
        return ["content", "description", "book_metadata", "library_summary"]

    def _build_collection_filter(self, collection: str, query: str) -> Filter | None:
        if collection != "calibre_books":
            return None
        chunk_types = self._calibre_chunk_types_for_query(query)
        return Filter(
            must=[
                FieldCondition(
                    key="chunk_type",
                    match=MatchAny(any=chunk_types),
                )
            ]
        )

    def _extract_vector_dim(self, collection_info: Any) -> int | None:
        config = getattr(collection_info, "config", None)
        params = getattr(config, "params", None)
        vectors = getattr(params, "vectors", None)
        if vectors is None:
            return None

        size = getattr(vectors, "size", None)
        if isinstance(size, int):
            return size

        # Named vectors may be represented as mapping-like objects.
        maybe_mapping = vectors
        root_mapping = getattr(vectors, "__root__", None)
        if root_mapping is not None:
            maybe_mapping = root_mapping

        values = maybe_mapping.values() if hasattr(maybe_mapping, "values") else None
        if values is None:
            return None

        sizes = {int(item.size) for item in values if isinstance(getattr(item, "size", None), int)}
        if len(sizes) == 1:
            return sizes.pop()
        return None

    def _get_collection_vector_dim(self, client: QdrantClient, collection: str) -> int | None:
        if collection in self._collection_vector_dims:
            return self._collection_vector_dims[collection]

        try:
            info = client.get_collection(collection_name=collection)
            vector_dim = self._extract_vector_dim(info)
            self._collection_vector_dims[collection] = vector_dim
            if vector_dim is None:
                logger.warning(
                    "Could not determine vector dimension for collection=%s; will try query anyway",
                    collection,
                )
            return vector_dim
        except Exception as exc:
            logger.warning(
                "Failed to read vector dimension for collection=%s: %s",
                collection,
                exc,
            )
            self._collection_vector_dims[collection] = None
            return None

    async def retrieve(
        self, query: str, enabled_collections: list[str] | None = None
    ) -> list[ContextChunk]:
        target_collections = (
            enabled_collections if enabled_collections is not None else self.collections
        )
        available_set = set(self.collections)
        target_collections = [
            collection for collection in target_collections if collection in available_set
        ]
        if not target_collections:
            logger.info("Skipping RAG retrieval because no collections are enabled")
            return []
        logger.info(
            "Running RAG retrieval across collections=%s",
            ",".join(target_collections),
        )
        vector = await self._embed_query(query)
        if vector is None:
            logger.warning("Skipping RAG retrieval because query embedding failed")
            return []
        query_dim = len(vector)

        client = self._get_qdrant_client()
        out: list[ContextChunk] = []
        for collection in target_collections:
            expected_dim = self._get_collection_vector_dim(client, collection)
            if expected_dim is not None and expected_dim != query_dim:
                logger.warning(
                    "Skipping collection=%s due to vector dimension mismatch "
                    "(expected=%d, got=%d). "
                    "Use a matching embedding model or rebuild that collection.",
                    collection,
                    expected_dim,
                    query_dim,
                )
                continue
            try:
                query_filter = self._build_collection_filter(collection, query)
                hits = self._search_collection(
                    client=client,
                    collection=collection,
                    vector=vector,
                    query_filter=query_filter,
                )
                logger.info(
                    "Qdrant search succeeded for %s with %d hits",
                    collection,
                    len(hits),
                )
            except UnexpectedResponse as exc:
                message = str(exc)
                if "Vector dimension error" in message:
                    logger.warning(
                        "Skipping collection=%s due to vector dimension error "
                        "reported by Qdrant: %s",
                        collection,
                        message,
                    )
                    continue
                logger.exception("Qdrant search failed for %s: %s", collection, exc)
            except Exception as exc:
                logger.exception("Qdrant search failed for %s: %s", collection, exc)
                continue

            for hit in hits:
                payload = hit.payload if isinstance(hit.payload, dict) else {}
                text = _extract_chunk_text(payload)
                if not text:
                    continue
                score = float(hit.score)
                if score < self.min_score:
                    continue
                out.append(
                    ContextChunk(
                        collection=collection,
                        score=score,
                        text=text,
                        metadata=payload,
                    )
                )

        out.sort(key=lambda item: item.score, reverse=True)
        logger.info("RAG retrieval produced %d usable chunks", len(out[: self.top_k]))
        return out[: self.top_k]
