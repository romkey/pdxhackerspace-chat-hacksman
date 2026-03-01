from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx
from qdrant_client import QdrantClient

from app.models import ContextChunk

logger = logging.getLogger(__name__)


def _extract_chunk_text(payload: dict[str, Any]) -> str:
    for key in ("text", "content", "chunk", "document", "body"):
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

    def _make_qdrant_client(self) -> QdrantClient:
        return QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key, timeout=5.0)

    async def _embed_query(self, query: str) -> list[float] | None:
        try:
            logger.info(
                "Creating embedding for query using model=%s at %s",
                self.embedding_model,
                self.embedding_url,
            )
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self.embedding_url.rstrip('/')}/api/embeddings",
                    json={"model": self.embedding_model, "prompt": query},
                )
                response.raise_for_status()
                data = response.json()
            embedding = data.get("embedding")
            if isinstance(embedding, list) and embedding:
                logger.info("Embedding generated successfully with %d dimensions", len(embedding))
                return [float(x) for x in embedding]
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
    ) -> list[Any]:
        if hasattr(client, "search"):
            hits = client.search(
                collection_name=collection,
                query_vector=vector,
                with_payload=True,
                limit=self.top_k,
            )
            return list(hits)

        if hasattr(client, "query_points"):
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

        client = self._make_qdrant_client()
        out: list[ContextChunk] = []
        for collection in target_collections:
            try:
                hits = self._search_collection(
                    client=client,
                    collection=collection,
                    vector=vector,
                )
                logger.info(
                    "Qdrant search succeeded for %s with %d hits",
                    collection,
                    len(hits),
                )
            except Exception as exc:
                logger.exception("Qdrant search failed for %s: %s", collection, exc)
                continue

            for hit in hits:
                payload = hit.payload if isinstance(hit.payload, dict) else {}
                text = _extract_chunk_text(payload)
                if not text:
                    continue
                out.append(
                    ContextChunk(
                        collection=collection,
                        score=float(hit.score),
                        text=text,
                        metadata=payload,
                    )
                )

        out.sort(key=lambda item: item.score, reverse=True)
        logger.info("RAG retrieval produced %d usable chunks", len(out[: self.top_k]))
        return out[: self.top_k]
