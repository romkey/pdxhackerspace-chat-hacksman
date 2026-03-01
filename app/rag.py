from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
from qdrant_client import QdrantClient

from app.models import ContextChunk


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
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self.embedding_url.rstrip('/')}/api/embeddings",
                    json={"model": self.embedding_model, "prompt": query},
                )
                response.raise_for_status()
                data = response.json()
            embedding = data.get("embedding")
            if isinstance(embedding, list) and embedding:
                return [float(x) for x in embedding]
        except Exception:
            return None
        return None

    async def retrieve(self, query: str) -> list[ContextChunk]:
        if not self.collections:
            return []
        vector = await self._embed_query(query)
        if vector is None:
            return []

        client = self._make_qdrant_client()
        out: list[ContextChunk] = []
        for collection in self.collections:
            try:
                hits = client.search(
                    collection_name=collection,
                    query_vector=vector,
                    with_payload=True,
                    limit=self.top_k,
                )
            except Exception:
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
        return out[: self.top_k]
