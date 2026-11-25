"""Retrieval over the shared hs_* Qdrant collections.

Each point follows the payload in the shared architecture plan: `text`,
`title`, `heading_path`, `url`, `source`, `tier`, `authority`,
`stale_after`, `superseded_by`, `page`, `updated_at` and the entity lists.
A question is embedded (dense) and, when a sparse vector is configured,
also matched by keyword (sparse, BM25 inferred by Qdrant itself); the two
result lists are fused with reciprocal rank fusion. The candidates from
every collection are then reranked together, and an authority pass prefers
policy, signs and the wiki over manuals and Slack when relevance is close.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

# Any: Qdrant payloads and collection info are untyped JSON / loosely typed objects.
from typing import Any

import httpx
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    DatetimeRange,
    Document,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    IsEmptyCondition,
    MatchAny,
    PayloadField,
    Prefetch,
)

from chat_hacksman.models import ContextChunk
from chat_hacksman.rerank import Reranker
from chat_hacksman.timing import EMBED, QDRANT, RERANK, SELECT, StageTimer

logger = logging.getLogger(__name__)

# A collection Qdrant says doesn't exist (e.g. hs_catalog before its
# ingester runs) is skipped for this long before it's looked for again.
MISSING_COLLECTION_RECHECK_SECONDS = 300.0


def _monotonic() -> float:
    return time.monotonic()


def _is_not_found(exc: Exception) -> bool:
    return getattr(exc, "status_code", None) == 404


# Authority of space-written rules (policy documents and signs).
POLICY_AUTHORITY = 5
# A policy or sign chunk this relevant (relevance scaled 0-1 over the
# candidates) is always given to the model, even if it missed the top N.
POLICY_MIN_RELEVANCE = 0.5


def _bearer_auth_headers(api_key: str | None) -> dict[str, str] | None:
    if not api_key:
        return None
    return {"Authorization": f"Bearer {api_key}"}


def authority_of(chunk: ContextChunk) -> int:
    """The chunk's payload authority, 1-5; 1 if missing or unusable."""
    value = chunk.metadata.get("authority")
    if isinstance(value, int) and not isinstance(value, bool):
        return min(max(value, 1), POLICY_AUTHORITY)
    return 1


def relevance_of(chunk: ContextChunk) -> float:
    """The reranker's score when reranking ran, else the search score."""
    return chunk.rerank_score if chunk.rerank_score is not None else chunk.score


def select_chunks(
    chunks: list[ContextChunk], top_n: int, authority_weight: float
) -> list[ContextChunk]:
    """The top_n chunks to give the model.

    Relevance is scaled to 0-1 over the candidates, then each chunk gets
    up to `authority_weight` more for its authority, so a policy chunk wins
    over a slightly more relevant Slack fact but not over a much more
    relevant one. The most relevant policy or sign chunk is always kept if
    its scaled relevance is at least POLICY_MIN_RELEVANCE.
    """
    if not chunks:
        return []
    scores = [relevance_of(chunk) for chunk in chunks]
    low, high = min(scores), max(scores)
    spread = high - low

    def scaled(score: float) -> float:
        return (score - low) / spread if spread > 0 else 1.0

    ranked = sorted(
        range(len(chunks)),
        key=lambda i: (
            scaled(scores[i]) + authority_weight * authority_of(chunks[i]) / POLICY_AUTHORITY
        ),
        reverse=True,
    )
    chosen = ranked[:top_n]
    policies = [
        i
        for i in sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
        if authority_of(chunks[i]) >= POLICY_AUTHORITY and scaled(scores[i]) >= POLICY_MIN_RELEVANCE
    ]
    if policies and policies[0] not in chosen:
        chosen = [*chosen[: top_n - 1], policies[0]]
    return [chunks[i] for i in chosen]


def _chunk_from_payload(collection: str, score: float, payload: dict[str, Any]) -> ContextChunk:
    text = payload.get("text")
    return ContextChunk(
        collection=collection,
        score=score,
        text=text.strip() if isinstance(text, str) else "",
        metadata=payload,
    )


def _rerank_text(chunk: ContextChunk) -> str:
    """What the reranker reads: where the chunk sits, then the chunk."""
    parts = [
        value.strip()
        for value in (chunk.metadata.get("title"), chunk.metadata.get("heading_path"))
        if isinstance(value, str) and value.strip()
    ]
    return f"{' > '.join(parts)}\n{chunk.text}" if parts else chunk.text


@dataclass(slots=True)
class RagService:
    qdrant_url: str
    qdrant_api_key: str | None
    collections: list[str]
    embedding_url: str
    embedding_model: str
    candidates: int = 50
    top_n: int = 8
    embedding_context_length: int = 8192
    embedding_timeout_seconds: float = 30.0
    # Applied only to dense-only search.
    min_score: float = 0.0
    embedding_api_key: str | None = None
    embedding_query_prefix: str = ""
    dense_vector: str = "dense"
    # Empty: dense-only search.
    sparse_vector: str = "sparse"
    sparse_model: str = "qdrant/bm25"
    allowed_tiers: tuple[str, ...] = ("public", "member")
    authority_weight: float = 0.1
    reranker: Reranker | None = None
    rerank_min_score: float | None = None
    _collection_vector_dims: dict[str, int | None] = field(default_factory=dict)
    # Collection -> _monotonic() until which it's taken to be missing.
    _missing_until: dict[str, float] = field(default_factory=dict)
    _client: QdrantClient | None = None

    @property
    def hybrid(self) -> bool:
        return bool(self.sparse_vector)

    def _make_qdrant_client(self) -> QdrantClient:
        # cloud_inference sends the sparse query's text to Qdrant to embed
        # (BM25 runs in the server) instead of embedding it here.
        return QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            timeout=5,
            cloud_inference=self.hybrid,
        )

    def _get_qdrant_client(self) -> QdrantClient:
        if self._client is None:
            self._client = self._make_qdrant_client()
        return self._client

    async def _request_embedding(self, text: str) -> list[float] | None:
        """One embedding from an OpenAI-compatible POST /embeddings endpoint.

        API: https://platform.openai.com/docs/api-reference/embeddings/create
        """
        async with httpx.AsyncClient(timeout=self.embedding_timeout_seconds) as client:
            response = await client.post(
                f"{self.embedding_url.rstrip('/')}/embeddings",
                json={
                    "model": self.embedding_model,
                    "input": [text],
                    "encoding_format": "float",
                },
                headers=_bearer_auth_headers(self.embedding_api_key),
            )
            response.raise_for_status()
            data = response.json()
        items = data.get("data") if isinstance(data, dict) else None
        if not isinstance(items, list) or not items or not isinstance(items[0], dict):
            return None
        embedding = items[0].get("embedding")
        if not isinstance(embedding, list) or not embedding:
            return None
        logger.info("Embedding generated with %d dimensions", len(embedding))
        return [float(x) for x in embedding]

    def _truncate_query_for_embedding(self, query: str) -> str:
        # Leave room for the query prefix, which must never be cut off.
        prefix_tokens = len(self.embedding_query_prefix.split())
        limit = max(1, int(self.embedding_context_length) - prefix_tokens)
        max_chars = max(128, limit * 4)
        normalized = query.strip()
        if not normalized:
            return query

        tokens = normalized.split()
        if len(tokens) > limit:
            logger.warning(
                "Embedding input exceeded configured context length model=%s "
                "estimated_tokens=%d limit=%d; truncating",
                self.embedding_model,
                len(tokens),
                limit,
            )
            return " ".join(tokens[:limit])

        if len(normalized) > max_chars:
            logger.warning(
                "Embedding input exceeded configured char budget model=%s chars=%d "
                "char_limit=%d (context_length=%d); truncating",
                self.embedding_model,
                len(normalized),
                max_chars,
                limit,
            )
            return normalized[:max_chars]

        return normalized

    async def _embed_query(self, query: str) -> list[float] | None:
        embedding_input = self.embedding_query_prefix + self._truncate_query_for_embedding(query)
        logger.info(
            "Creating embedding for query using model=%s at %s",
            self.embedding_model,
            self.embedding_url,
        )
        try:
            embedding = await self._request_embedding(embedding_input)
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning(
                "Embedding request failed url=%s/embeddings model=%s error_type=%s error=%s",
                self.embedding_url.rstrip("/"),
                self.embedding_model,
                type(exc).__name__,
                exc,
            )
            return None
        if embedding is None:
            logger.warning("Embedding response had no usable vector payload")
        return embedding

    def base_filter(self, now: datetime | None = None) -> Filter:
        """Allowed tiers only, nothing superseded, nothing past stale_after."""
        moment = now or datetime.now(tz=UTC)
        return Filter(
            must=[
                FieldCondition(key="tier", match=MatchAny(any=list(self.allowed_tiers))),
                IsEmptyCondition(is_empty=PayloadField(key="superseded_by")),
                Filter(
                    should=[
                        IsEmptyCondition(is_empty=PayloadField(key="stale_after")),
                        FieldCondition(key="stale_after", range=DatetimeRange(gt=moment)),
                    ]
                ),
            ]
        )

    def _dense_search(
        self, client: QdrantClient, collection: str, vector: list[float], query_filter: Filter
    ) -> list[Any]:
        response = client.query_points(
            collection_name=collection,
            query=vector,
            using=self.dense_vector or None,
            query_filter=query_filter,
            with_payload=True,
            limit=self.candidates,
        )
        return [hit for hit in response.points if float(hit.score) >= self.min_score]

    def _hybrid_search(
        self,
        client: QdrantClient,
        collection: str,
        vector: list[float],
        query: str,
        query_filter: Filter,
    ) -> list[Any]:
        response = client.query_points(
            collection_name=collection,
            prefetch=[
                Prefetch(
                    query=vector,
                    using=self.dense_vector or None,
                    filter=query_filter,
                    limit=self.candidates,
                ),
                Prefetch(
                    query=Document(text=query, model=self.sparse_model),
                    using=self.sparse_vector,
                    filter=query_filter,
                    limit=self.candidates,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            with_payload=True,
            limit=self.candidates,
        )
        return list(response.points)

    def _search_collection(
        self,
        client: QdrantClient,
        collection: str,
        vector: list[float],
        query: str,
        query_filter: Filter,
    ) -> list[Any]:
        if self.hybrid:
            try:
                return self._hybrid_search(client, collection, vector, query, query_filter)
            except Exception as exc:
                if _is_not_found(exc):
                    raise
                logger.warning(
                    "Hybrid search failed collection=%s error_type=%s error=%s; "
                    "falling back to dense search",
                    collection,
                    type(exc).__name__,
                    exc,
                )
        return self._dense_search(client, collection, vector, query_filter)

    def _extract_vector_dim(self, collection_info: Any) -> int | None:
        config = getattr(collection_info, "config", None)
        params = getattr(config, "params", None)
        vectors = getattr(params, "vectors", None)
        if vectors is None:
            return None
        if isinstance(vectors, dict):
            named = vectors.get(self.dense_vector) if self.dense_vector else None
            size = getattr(named, "size", None)
        else:
            size = None if self.dense_vector else getattr(vectors, "size", None)
        return size if isinstance(size, int) else None

    def _is_missing(self, collection: str) -> bool:
        until = self._missing_until.get(collection)
        if until is None:
            return False
        if _monotonic() < until:
            return True
        del self._missing_until[collection]
        return False

    def _mark_missing(self, collection: str) -> None:
        logger.info(
            "Collection %s doesn't exist in Qdrant; skipping it for %d s",
            collection,
            MISSING_COLLECTION_RECHECK_SECONDS,
        )
        self._missing_until[collection] = _monotonic() + MISSING_COLLECTION_RECHECK_SECONDS

    def _get_collection_vector_dim(self, client: QdrantClient, collection: str) -> int | None:
        if collection in self._collection_vector_dims:
            return self._collection_vector_dims[collection]
        try:
            vector_dim = self._extract_vector_dim(client.get_collection(collection_name=collection))
        except Exception as exc:
            if _is_not_found(exc):
                self._mark_missing(collection)
                return None
            logger.warning("Failed to read vector dimension for collection=%s: %s", collection, exc)
            vector_dim = None
        if vector_dim is None:
            logger.warning(
                "Could not find dense vector %r in collection=%s; will try the query anyway",
                self.dense_vector or "(unnamed)",
                collection,
            )
        self._collection_vector_dims[collection] = vector_dim
        return vector_dim

    async def retrieve(
        self,
        query: str,
        enabled_collections: list[str] | None = None,
        timer: StageTimer | None = None,
    ) -> list[ContextChunk]:
        """The chunks to answer `query` from: best first, at most top_n.

        `timer` gets the embed, qdrant (and qdrant:<collection>), rerank and
        select stages.
        """
        timer = timer if timer is not None else StageTimer()
        target = enabled_collections if enabled_collections is not None else self.collections
        available = set(self.collections)
        target_collections = [collection for collection in target if collection in available]
        if not target_collections:
            logger.info("Skipping retrieval because no collections are enabled")
            return []
        logger.info("Running retrieval across collections=%s", ",".join(target_collections))
        with timer.stage(EMBED):
            vector = await self._embed_query(query)
        if vector is None:
            logger.warning("Skipping retrieval because query embedding failed")
            return []

        client = self._get_qdrant_client()
        query_filter = self.base_filter()
        candidates: list[ContextChunk] = []
        for collection in target_collections:
            if self._is_missing(collection):
                continue
            with timer.stage(QDRANT), timer.stage(f"{QDRANT}:{collection}"):
                hits = self._query_collection(client, collection, vector, query, query_filter)
            if hits is None:
                continue
            logger.info("Qdrant search for %s found %d hits", collection, len(hits))
            for hit in hits:
                payload = hit.payload if isinstance(hit.payload, dict) else {}
                chunk = _chunk_from_payload(collection, float(hit.score), payload)
                if chunk.text:
                    candidates.append(chunk)

        candidates.sort(key=lambda chunk: chunk.score, reverse=True)
        if self.reranker is not None and candidates:
            with timer.stage(RERANK):
                reranked = await self._rerank(self.reranker, query, candidates)
            if reranked is not None:
                candidates = reranked
            else:
                logger.warning("Reranking failed; using search order")
        with timer.stage(SELECT):
            chosen = select_chunks(candidates, self.top_n, self.authority_weight)
        logger.info("Retrieval chose %d of %d candidate chunks", len(chosen), len(candidates))
        return chosen

    def _query_collection(
        self,
        client: QdrantClient,
        collection: str,
        vector: list[float],
        query: str,
        query_filter: Filter,
    ) -> list[Any] | None:
        """The collection's hits; None if it's missing, mismatched or failed."""
        expected_dim = self._get_collection_vector_dim(client, collection)
        if self._is_missing(collection):
            return None
        if expected_dim is not None and expected_dim != len(vector):
            logger.warning(
                "Skipping collection=%s due to vector dimension mismatch "
                "(expected=%d, got=%d). "
                "Use a matching embedding model or rebuild that collection.",
                collection,
                expected_dim,
                len(vector),
            )
            return None
        try:
            return self._search_collection(client, collection, vector, query, query_filter)
        except Exception as exc:
            if _is_not_found(exc):
                self._mark_missing(collection)
                return None
            logger.warning(
                "Qdrant search failed collection=%s error_type=%s error=%s",
                collection,
                type(exc).__name__,
                exc,
            )
            return None

    async def _rerank(
        self, reranker: Reranker, query: str, chunks: list[ContextChunk]
    ) -> list[ContextChunk] | None:
        """The chunks with rerank scores, best first, without those under
        rerank_min_score; None if the reranker failed."""
        scores = await reranker.rerank(query, [_rerank_text(chunk) for chunk in chunks])
        if scores is None:
            return None
        scored = [
            chunk.model_copy(update={"rerank_score": score})
            for chunk, score in zip(chunks, scores, strict=True)
            if self.rerank_min_score is None or score >= self.rerank_min_score
        ]
        # Stable sort: equal rerank scores keep their search order.
        scored.sort(key=lambda chunk: chunk.rerank_score or 0.0, reverse=True)
        return scored
