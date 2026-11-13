import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

# Any: Qdrant payloads and collection info are untyped JSON / loosely typed objects.
from typing import Any

import httpx
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import Condition, DatetimeRange, FieldCondition, Filter, MatchAny

from chat_hacksman.models import ContextChunk
from chat_hacksman.rerank import Reranker

logger = logging.getLogger(__name__)


def _is_slack_collection(collection: str) -> bool:
    normalized = collection.strip().lower()
    return (
        normalized == "slack" or normalized.startswith("slack_") or normalized.startswith("slack-")
    )


def _is_events_collection(collection: str) -> bool:
    normalized = collection.strip().lower()
    return (
        normalized == "events"
        or normalized.startswith("events_")
        or normalized.startswith("events-")
    )


def _is_slack_payload(payload: dict[str, Any]) -> bool:
    slack_keys = {
        "channel_name",
        "channel_id",
        "user_name",
        "user_id",
        "permalink",
        "thread_ts",
        "ts",
    }
    return any(key in payload for key in slack_keys)


def _is_calibre_collection(collection: str) -> bool:
    normalized = collection.strip().lower()
    return (
        normalized == "calibre_books"
        or normalized.startswith("calibre_")
        or normalized.startswith("calibre-")
    )


def _extract_slack_chunk_text(payload: dict[str, Any]) -> str:
    doc_type = payload.get("doc_type")
    normalized_doc_type = doc_type.strip().lower() if isinstance(doc_type, str) else ""
    preferred_keys: tuple[str, ...]
    if normalized_doc_type == "thread_summary":
        preferred_keys = ("thread_summary", "summary", "thread_text", "text")
    elif normalized_doc_type == "channel_summary":
        preferred_keys = ("channel_summary", "summary", "channel_text", "text")
    elif normalized_doc_type in {"user_summary", "team_summary", "workspace_summary"}:
        preferred_keys = ("summary", "profile", "description", "text")
    elif normalized_doc_type == "message":
        preferred_keys = ("text", "message_text", "body", "content")
    else:
        preferred_keys = ("summary", "text", "content", "body")

    for key in preferred_keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    message_obj = payload.get("message")
    if isinstance(message_obj, dict):
        message_text = message_obj.get("text")
        if isinstance(message_text, str) and message_text.strip():
            return message_text.strip()

    return ""


def _extract_chunk_text(payload: dict[str, Any]) -> str:
    if _is_slack_payload(payload):
        slack_text = _extract_slack_chunk_text(payload)
        if slack_text:
            return slack_text
    for key in ("chunk_text", "text", "content", "chunk", "document", "body"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _bearer_auth_headers(api_key: str | None) -> dict[str, str] | None:
    if not api_key:
        return None
    return {"Authorization": f"Bearer {api_key}"}


@dataclass(slots=True)
class RagService:
    qdrant_url: str
    qdrant_api_key: str | None
    collections: list[str]
    embedding_url: str
    embedding_model: str
    top_k: int
    top_k_events: int | None = None
    top_k_slack: int | None = None
    top_k_calibre: int | None = None
    embedding_context_length: int = 8192
    embedding_timeout_seconds: float = 30.0
    min_score: float = 0.0
    embedding_api_key: str | None = None
    embedding_query_prefix: str = ""
    # With a reranker, each collection is searched for rerank_candidates
    # chunks (at least its top_k), all of them are reranked together, and
    # the best rerank_top_n (default: the sum of the collections' top_k) at
    # or above rerank_min_score are kept.
    reranker: Reranker | None = None
    rerank_candidates: int = 10
    rerank_top_n: int | None = None
    rerank_min_score: float | None = None
    _collection_vector_dims: dict[str, int | None] = field(default_factory=dict)
    _client: QdrantClient | None = None

    def _make_qdrant_client(self) -> QdrantClient:
        return QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key, timeout=5)

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
            truncated = " ".join(tokens[:limit])
            logger.warning(
                "Embedding input exceeded configured context length model=%s "
                "estimated_tokens=%d limit=%d; truncating",
                self.embedding_model,
                len(tokens),
                limit,
            )
            return truncated

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

    def _search_collection(
        self,
        *,
        client: QdrantClient,
        collection: str,
        vector: list[float],
        query_filter: Filter | None = None,
    ) -> list[Any]:
        target_top_k = self._candidate_limit(collection)
        # Fetch extra when RAG_MIN_SCORE will drop some; retrieve() caps
        # what's left at the candidate limit.
        search_limit = target_top_k * 2 if self.min_score > 0 else target_top_k
        if hasattr(client, "search"):
            try:
                hits = client.search(
                    collection_name=collection,
                    query_vector=vector,
                    query_filter=query_filter,
                    with_payload=True,
                    limit=search_limit,
                )
            except TypeError:
                try:
                    hits = client.search(
                        collection_name=collection,
                        query_vector=vector,
                        filter=query_filter,
                        with_payload=True,
                        limit=search_limit,
                    )
                except TypeError:
                    hits = client.search(
                        collection_name=collection,
                        query_vector=vector,
                        with_payload=True,
                        limit=search_limit,
                    )
            return list(hits)

        if hasattr(client, "query_points"):
            try:
                response = client.query_points(
                    collection_name=collection,
                    query=vector,
                    query_filter=query_filter,
                    with_payload=True,
                    limit=search_limit,
                )
            except TypeError:
                try:
                    response = client.query_points(
                        collection_name=collection,
                        query=vector,
                        filter=query_filter,
                        with_payload=True,
                        limit=search_limit,
                    )
                except TypeError:
                    response = client.query_points(
                        collection_name=collection,
                        query=vector,
                        with_payload=True,
                        limit=search_limit,
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

    def _slack_doc_types_for_query(self, query: str) -> list[str] | None:
        tokens = set(query.strip().lower().split())
        channel_words = {"channel", "channels", "join", "where", "post", "discuss", "room"}
        people_words = {"who", "person", "member", "members", "team", "contact", "expert"}
        thread_words = {"decided", "decision", "discussed", "outcome", "resolved", "thread"}

        if tokens & channel_words:
            return ["channel_summary", "workspace_summary"]
        if tokens & people_words:
            return ["user_summary", "team_summary", "message"]
        if tokens & thread_words:
            return ["thread_summary", "message"]
        return None

    def _events_filter_for_query(self, query: str) -> Filter | None:
        tokens = set(query.strip().lower().split())
        past_words = {"past", "previous", "last", "history", "happened", "was", "ran"}
        future_words = {"upcoming", "next", "future", "soon", "scheduled", "when", "will"}
        now_words = {"now", "today", "current", "happening", "ongoing"}
        summary_words = {"often", "frequency", "recurring", "regular", "schedule", "series"}
        occurrence_words = {"happened", "specific", "that", "tuesday", "week"}
        now = datetime.now(tz=UTC)

        must: list[Condition] = []
        if tokens & future_words:
            must.append(
                FieldCondition(
                    key="start_time",
                    range=DatetimeRange(gte=now),
                )
            )
        elif tokens & now_words:
            must.append(
                FieldCondition(
                    key="temporal_status",
                    match=MatchAny(any=["current", "future"]),
                )
            )
        elif tokens & past_words:
            must.append(
                FieldCondition(
                    key="start_time",
                    range=DatetimeRange(lt=now),
                )
            )

        if tokens & summary_words:
            must.append(
                FieldCondition(
                    key="record_type",
                    match=MatchAny(any=["event_summary"]),
                )
            )
        elif tokens & occurrence_words:
            must.append(
                FieldCondition(
                    key="record_type",
                    match=MatchAny(any=["occurrence"]),
                )
            )

        if not must:
            return None
        return Filter(must=must)

    def _candidate_limit(self, collection: str) -> int:
        """Chunks to take from a collection: its top_k, or more for the
        reranker to choose from."""
        top_k = self._top_k_for_collection(collection)
        return max(top_k, self.rerank_candidates) if self.reranker is not None else top_k

    def _top_k_for_collection(self, collection: str) -> int:
        if _is_calibre_collection(collection) and self.top_k_calibre is not None:
            return max(1, self.top_k_calibre)
        if _is_slack_collection(collection) and self.top_k_slack is not None:
            return max(1, self.top_k_slack)
        if _is_events_collection(collection) and self.top_k_events is not None:
            return max(1, self.top_k_events)
        return max(1, self.top_k)

    def _build_collection_filter(self, collection: str, query: str) -> Filter | None:
        if _is_calibre_collection(collection):
            chunk_types = self._calibre_chunk_types_for_query(query)
            return Filter(
                must=[
                    FieldCondition(
                        key="chunk_type",
                        match=MatchAny(any=chunk_types),
                    )
                ]
            )
        if _is_slack_collection(collection):
            doc_types = self._slack_doc_types_for_query(query)
            if doc_types:
                return Filter(
                    must=[
                        FieldCondition(
                            key="doc_type",
                            match=MatchAny(any=doc_types),
                        )
                    ]
                )
        if _is_events_collection(collection):
            return self._events_filter_for_query(query)
        return None

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
                logger.warning(
                    "Qdrant search failed collection=%s error_type=%s error=%s",
                    collection,
                    type(exc).__name__,
                    exc,
                )
                continue
            except Exception as exc:
                logger.warning(
                    "Qdrant search failed collection=%s error_type=%s error=%s",
                    collection,
                    type(exc).__name__,
                    exc,
                )
                continue

            # Without reranking, extra hits are trimmed by the overall limit
            # below; with it, every hit kept here is sent to the reranker.
            kept = 0
            for hit in hits:
                if self.reranker is not None and kept >= self._candidate_limit(collection):
                    break
                payload = hit.payload if isinstance(hit.payload, dict) else {}
                text = _extract_chunk_text(payload)
                if not text:
                    continue
                score = float(hit.score)
                if score < self.min_score:
                    continue
                kept += 1
                out.append(
                    ContextChunk(
                        collection=collection,
                        score=score,
                        text=text,
                        metadata=payload,
                    )
                )

        final_limit = sum(
            self._top_k_for_collection(collection) for collection in target_collections
        )
        if final_limit <= 0:
            final_limit = max(1, self.top_k)
        out.sort(key=lambda item: item.score, reverse=True)
        if self.reranker is not None and out:
            reranked = await self._rerank(self.reranker, query, out, final_limit)
            if reranked is not None:
                logger.info("RAG retrieval produced %d reranked chunks", len(reranked))
                return reranked
            logger.warning("Reranking failed; using vector search order")
        logger.info("RAG retrieval produced %d usable chunks", len(out[:final_limit]))
        return out[:final_limit]

    async def _rerank(
        self, reranker: Reranker, query: str, chunks: list[ContextChunk], final_limit: int
    ) -> list[ContextChunk] | None:
        """The chunks most relevant to the query by rerank score, or None if
        the reranker failed."""
        scores = await reranker.rerank(query, [chunk.text for chunk in chunks])
        if scores is None:
            return None
        scored = [
            chunk.model_copy(update={"rerank_score": score})
            for chunk, score in zip(chunks, scores, strict=True)
            if self.rerank_min_score is None or score >= self.rerank_min_score
        ]
        # Stable sort: equal rerank scores keep their vector-score order.
        scored.sort(key=lambda chunk: chunk.rerank_score or 0.0, reverse=True)
        return scored[: self.rerank_top_n or final_limit]
