"""Reranking: score retrieved chunks against the question with a cross-encoder.

Vector search finds chunks whose embeddings are near the question's; a
reranker reads the question and each chunk together and scores how well
the chunk answers it, which is slower but much more precise. Two request
formats are supported, chosen by RERANK_API:

- "jina": POST {url}/rerank with {"model", "query", "documents", "top_n"},
  answered with {"results": [{"index", "relevance_score"}, ...]}. Jina's
  API (https://jina.ai/reranker/), Cohere's v1/v2 rerank API
  (https://docs.cohere.com/v2/reference/rerank), vLLM, llama.cpp's
  llama-server and Infinity all speak it.
- "tei": POST {url}/rerank with {"query", "texts", "truncate"}, answered with
  [{"index", "score"}, ...]: Hugging Face text-embeddings-inference
  (https://huggingface.github.io/text-embeddings-inference/), also accepted
  by llama-server.
"""

import logging
from dataclasses import dataclass
from typing import Literal

import httpx

logger = logging.getLogger(__name__)

RerankApi = Literal["jina", "tei"]
RERANK_APIS: tuple[RerankApi, ...] = ("jina", "tei")


def truncate_words(text: str, limit: int) -> str:
    """At most `limit` whitespace-separated words, and about 4 characters per
    word, a cheap stand-in for the model's token limit."""
    limit = max(1, limit)
    words = text.split()
    if len(words) > limit:
        return " ".join(words[:limit])
    max_chars = max(128, limit * 4)
    return text if len(text) <= max_chars else text[:max_chars]


def _parse_scores(data: object, api: RerankApi, count: int) -> list[float] | None:
    """One score per document, in the order they were sent; None if the
    response doesn't score every document."""
    if api == "jina":
        results = data.get("results") if isinstance(data, dict) else None
        score_key = "relevance_score"
    else:
        results = data
        score_key = "score"
    if not isinstance(results, list):
        return None
    scores: dict[int, float] = {}
    for item in results:
        if not isinstance(item, dict):
            return None
        index, score = item.get("index"), item.get(score_key)
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not 0 <= index < count
            or not isinstance(score, int | float)
            or isinstance(score, bool)
        ):
            return None
        scores[index] = float(score)
    if len(scores) != count:
        return None
    return [scores[index] for index in range(count)]


@dataclass(slots=True)
class Reranker:
    url: str
    model: str
    api: RerankApi = "jina"
    api_key: str | None = None
    timeout_seconds: float = 30.0
    context_length: int = 8192

    def _payload(self, query: str, documents: list[str]) -> dict[str, object]:
        if self.api == "tei":
            # TEI has no model field (it serves one model); truncate makes it
            # cut over-long pairs instead of rejecting the request.
            return {"query": query, "texts": documents, "truncate": True}
        return {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": len(documents),
        }

    async def rerank(self, query: str, documents: list[str]) -> list[float] | None:
        """A relevance score for each document, in order; higher is more
        relevant. None if the reranker fails, so callers can fall back to
        vector order."""
        if not documents:
            return []
        query_text = truncate_words(query.strip(), self.context_length // 4)
        # Each query/document pair must fit the model's context.
        document_limit = max(1, self.context_length - len(query_text.split()))
        texts = [truncate_words(document, document_limit) for document in documents]
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        url = f"{self.url.rstrip('/')}/rerank"
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    url, json=self._payload(query_text, texts), headers=headers
                )
                response.raise_for_status()
                data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning(
                "Rerank request failed url=%s model=%s error_type=%s error=%s",
                url,
                self.model,
                type(exc).__name__,
                exc,
            )
            return None
        scores = _parse_scores(data, self.api, len(texts))
        if scores is None:
            logger.warning(
                "Rerank response from %s didn't score all %d documents; check RERANK_API=%s",
                url,
                len(texts),
                self.api,
            )
            return None
        logger.info("Reranked %d chunks with model=%s", len(texts), self.model)
        return scores
