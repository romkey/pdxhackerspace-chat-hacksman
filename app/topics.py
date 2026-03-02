from __future__ import annotations

import logging
from dataclasses import dataclass, field
from time import monotonic
from typing import Any

import httpx

logger = logging.getLogger(__name__)


def _to_topic_strings(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    out: list[str] = []
    for item in values:
        if isinstance(item, str):
            cleaned = item.strip()
            if cleaned:
                out.append(cleaned)
            continue
        if isinstance(item, dict):
            for key in ("title", "name", "topic", "interest", "training_topic"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    out.append(value.strip())
                    break
    return out


def _unique_sorted(values: list[str]) -> list[str]:
    deduped: dict[str, str] = {}
    for value in values:
        key = value.casefold().strip()
        if not key:
            continue
        if key not in deduped:
            deduped[key] = value.strip()
    return sorted(deduped.values(), key=lambda item: item.casefold())


def parse_topics_payload(payload: Any) -> dict[str, list[str]]:
    if not isinstance(payload, dict):
        return {"interests": [], "training_topics": [], "all_topics": []}

    interests = _to_topic_strings(payload.get("interests", []))
    training_topics = _to_topic_strings(
        payload.get("training_topics", payload.get("trainingTopics", []))
    )

    if not interests and not training_topics:
        # Flexible fallback for unknown payloads.
        fallback = _unique_sorted(_to_topic_strings(payload.get("topics", [])))
        if not fallback:
            # events.json fallback: derive topics from event titles
            fallback = _unique_sorted(_to_topic_strings(payload.get("events", [])))
        return {"interests": fallback, "training_topics": [], "all_topics": fallback}

    all_topics = _unique_sorted(interests + training_topics)
    return {
        "interests": interests,
        "training_topics": training_topics,
        "all_topics": all_topics,
    }


@dataclass(slots=True)
class TopicsService:
    topics_url: str
    ttl_seconds: int
    _cache_data: dict[str, list[str]] | None = None
    _cache_events: list[dict[str, Any]] = field(default_factory=list)
    _cache_occurrences: list[dict[str, Any]] = field(default_factory=list)
    _cache_expires_at: float = 0.0

    async def get_topics(self) -> dict[str, list[str]]:
        now = monotonic()
        if self._cache_data is not None and now < self._cache_expires_at:
            logger.info(
                "Using cached topics payload (interests=%d, training_topics=%d)",
                len(self._cache_data["interests"]),
                len(self._cache_data["training_topics"]),
            )
            return self._cache_data

        logger.info("Fetching topics feed from %s", self.topics_url)
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(self.topics_url)
            response.raise_for_status()
            payload = response.json()

        parsed = parse_topics_payload(payload)
        events = payload.get("events", []) if isinstance(payload, dict) else []
        occurrences = payload.get("occurrences", []) if isinstance(payload, dict) else []
        if not isinstance(events, list):
            events = []
        if not isinstance(occurrences, list):
            occurrences = []
        self._cache_data = parsed
        self._cache_events = [item for item in events if isinstance(item, dict)]
        self._cache_occurrences = [item for item in occurrences if isinstance(item, dict)]
        self._cache_expires_at = now + max(1, self.ttl_seconds)
        logger.info(
            "Loaded topics feed (interests=%d, training_topics=%d, all=%d)",
            len(parsed["interests"]),
            len(parsed["training_topics"]),
            len(parsed["all_topics"]),
        )
        logger.info(
            "Loaded feed records events=%d occurrences=%d",
            len(self._cache_events),
            len(self._cache_occurrences),
        )
        return parsed

    def get_cached_events(self) -> list[dict[str, Any]]:
        return self._cache_events or []

    def get_cached_occurrences(self) -> list[dict[str, Any]]:
        return self._cache_occurrences or []
