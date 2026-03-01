from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Any

import httpx


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


def parse_topics_payload(payload: Any) -> dict[str, list[str]]:
    if not isinstance(payload, dict):
        return {"interests": [], "training_topics": [], "all_topics": []}

    interests = _to_topic_strings(payload.get("interests", []))
    training_topics = _to_topic_strings(payload.get("training_topics", []))

    if not interests and not training_topics:
        # Flexible fallback for unknown payloads.
        fallback = _to_topic_strings(payload.get("topics", []))
        return {"interests": fallback, "training_topics": [], "all_topics": fallback}

    all_topics = list(dict.fromkeys(interests + training_topics))
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
    _cache_expires_at: float = 0.0

    async def get_topics(self) -> dict[str, list[str]]:
        now = monotonic()
        if self._cache_data is not None and now < self._cache_expires_at:
            return self._cache_data

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(self.topics_url)
            response.raise_for_status()
            payload = response.json()

        parsed = parse_topics_payload(payload)
        self._cache_data = parsed
        self._cache_expires_at = now + max(1, self.ttl_seconds)
        return parsed
