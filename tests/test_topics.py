import asyncio

import httpx

from app.topics import TopicsFetchError, TopicsService, parse_topics_payload


def test_parse_topics_payload_handles_expected_keys() -> None:
    payload = {
        "interests": ["welding", "cad"],
        "training_topics": [{"title": "laser cutter"}, {"name": "3d printing"}],
    }
    parsed = parse_topics_payload(payload)
    assert parsed["interests"] == ["welding", "cad"]
    assert parsed["training_topics"] == ["laser cutter", "3d printing"]
    assert parsed["all_topics"] == ["3d printing", "cad", "laser cutter", "welding"]


def test_parse_topics_payload_fallback_topics() -> None:
    payload = {"topics": [{"name": "networking"}, "soldering"]}
    parsed = parse_topics_payload(payload)
    assert parsed["all_topics"] == ["networking", "soldering"]


def test_parse_topics_payload_dedupes_and_supports_camel_case_training_key() -> None:
    payload = {
        "interests": ["Laser", "soldering", "laser"],
        "trainingTopics": ["Admin", "Soldering"],
    }
    parsed = parse_topics_payload(payload)
    assert parsed["all_topics"] == ["Admin", "Laser", "soldering"]


def test_parse_topics_payload_falls_back_to_event_titles() -> None:
    payload = {
        "events": [
            {"title": "Dorkbot"},
            {"name": "Exploit Workshop"},
            {"title": "dorkbot"},
        ]
    }
    parsed = parse_topics_payload(payload)
    assert parsed["all_topics"] == ["Dorkbot", "Exploit Workshop"]


def test_topics_service_raises_clean_fetch_error_without_cache(monkeypatch) -> None:
    service = TopicsService("https://example.invalid/topics.json", 300)

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return False

        async def get(self, url):  # noqa: ANN001
            del url
            raise httpx.ConnectTimeout("timed out")

    monkeypatch.setattr("app.topics.httpx.AsyncClient", lambda timeout: FakeAsyncClient())

    try:
        asyncio.run(service.get_topics())
        raise AssertionError("Expected TopicsFetchError")
    except TopicsFetchError as exc:
        message = str(exc)
        assert "url=https://example.invalid/topics.json" in message
        assert "error_type=ConnectTimeout" in message


def test_topics_service_uses_cached_topics_on_fetch_failure(monkeypatch) -> None:
    service = TopicsService("https://example.invalid/topics.json", 300)
    service._cache_data = {  # noqa: SLF001
        "interests": ["welding"],
        "training_topics": ["laser"],
        "all_topics": ["laser", "welding"],
    }
    service._cache_expires_at = 0.0  # noqa: SLF001

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return False

        async def get(self, url):  # noqa: ANN001
            del url
            raise httpx.ConnectTimeout("timed out")

    monkeypatch.setattr("app.topics.httpx.AsyncClient", lambda timeout: FakeAsyncClient())
    result = asyncio.run(service.get_topics())
    assert result["all_topics"] == ["laser", "welding"]
