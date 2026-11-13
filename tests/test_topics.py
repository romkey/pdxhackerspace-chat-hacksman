import asyncio
from collections.abc import Callable

import httpx
import pytest

from chat_hacksman.topics import TopicsFetchError, TopicsService, parse_topics_payload
from tests.conftest import Handler


def timeout(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectTimeout("timed out", request=request)


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


def test_topics_service_raises_clean_fetch_error_without_cache(
    mock_http: Callable[[Handler], None],
) -> None:
    service = TopicsService("https://example.invalid/topics.json", 300)

    mock_http(timeout)

    with pytest.raises(TopicsFetchError) as exc_info:
        asyncio.run(service.get_topics())
    message = str(exc_info.value)
    assert "url=https://example.invalid/topics.json" in message
    assert "error_type=ConnectTimeout" in message


def test_topics_service_uses_cached_topics_on_fetch_failure(
    mock_http: Callable[[Handler], None],
) -> None:
    service = TopicsService("https://example.invalid/topics.json", 300)
    service._cache_data = {
        "interests": ["welding"],
        "training_topics": ["laser"],
        "all_topics": ["laser", "welding"],
    }
    service._cache_expires_at = 0.0

    mock_http(timeout)
    result = asyncio.run(service.get_topics())
    assert result["all_topics"] == ["laser", "welding"]
