import asyncio
import json
from collections.abc import Callable

import httpx
import pytest

from chat_hacksman.llm import LlmError, LlmService, _build_system_prompt
from chat_hacksman.models import AppSettings, ContextChunk, SamplingTweaks
from tests.conftest import Handler


def test_build_system_prompt_enriches_calibre_chunk_metadata() -> None:
    prompt = _build_system_prompt(
        "Base prompt",
        [
            ContextChunk(
                collection="calibre_books",
                score=0.88,
                text="In the beginning...",
                metadata={
                    "title": "The Bible",
                    "authors": ["Various Authors"],
                    "chapter_title": "Genesis",
                    "chunk_type": "content",
                    "source_url": "http://calibre/download/42/epub",
                },
            )
        ],
    )
    assert 'Book: "The Bible" by Various Authors' in prompt
    assert "Chapter: Genesis" in prompt
    assert "Type: content" in prompt
    assert "Source: http://calibre/download/42/epub" in prompt
    assert "collection=calibre_books" in prompt


def test_build_system_prompt_keeps_default_format_for_non_calibre_chunks() -> None:
    prompt = _build_system_prompt(
        "Base prompt",
        [
            ContextChunk(
                collection="wiki",
                score=0.91,
                text="Safety glasses are required.",
                metadata={},
            )
        ],
    )
    assert "collection=wiki" in prompt
    assert "Safety glasses are required." in prompt


def test_build_system_prompt_does_not_include_retrieval_scores() -> None:
    prompt = _build_system_prompt(
        "Base prompt",
        [
            ContextChunk(
                collection="wiki",
                score=0.91,
                text="Safety glasses are required.",
                metadata={},
            )
        ],
    )
    assert "score=" not in prompt
    assert "(score:" not in prompt


def test_build_system_prompt_formats_calibre_alias_collection() -> None:
    prompt = _build_system_prompt(
        "Base prompt",
        [
            ContextChunk(
                collection="calibre_ebooks",
                score=0.85,
                text="The text body",
                metadata={"title": "Book", "authors": ["Author"]},
            )
        ],
    )
    assert 'Book: "Book" by Author' in prompt


def test_build_system_prompt_formats_slack_chunk_metadata() -> None:
    prompt = _build_system_prompt(
        "Base prompt",
        [
            ContextChunk(
                collection="slack",
                score=0.88,
                text="Let's discuss laser cutter maintenance.",
                metadata={
                    "channel_name": "tools",
                    "user_name": "alice",
                    "datetime": "2026-03-01 10:22",
                    "doc_type": "thread_summary",
                    "reply_count": 3,
                    "reaction_count": 7,
                    "permalink": "https://slack.example/thread/123",
                },
            )
        ],
    )
    assert "#tools" in prompt
    assert "@alice" in prompt
    assert "[thread_summary]" in prompt
    assert "3 replies" in prompt
    assert "7 reactions" in prompt
    assert "Link: https://slack.example/thread/123" in prompt


def test_build_system_prompt_adds_slack_guidance_when_present() -> None:
    prompt = _build_system_prompt(
        "Base prompt",
        [
            ContextChunk(
                collection="slack",
                score=0.9,
                text="Compressor discussion",
                metadata={},
            )
        ],
    )
    assert "For Slack context: cite channels as #channel-name" in prompt


def test_build_system_prompt_does_not_add_slack_guidance_without_slack_chunks() -> None:
    prompt = _build_system_prompt(
        "Base prompt",
        [
            ContextChunk(
                collection="wiki",
                score=0.9,
                text="General docs",
                metadata={},
            )
        ],
    )
    assert "For Slack context: cite channels as #channel-name" not in prompt


def test_build_system_prompt_formats_events_chunk_metadata() -> None:
    prompt = _build_system_prompt(
        "Base prompt",
        [
            ContextChunk(
                collection="events",
                score=0.93,
                text="Laser cutter training this week.",
                metadata={
                    "title": "Laser Cutter Training",
                    "record_type": "event_summary",
                    "frequency": "weekly",
                    "occurrence_count": 12,
                    "has_future_occurrences": True,
                    "next_start_time": "2026-03-10T18:00:00-08:00",
                    "duration": 120,
                    "location": "Section 2",
                    "tags": ["training", "laser"],
                    "temporal_status": "future",
                    "source_url": "https://events.example/laser",
                },
            )
        ],
    )
    assert "Event: Laser Cutter Training" in prompt
    assert "Schedule: weekly (12 occurrences)" in prompt
    assert "Upcoming: yes" in prompt
    assert "Where: Section 2" in prompt
    assert "Tags: training, laser" in prompt
    assert "Link: https://events.example/laser" in prompt


def test_build_system_prompt_adds_events_guidance_when_present() -> None:
    prompt = _build_system_prompt(
        "Base prompt",
        [
            ContextChunk(
                collection="events",
                score=0.8,
                text="Upcoming open house",
                metadata={},
            )
        ],
    )
    assert "For events context: always mention the date and time" in prompt


def _completion(content: object = "ok", **extra: object) -> httpx.Response:
    body: dict[str, object] = {
        "model": "served-model",
        "choices": [
            {"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
    }
    return httpx.Response(200, json=body | extra)


def test_chat_uses_openai_chat_completions(mock_http: Callable[[Handler], None]) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _completion(timings={"predicted_ms": 500.0, "prompt_ms": 20.0})

    mock_http(handler)
    service = LlmService(base_url="http://llm.local/v1/", api_key="llm-secret", default_model="m")
    settings = AppSettings(system_prompt="Base prompt", tweaks=SamplingTweaks(seed=7))
    result = asyncio.run(service.chat(settings=settings, question="hello", context=[]))

    assert result.answer == "ok"
    assert result.model == "m"
    assert (result.input_tokens, result.output_tokens, result.total_tokens) == (12, 4, 16)
    # 4 tokens in llama-server's 500 ms of generation.
    assert result.tokens_per_second == pytest.approx(8.0)
    assert result.provider_metrics["model"] == "served-model"
    assert result.provider_metrics["finish_reason"] == "stop"
    (request,) = seen
    assert str(request.url) == "http://llm.local/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer llm-secret"
    body = json.loads(request.content)
    assert body["model"] == "m"
    assert body["messages"] == [
        {"role": "system", "content": "Base prompt"},
        {"role": "user", "content": "hello"},
    ]
    assert (body["temperature"], body["top_p"], body["max_tokens"], body["seed"]) == (
        0.2,
        0.95,
        512,
        7,
    )
    assert body["stream"] is False


def test_saved_model_wins_and_seed_is_optional(mock_http: Callable[[Handler], None]) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _completion()

    mock_http(handler)
    service = LlmService(base_url="http://llm.local/v1", default_model="env-model")
    settings = AppSettings(model="picked", system_prompt="p")
    result = asyncio.run(service.chat(settings=settings, question="q", context=[]))
    body = json.loads(seen[0].content)
    assert body["model"] == result.model == "picked"
    assert "seed" not in body
    assert "Authorization" not in seen[0].headers


def test_no_model_configured_uses_the_servers_first(mock_http: Callable[[Handler], None]) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "only-model"}]})
        return _completion()

    mock_http(handler)
    service = LlmService(base_url="http://llm.local/v1")
    result = asyncio.run(
        service.chat(settings=AppSettings(system_prompt="p"), question="q", context=[])
    )
    assert result.model == "only-model"
    assert seen == ["/v1/models", "/v1/chat/completions"]


def test_no_model_anywhere_is_an_error(mock_http: Callable[[Handler], None]) -> None:
    mock_http(lambda request: httpx.Response(200, json={"data": []}))
    service = LlmService(base_url="http://llm.local/v1")
    with pytest.raises(LlmError, match="set LLM_MODEL"):
        asyncio.run(service.resolve_model(""))


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(400, text="bad model"),
        httpx.Response(200, text="not json"),
        httpx.Response(200, json=["a list"]),
        httpx.Response(200, json={"choices": []}),
        _completion(content=None),
    ],
)
def test_bad_chat_responses_raise(
    mock_http: Callable[[Handler], None], response: httpx.Response
) -> None:
    mock_http(lambda request: response)
    service = LlmService(base_url="http://llm.local/v1", default_model="m")
    with pytest.raises(LlmError):
        asyncio.run(service.chat(settings=AppSettings(system_prompt="p"), question="q", context=[]))


def test_connection_errors_are_retried(mock_http: Callable[[Handler], None]) -> None:
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) == 1:
            raise httpx.ConnectError("refused", request=request)
        return _completion()

    mock_http(handler)
    service = LlmService(
        base_url="http://llm.local/v1", default_model="m", retry_backoff_seconds=0.0
    )
    result = asyncio.run(
        service.chat(settings=AppSettings(system_prompt="p"), question="q", context=[])
    )
    assert result.answer == "ok"
    assert len(attempts) == 2

    mock_http(lambda request: (_ for _ in ()).throw(httpx.ConnectError("refused", request=request)))
    with pytest.raises(LlmError, match="after 2 attempt"):
        asyncio.run(service.chat(settings=AppSettings(system_prompt="p"), question="q", context=[]))
