from __future__ import annotations

import asyncio
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import httpx

from app.models import AppSettings, ContextChunk


class LlmError(RuntimeError):
    pass


@dataclass(slots=True)
class LlmResult:
    answer: str
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    llm_latency_ms: float
    tokens_per_second: float | None
    provider_metrics: dict[str, Any]


def _is_slack_collection(collection: str) -> bool:
    normalized = collection.strip().lower()
    return normalized == "slack" or normalized.startswith("slack_") or normalized.startswith(
        "slack-"
    )


def _is_events_collection(collection: str) -> bool:
    normalized = collection.strip().lower()
    return (
        normalized == "events"
        or normalized.startswith("events_")
        or normalized.startswith("events-")
    )


def _is_calibre_collection(collection: str) -> bool:
    normalized = collection.strip().lower()
    return (
        normalized == "calibre_books"
        or normalized.startswith("calibre_")
        or normalized.startswith("calibre-")
    )


def _to_positive_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value if value > 0 else 0
    if isinstance(value, str):
        try:
            parsed = int(value.strip())
            return parsed if parsed > 0 else 0
        except ValueError:
            return 0
    return 0


def _format_calibre_chunk(idx: int, chunk: ContextChunk) -> str:
    metadata = chunk.metadata if isinstance(chunk.metadata, dict) else {}
    header_parts: list[str] = []
    title = metadata.get("title")
    if isinstance(title, str) and title.strip():
        book_line = f'Book: "{title.strip()}"'
        authors = metadata.get("authors")
        if isinstance(authors, list):
            author_names = [str(item).strip() for item in authors if str(item).strip()]
            if author_names:
                book_line += f" by {', '.join(author_names)}"
        elif isinstance(authors, str) and authors.strip():
            book_line += f" by {authors.strip()}"
        header_parts.append(book_line)

    chapter_title = metadata.get("chapter_title")
    if isinstance(chapter_title, str) and chapter_title.strip():
        header_parts.append(f"Chapter: {chapter_title.strip()}")

    chunk_type = metadata.get("chunk_type")
    if isinstance(chunk_type, str) and chunk_type.strip():
        header_parts.append(f"Type: {chunk_type.strip()}")

    source_url = metadata.get("source_url")
    if isinstance(source_url, str) and source_url.strip():
        header_parts.append(f"Source: {source_url.strip()}")

    header_parts.append(f"collection={chunk.collection}")
    header_parts.append(f"score={chunk.score:.4f}")
    header = " | ".join(header_parts)
    return f"[{idx}] {header}\n{chunk.text}"


def _format_slack_chunk(idx: int, chunk: ContextChunk) -> str:
    metadata = chunk.metadata if isinstance(chunk.metadata, dict) else {}
    header_parts: list[str] = []

    channel = metadata.get("channel_name") or metadata.get("channel_id")
    if isinstance(channel, str) and channel.strip():
        channel_name = channel.strip()
        if not channel_name.startswith("#"):
            channel_name = f"#{channel_name}"
        header_parts.append(channel_name)

    user = metadata.get("user_name") or metadata.get("user_id")
    if isinstance(user, str) and user.strip():
        user_name = user.strip()
        if not user_name.startswith("@"):
            user_name = f"@{user_name}"
        header_parts.append(user_name)

    dt = metadata.get("datetime") or metadata.get("date") or metadata.get("ts")
    if isinstance(dt, str) and dt.strip():
        header_parts.append(dt.strip())

    doc_type = metadata.get("doc_type")
    if isinstance(doc_type, str) and doc_type.strip() and doc_type.strip() != "message":
        header_parts.append(f"[{doc_type.strip()}]")

    reply_count = _to_positive_int(metadata.get("reply_count"))
    if reply_count:
        header_parts.append(f"{reply_count} replies")

    reaction_count = _to_positive_int(metadata.get("reaction_count"))
    if reaction_count:
        header_parts.append(f"{reaction_count} reactions")

    header = " | ".join(header_parts) if header_parts else f"collection={chunk.collection}"
    result = f"[{idx}] {header} (score={chunk.score:.4f})\n{chunk.text}"

    permalink = metadata.get("permalink")
    if isinstance(permalink, str) and permalink.strip():
        result += f"\nLink: {permalink.strip()}"
    return result


def _format_events_chunk(idx: int, chunk: ContextChunk) -> str:
    metadata = chunk.metadata if isinstance(chunk.metadata, dict) else {}
    parts: list[str] = []

    title = metadata.get("title")
    if isinstance(title, str) and title.strip():
        parts.append(f"Event: {title.strip()}")

    record_type = metadata.get("record_type")
    if record_type == "event_summary":
        freq = metadata.get("frequency")
        count = metadata.get("occurrence_count")
        if isinstance(freq, str) and freq.strip() and isinstance(count, int):
            parts.append(f"Schedule: {freq.strip()} ({count} occurrences)")
        has_future = metadata.get("has_future_occurrences")
        if isinstance(has_future, bool):
            parts.append("Upcoming: yes" if has_future else "Upcoming: no")

    start = metadata.get("start_time") or metadata.get("next_start_time")
    if isinstance(start, str) and start.strip():
        parts.append(f"When: {start.strip()}")

    duration = metadata.get("duration")
    if isinstance(duration, int) and duration > 0:
        parts.append(f"Duration: {duration}")
    elif isinstance(duration, str) and duration.strip():
        parts.append(f"Duration: {duration.strip()}")

    location: str | None = None
    raw_location = metadata.get("location")
    if isinstance(raw_location, str) and raw_location.strip():
        location = raw_location.strip()
    else:
        locations = metadata.get("locations")
        if isinstance(locations, list):
            names = [str(item).strip() for item in locations if str(item).strip()]
            if names:
                location = ", ".join(names)
    if location:
        parts.append(f"Where: {location}")

    tags = metadata.get("tags")
    if isinstance(tags, list):
        tag_names = [str(item).strip() for item in tags if str(item).strip()]
        if tag_names:
            parts.append(f"Tags: {', '.join(tag_names)}")

    temporal = metadata.get("temporal_status")
    if isinstance(temporal, str) and temporal.strip():
        parts.append(f"Status: {temporal.strip()}")

    header = " | ".join(parts) if parts else f"collection={chunk.collection}"
    result = f"[{idx}] {header} (score={chunk.score:.4f})\n{chunk.text}"
    source_url = metadata.get("source_url")
    if isinstance(source_url, str) and source_url.strip():
        result += f"\nLink: {source_url.strip()}"
    return result


def _format_chunk(idx: int, chunk: ContextChunk) -> str:
    if _is_calibre_collection(chunk.collection):
        return _format_calibre_chunk(idx, chunk)
    if _is_slack_collection(chunk.collection):
        return _format_slack_chunk(idx, chunk)
    if _is_events_collection(chunk.collection):
        return _format_events_chunk(idx, chunk)
    return f"[{idx}] collection={chunk.collection} score={chunk.score:.4f}\n{chunk.text}"


def _build_system_prompt(base_prompt: str, context: list[ContextChunk]) -> str:
    if not context:
        return base_prompt

    has_slack = any(_is_slack_collection(chunk.collection) for chunk in context)
    has_events = any(_is_events_collection(chunk.collection) for chunk in context)
    context_parts = []
    for idx, chunk in enumerate(context, start=1):
        context_parts.append(_format_chunk(idx, chunk))
    context_block = "\n\n".join(context_parts)
    guidance = (
        "Use the retrieved context when it is relevant. If it conflicts with known facts, "
        "say what is uncertain."
    )
    if has_slack:
        guidance += (
            "\n\nFor Slack context: cite channels as #channel-name and users as @username. "
            "When recommending channels, prefer those with higher activity and reaction counts. "
            "Include permalinks when available so members can read the full thread. "
            "Messages with many reactions were found valuable by the community."
        )
    if has_events:
        guidance += (
            "\n\nFor events context: always mention the date and time when discussing events. "
            "Distinguish between upcoming events (status: future/current) and past ones. "
            "For recurring events, mention the frequency. "
            "Note that event information is periodically synchronized and may not reflect very "
            "recent changes - direct members to the source URL to confirm details."
        )
    return (
        f"{base_prompt}\n\n"
        f"{guidance}\n\n"
        f"Retrieved context:\n{context_block}"
    )


def _ns_to_ms(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value) / 1_000_000.0
    return None


@dataclass(slots=True)
class LlmService:
    request_timeout_seconds: float = 180.0
    retry_attempts: int = 2
    retry_backoff_seconds: float = 0.5

    async def _post_json_with_retries(self, *, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        last_exc: httpx.HTTPError | None = None
        max_attempts = max(1, self.retry_attempts)
        for attempt in range(1, max_attempts + 1):
            try:
                async with httpx.AsyncClient(timeout=self.request_timeout_seconds) as client:
                    response = await client.post(url, json=payload)
                if response.status_code >= 400:
                    msg = f"LLM request failed ({response.status_code}): {response.text}"
                    raise LlmError(msg)
                try:
                    data = response.json()
                except ValueError as exc:
                    raise LlmError(f"LLM provider returned invalid JSON: {exc}") from exc
                if not isinstance(data, dict):
                    raise LlmError("LLM provider returned a non-object JSON payload.")
                return data
            except LlmError:
                raise
            except httpx.HTTPError as exc:
                last_exc = exc
                if attempt < max_attempts:
                    await asyncio.sleep(self.retry_backoff_seconds * attempt)
                    continue
                break

        error_type = type(last_exc).__name__ if last_exc is not None else "unknown"
        msg = (
            f"LLM request failed url={url} after {max_attempts} attempt(s): "
            f"error_type={error_type} error={last_exc}"
        )
        raise LlmError(msg)

    async def chat(
        self,
        *,
        settings: AppSettings,
        question: str,
        context: list[ContextChunk],
    ) -> LlmResult:
        started = perf_counter()
        provider = settings.provider
        if provider == "ollama":
            answer, provider_metrics = await self._chat_ollama(
                settings=settings, question=question, context=context
            )
        elif provider == "llama_cpp":
            answer, provider_metrics = await self._chat_llama_cpp(
                settings=settings, question=question, context=context
            )
        else:
            msg = f"Unsupported provider '{provider}'"
            raise LlmError(msg)

        llm_latency_ms = (perf_counter() - started) * 1000.0
        input_tokens = provider_metrics.get("prompt_tokens")
        output_tokens = provider_metrics.get("completion_tokens")
        total_tokens = provider_metrics.get("total_tokens")

        tokens_per_second = None
        if isinstance(output_tokens, int) and output_tokens > 0:
            eval_ms = provider_metrics.get("eval_duration_ms")
            if isinstance(eval_ms, (int, float)) and float(eval_ms) > 0:
                tokens_per_second = output_tokens / (float(eval_ms) / 1000.0)
            elif llm_latency_ms > 0:
                tokens_per_second = output_tokens / (llm_latency_ms / 1000.0)

        return LlmResult(
            answer=answer,
            input_tokens=input_tokens if isinstance(input_tokens, int) else None,
            output_tokens=output_tokens if isinstance(output_tokens, int) else None,
            total_tokens=total_tokens if isinstance(total_tokens, int) else None,
            llm_latency_ms=llm_latency_ms,
            tokens_per_second=tokens_per_second,
            provider_metrics=provider_metrics,
        )

    async def _chat_ollama(
        self,
        *,
        settings: AppSettings,
        question: str,
        context: list[ContextChunk],
    ) -> tuple[str, dict[str, Any]]:
        url = f"{settings.llm_base_url.rstrip('/')}/api/chat"
        system_prompt = _build_system_prompt(settings.system_prompt, context)
        payload = {
            "model": settings.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            "stream": False,
            "options": {
                "temperature": settings.tweaks.temperature,
                "top_p": settings.tweaks.top_p,
                "num_ctx": settings.tweaks.num_ctx,
                "repeat_penalty": settings.tweaks.repeat_penalty,
                "seed": settings.tweaks.seed,
            },
        }
        data = await self._post_json_with_retries(url=url, payload=payload)
        content = data.get("message", {}).get("content")
        if not isinstance(content, str):
            raise LlmError("Ollama returned an unexpected payload.")

        prompt_eval_count = data.get("prompt_eval_count")
        eval_count = data.get("eval_count")
        total_duration_ms = _ns_to_ms(data.get("total_duration"))
        prompt_eval_duration_ms = _ns_to_ms(data.get("prompt_eval_duration"))
        eval_duration_ms = _ns_to_ms(data.get("eval_duration"))
        provider_metrics: dict[str, Any] = {
            "provider": "ollama",
            "prompt_tokens": int(prompt_eval_count) if isinstance(prompt_eval_count, int) else None,
            "completion_tokens": int(eval_count) if isinstance(eval_count, int) else None,
            "total_tokens": (
                int(prompt_eval_count) + int(eval_count)
                if isinstance(prompt_eval_count, int) and isinstance(eval_count, int)
                else None
            ),
            "total_duration_ms": total_duration_ms,
            "prompt_eval_duration_ms": prompt_eval_duration_ms,
            "eval_duration_ms": eval_duration_ms,
            "load_duration_ms": _ns_to_ms(data.get("load_duration")),
        }
        return content, provider_metrics

    async def _chat_llama_cpp(
        self,
        *,
        settings: AppSettings,
        question: str,
        context: list[ContextChunk],
    ) -> tuple[str, dict[str, Any]]:
        url = f"{settings.llm_base_url.rstrip('/')}/v1/chat/completions"
        system_prompt = _build_system_prompt(settings.system_prompt, context)
        payload = {
            "model": settings.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            "temperature": settings.tweaks.temperature,
            "top_p": settings.tweaks.top_p,
            "max_tokens": settings.tweaks.max_tokens,
            "repeat_penalty": settings.tweaks.repeat_penalty,
            "seed": settings.tweaks.seed,
        }
        data = await self._post_json_with_retries(url=url, payload=payload)

        choices = data.get("choices", [])
        if not isinstance(choices, list) or not choices:
            raise LlmError("llama.cpp returned an unexpected payload.")
        content = choices[0].get("message", {}).get("content")
        if not isinstance(content, str):
            raise LlmError("llama.cpp returned an unexpected message payload.")

        usage = data.get("usage", {}) if isinstance(data.get("usage"), dict) else {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
        provider_metrics = {
            "provider": "llama_cpp",
            "prompt_tokens": int(prompt_tokens) if isinstance(prompt_tokens, int) else None,
            "completion_tokens": (
                int(completion_tokens) if isinstance(completion_tokens, int) else None
            ),
            "total_tokens": int(total_tokens) if isinstance(total_tokens, int) else None,
            "finish_reason": choices[0].get("finish_reason"),
        }
        return content, provider_metrics
