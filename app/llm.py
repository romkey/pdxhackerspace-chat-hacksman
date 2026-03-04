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


def _format_chunk(idx: int, chunk: ContextChunk) -> str:
    metadata = chunk.metadata if isinstance(chunk.metadata, dict) else {}
    if chunk.collection != "calibre_books":
        return f"[{idx}] collection={chunk.collection} score={chunk.score:.4f}\n{chunk.text}"

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


def _build_system_prompt(base_prompt: str, context: list[ContextChunk]) -> str:
    if not context:
        return base_prompt

    context_parts = []
    for idx, chunk in enumerate(context, start=1):
        context_parts.append(_format_chunk(idx, chunk))
    context_block = "\n\n".join(context_parts)
    return (
        f"{base_prompt}\n\n"
        "Use the retrieved context when it is relevant. If it conflicts with known facts, "
        "say what is uncertain.\n\n"
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

        msg = f"LLM request failed after {max_attempts} attempt(s): {last_exc}"
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
