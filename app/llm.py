from __future__ import annotations

from dataclasses import dataclass

import httpx

from app.models import AppSettings, ContextChunk


class LlmError(RuntimeError):
    pass


def _build_system_prompt(base_prompt: str, context: list[ContextChunk]) -> str:
    if not context:
        return base_prompt

    context_parts = []
    for idx, chunk in enumerate(context, start=1):
        context_parts.append(
            f"[{idx}] collection={chunk.collection} score={chunk.score:.4f}\n{chunk.text}"
        )
    context_block = "\n\n".join(context_parts)
    return (
        f"{base_prompt}\n\n"
        "Use the retrieved context when it is relevant. If it conflicts with known facts, "
        "say what is uncertain.\n\n"
        f"Retrieved context:\n{context_block}"
    )


@dataclass(slots=True)
class LlmService:
    async def chat(
        self,
        *,
        settings: AppSettings,
        question: str,
        context: list[ContextChunk],
    ) -> str:
        provider = settings.provider
        if provider == "ollama":
            return await self._chat_ollama(settings=settings, question=question, context=context)
        if provider == "llama_cpp":
            return await self._chat_llama_cpp(settings=settings, question=question, context=context)
        msg = f"Unsupported provider '{provider}'"
        raise LlmError(msg)

    async def _chat_ollama(
        self,
        *,
        settings: AppSettings,
        question: str,
        context: list[ContextChunk],
    ) -> str:
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
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, json=payload)
            if response.status_code >= 400:
                msg = f"Ollama request failed ({response.status_code}): {response.text}"
                raise LlmError(msg)
            data = response.json()
        content = data.get("message", {}).get("content")
        if not isinstance(content, str):
            raise LlmError("Ollama returned an unexpected payload.")
        return content

    async def _chat_llama_cpp(
        self,
        *,
        settings: AppSettings,
        question: str,
        context: list[ContextChunk],
    ) -> str:
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
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, json=payload)
            if response.status_code >= 400:
                msg = f"llama.cpp request failed ({response.status_code}): {response.text}"
                raise LlmError(msg)
            data = response.json()

        choices = data.get("choices", [])
        if not isinstance(choices, list) or not choices:
            raise LlmError("llama.cpp returned an unexpected payload.")
        content = choices[0].get("message", {}).get("content")
        if not isinstance(content, str):
            raise LlmError("llama.cpp returned an unexpected message payload.")
        return content
