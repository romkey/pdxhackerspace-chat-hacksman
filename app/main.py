from __future__ import annotations

import logging
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from time import perf_counter

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import load_config
from app.llm import LlmError, LlmService
from app.models import (
    AppSettings,
    ChatRequest,
    ChatResponse,
    MetaResponse,
    ModelsResponse,
    RagCollectionsResponse,
    SettingsUpdate,
)
from app.rag import RagService
from app.storage import Storage
from app.topics import TopicsService

config = load_config()
logging.basicConfig(
    level=getattr(logging, config.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
default_settings = AppSettings(
    provider=config.default_provider,  # type: ignore[arg-type]
    llm_base_url=config.default_llm_base_url,
    model=config.default_model,
    system_prompt=config.default_system_prompt,
    enabled_rag_collections=config.rag_collections,
)
storage = Storage(db_path=config.db_path, default_settings=default_settings)
topics_service = TopicsService(config.topics_url, config.topics_cache_ttl_seconds)
rag_service = RagService(
    qdrant_url=config.qdrant_url,
    qdrant_api_key=config.qdrant_api_key,
    collections=config.rag_collections,
    embedding_url=config.embedding_url,
    embedding_model=config.embedding_model,
    top_k=config.rag_top_k,
)
llm_service = LlmService()
logger.info(
    "Configured Qdrant collections from env: %s",
    ",".join(config.rag_collections) if config.rag_collections else "(none)",
)
try:
    APP_VERSION = package_version("chat-hacksman")
except PackageNotFoundError:
    APP_VERSION = "unknown"

app = FastAPI(title="Chat Hacksman")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def root() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/meta", response_model=MetaResponse)
async def get_meta() -> MetaResponse:
    return MetaResponse(
        app_name="Chat Hacksman",
        version=APP_VERSION,
        repo_url=config.repo_url,
    )


@app.get("/api/settings", response_model=AppSettings)
async def get_settings() -> AppSettings:
    settings = storage.get_settings()
    if not settings.enabled_rag_collections and config.rag_collections:
        # Backward-compatible default for older stored settings payloads.
        settings.enabled_rag_collections = config.rag_collections
        storage.save_settings(settings)
    return settings


@app.put("/api/settings", response_model=AppSettings)
async def put_settings(update: SettingsUpdate) -> AppSettings:
    settings = AppSettings.model_validate(update.model_dump())
    settings.enabled_rag_collections = [
        collection
        for collection in settings.enabled_rag_collections
        if collection in set(config.rag_collections)
    ]
    storage.save_settings(settings)
    logger.info(
        "Saved settings provider=%s model=%s enabled_rag_collections=%s",
        settings.provider,
        settings.model,
        (
            ",".join(settings.enabled_rag_collections)
            if settings.enabled_rag_collections
            else "(none)"
        ),
    )
    return settings


@app.get("/api/rag/collections", response_model=RagCollectionsResponse)
async def get_rag_collections() -> RagCollectionsResponse:
    settings = storage.get_settings()
    enabled = [
        collection
        for collection in settings.enabled_rag_collections
        if collection in set(config.rag_collections)
    ]
    return RagCollectionsResponse(
        available_collections=config.rag_collections,
        enabled_collections=enabled,
    )


@app.get("/api/models", response_model=ModelsResponse)
async def get_models(
    provider: str | None = Query(default=None),
    base_url: str | None = Query(default=None),
) -> ModelsResponse:
    settings = storage.get_settings()
    requested_provider = (provider or settings.provider).strip()
    target_provider = (
        requested_provider
        if requested_provider in {"ollama", "llama_cpp"}
        else settings.provider
    )
    target_base_url = (base_url or settings.llm_base_url).strip()

    if target_provider != "ollama":
        return ModelsResponse(provider=target_provider, models=[], error=None)

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{target_base_url.rstrip('/')}/api/tags")
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:
        logger.warning("Failed to fetch Ollama models from %s: %s", target_base_url, exc)
        return ModelsResponse(provider="ollama", models=[], error=str(exc))

    raw_models = payload.get("models", [])
    if not isinstance(raw_models, list):
        return ModelsResponse(
            provider="ollama",
            models=[],
            error="Unexpected tags response format.",
        )

    names: list[str] = []
    for item in raw_models:
        if isinstance(item, dict):
            name = item.get("name")
            if isinstance(name, str) and name.strip():
                names.append(name.strip())

    names = sorted(list(dict.fromkeys(names)), key=lambda item: item.casefold())
    return ModelsResponse(provider="ollama", models=names, error=None)


@app.get("/api/topics")
async def get_topics() -> dict[str, list[str]]:
    try:
        return await topics_service.get_topics()
    except Exception as exc:
        logger.exception("Topics feed request failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Failed to load topics feed: {exc}") from exc


@app.get("/api/history")
async def get_history(limit: int = Query(default=50, ge=1, le=500)):
    return storage.get_history(limit=limit)


@app.delete("/api/history")
async def delete_history() -> dict[str, int]:
    deleted_rows = storage.clear_history()
    logger.info("Cleared chat history rows=%d", deleted_rows)
    return {"deleted_rows": deleted_rows}


@app.delete("/api/history/latest")
async def delete_latest_history() -> dict[str, int]:
    deleted_rows = storage.delete_latest_history()
    logger.info("Deleted most recent history row count=%d", deleted_rows)
    return {"deleted_rows": deleted_rows}


@app.post("/api/chat", response_model=ChatResponse)
async def post_chat(request: ChatRequest) -> ChatResponse:
    request_started = perf_counter()
    settings = storage.get_settings()
    enabled_collections = [
        collection
        for collection in settings.enabled_rag_collections
        if collection in set(config.rag_collections)
    ]
    rag_started = perf_counter()
    if request.use_rag:
        logger.info(
            "Chat request using RAG=%s enabled_collections=%s",
            request.use_rag,
            ",".join(enabled_collections) if enabled_collections else "(none)",
        )
        context = await rag_service.retrieve(
            request.question,
            enabled_collections=enabled_collections,
        )
    else:
        logger.info("Chat request with RAG disabled")
        context = []
    rag_latency_ms = (perf_counter() - rag_started) * 1000.0

    try:
        llm_result = await llm_service.chat(
            settings=settings,
            question=request.question,
            context=context,
        )
    except LlmError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    total_latency_ms = (perf_counter() - request_started) * 1000.0

    metrics = {
        "total_latency_ms": round(total_latency_ms, 2),
        "rag_latency_ms": round(rag_latency_ms, 2),
        "llm_latency_ms": round(llm_result.llm_latency_ms, 2),
        "prompt_tokens": llm_result.input_tokens,
        "completion_tokens": llm_result.output_tokens,
        "total_tokens": llm_result.total_tokens,
        "tokens_per_second": (
            round(llm_result.tokens_per_second, 2)
            if isinstance(llm_result.tokens_per_second, (int, float))
            else None
        ),
        "context_chunks_used": len(context),
        "temporary_chat": request.temporary_chat,
        "history_saved": not request.temporary_chat,
        "provider_metrics": llm_result.provider_metrics,
    }

    if request.temporary_chat:
        logger.info("Temporary chat mode enabled; skipping history persistence")
    else:
        storage.append_history(
            provider=settings.provider,
            model=settings.model,
            system_prompt=settings.system_prompt,
            question=request.question,
            answer=llm_result.answer,
            rag_collections=[chunk.collection for chunk in context],
            rag_hits=len(context),
            config_snapshot={**settings.model_dump(), "last_metrics": metrics},
        )
    return ChatResponse(answer=llm_result.answer, context=context, metrics=metrics)


def main() -> None:
    import uvicorn

    uvicorn.run("app.main:app", host=config.app_host, port=config.app_port, reload=False)


if __name__ == "__main__":
    main()
