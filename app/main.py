from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import load_config
from app.llm import LlmError, LlmService
from app.models import AppSettings, ChatRequest, ChatResponse, SettingsUpdate
from app.rag import RagService
from app.storage import Storage
from app.topics import TopicsService

config = load_config()
default_settings = AppSettings(
    provider=config.default_provider,  # type: ignore[arg-type]
    llm_base_url=config.default_llm_base_url,
    model=config.default_model,
    system_prompt=config.default_system_prompt,
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


@app.get("/api/settings", response_model=AppSettings)
async def get_settings() -> AppSettings:
    return storage.get_settings()


@app.put("/api/settings", response_model=AppSettings)
async def put_settings(update: SettingsUpdate) -> AppSettings:
    settings = AppSettings.model_validate(update.model_dump())
    storage.save_settings(settings)
    return settings


@app.get("/api/topics")
async def get_topics() -> dict[str, list[str]]:
    try:
        return await topics_service.get_topics()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to load topics feed: {exc}") from exc


@app.get("/api/history")
async def get_history(limit: int = Query(default=50, ge=1, le=500)):
    return storage.get_history(limit=limit)


@app.post("/api/chat", response_model=ChatResponse)
async def post_chat(request: ChatRequest) -> ChatResponse:
    settings = storage.get_settings()
    context = await rag_service.retrieve(request.question) if request.use_rag else []

    try:
        answer = await llm_service.chat(
            settings=settings,
            question=request.question,
            context=context,
        )
    except LlmError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    storage.append_history(
        provider=settings.provider,
        model=settings.model,
        system_prompt=settings.system_prompt,
        question=request.question,
        answer=answer,
        rag_collections=[chunk.collection for chunk in context],
        rag_hits=len(context),
        config_snapshot=settings.model_dump(),
    )
    return ChatResponse(answer=answer, context=context)


def main() -> None:
    import uvicorn

    uvicorn.run("app.main:app", host=config.app_host, port=config.app_port, reload=False)


if __name__ == "__main__":
    main()
