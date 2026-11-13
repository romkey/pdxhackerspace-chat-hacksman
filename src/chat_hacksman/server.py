"""The FastAPI app: chat, settings, history, feedback, topics and model endpoints.

create_app() builds it from Settings; nothing here reads the environment.
"""

import base64
import binascii
import logging
import secrets
import socket
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

# Any: feed payloads and chat metrics are untyped JSON.
from typing import Any
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from chat_hacksman import __version__
from chat_hacksman.config import Settings
from chat_hacksman.llm import LlmError, LlmService
from chat_hacksman.models import (
    AppSettings,
    ChatRequest,
    ChatResponse,
    ContextChunk,
    EventRecord,
    FeedbackCreateRequest,
    FeedbackRecord,
    HistoryRecord,
    LlmStatusResponse,
    MetaResponse,
    ModelsResponse,
    OccurrenceRecord,
    RagCollectionsResponse,
    SettingsUpdate,
)
from chat_hacksman.rag import RagService
from chat_hacksman.rerank import Reranker
from chat_hacksman.storage import Storage
from chat_hacksman.topics import TopicsFetchError, TopicsService

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
PUBLIC_PATHS = frozenset({"/health", "/healthz"})


@dataclass(slots=True)
class Services:
    """What the endpoints use, built once per app."""

    settings: Settings
    storage: Storage
    topics: TopicsService
    rag: RagService
    llm: LlmService


def build_services(settings: Settings) -> Services:
    default_settings = AppSettings(
        system_prompt=settings.default_system_prompt,
        enabled_rag_collections=list(settings.rag_collections),
    )
    storage = Storage(db_path=settings.db_path, default_settings=default_settings)
    return Services(
        settings=settings,
        storage=storage,
        topics=TopicsService(settings.topics_url, settings.topics_cache_ttl_seconds),
        rag=RagService(
            qdrant_url=settings.qdrant_url,
            qdrant_api_key=settings.qdrant_api_key,
            collections=list(settings.rag_collections),
            embedding_url=settings.embedding_url,
            embedding_model=settings.embedding_model,
            embedding_context_length=settings.embedding_context_length,
            top_k=settings.rag_top_k,
            top_k_events=settings.rag_top_k_events,
            top_k_slack=settings.rag_top_k_slack,
            top_k_calibre=settings.rag_top_k_calibre,
            min_score=settings.rag_min_score,
            embedding_timeout_seconds=settings.embedding_timeout_seconds,
            embedding_api_key=settings.embedding_api_key,
            embedding_query_prefix=settings.embedding_query_prefix,
            reranker=build_reranker(settings),
            rerank_candidates=settings.rerank_candidates,
            rerank_top_n=settings.rerank_top_n,
            rerank_min_score=settings.rerank_min_score,
        ),
        llm=LlmService(
            base_url=settings.llm_url,
            api_key=settings.llm_api_key,
            default_model=settings.llm_model,
            request_timeout_seconds=settings.llm_timeout_seconds,
            retry_attempts=settings.llm_retry_attempts,
            retry_backoff_seconds=settings.llm_retry_backoff_seconds,
        ),
    )


def build_reranker(settings: Settings) -> Reranker | None:
    """The reranker, if RERANK_URL is set; reranking is off otherwise."""
    if settings.rerank_url is None:
        return None
    return Reranker(
        url=settings.rerank_url,
        model=settings.rerank_model,
        api=settings.rerank_api,
        api_key=settings.rerank_api_key,
        timeout_seconds=settings.rerank_timeout_seconds,
        context_length=settings.rerank_context_length,
    )


def _client_ip_from_request(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for", "")
    if forwarded_for:
        first_ip = forwarded_for.split(",", 1)[0].strip()
        if first_ip:
            return first_ip
    real_ip = request.headers.get("x-real-ip", "").strip()
    if real_ip:
        return real_ip
    cf_ip = request.headers.get("cf-connecting-ip", "").strip()
    if cf_ip:
        return cf_ip
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _resolve_host_ip(base_url: str) -> str:
    hostname = urlparse(base_url).hostname or ""
    if not hostname:
        return "unknown"
    try:
        return socket.gethostbyname(hostname)
    except OSError:
        return "unknown"


def _bearer_headers(api_key: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


async def _check_llm_availability(base_url: str, api_key: str | None) -> bool:
    """Whether the server answers GET {base_url}/models at all (any status)."""
    url = base_url.rstrip("/") + "/models"
    try:
        async with httpx.AsyncClient(timeout=3.0, follow_redirects=True) as client:
            await client.get(url, headers=_bearer_headers(api_key))
    except httpx.HTTPError:
        return False
    return True


def _is_events_schedule_query(question: str) -> bool:
    tokens = set(question.strip().lower().split())
    schedule_words = {
        "event",
        "events",
        "upcoming",
        "next",
        "schedule",
        "when",
        "today",
        "tomorrow",
        "week",
        "month",
    }
    return bool(tokens & schedule_words)


def _build_fallback_events_context(services: Services, limit: int = 3) -> list[ContextChunk]:
    now_unix = int(datetime.now(tz=UTC).timestamp())
    cached_occurrences = services.topics.get_cached_occurrences()
    # Feed payloads are arbitrary JSON objects.
    selected_payloads: list[dict[str, Any]] = []
    for item in cached_occurrences:
        occurs_at_unix_raw = item.get("occurs_at_unix")
        if not isinstance(occurs_at_unix_raw, int):
            continue
        if occurs_at_unix_raw < now_unix:
            continue
        title = item.get("title") if isinstance(item.get("title"), str) else item.get("slug")
        if not isinstance(title, str) or not title.strip():
            event_obj = item.get("event")
            if isinstance(event_obj, dict):
                event_title = event_obj.get("title")
                if isinstance(event_title, str) and event_title.strip():
                    title = event_title
        if not isinstance(title, str) or not title.strip():
            title = "event"
        occurs_at = item.get("occurs_at")
        if not isinstance(occurs_at, str) or not occurs_at.strip():
            occurs_at = item.get("start_time")
        if not isinstance(occurs_at, str) or not occurs_at.strip():
            occurs_at = "unknown time"
        selected_payloads.append(
            {
                "occurs_at_unix": occurs_at_unix_raw,
                "title": title.strip(),
                "occurs_at": occurs_at.strip(),
                "open_to": item.get("open_to"),
                "duration": item.get("duration"),
            }
        )

    selected_payloads.sort(key=lambda row: int(row["occurs_at_unix"]))
    selected_payloads = selected_payloads[:limit]

    if not selected_payloads:
        rows = services.storage.get_occurrences(limit=200)
        upcoming = [row for row in rows if row.occurs_at_unix >= now_unix]
        selected = sorted(upcoming, key=lambda row: row.occurs_at_unix)[:limit]
        for row in selected:
            row_title = row.slug
            event_obj = row.payload.get("event")
            if isinstance(event_obj, dict):
                maybe_title = event_obj.get("title")
                if isinstance(maybe_title, str) and maybe_title.strip():
                    row_title = maybe_title.strip()
            selected_payloads.append(
                {
                    "title": row_title,
                    "occurs_at": row.occurs_at,
                    "open_to": row.open_to,
                    "duration": row.duration,
                }
            )

    out: list[ContextChunk] = []
    for payload in selected_payloads:
        title = str(payload["title"])
        text = f"{title} at {payload['occurs_at']}"
        open_to = payload.get("open_to")
        if isinstance(open_to, str) and open_to.strip():
            text += f" (open_to={open_to})"
        out.append(
            ContextChunk(
                collection="events",
                score=1.0,
                text=text,
                metadata={
                    "title": title,
                    "record_type": "occurrence",
                    "start_time": payload["occurs_at"],
                    "duration": payload.get("duration"),
                    "temporal_status": "future",
                },
            )
        )
    return out


def basic_auth_valid(settings: Settings, authorization_header: str | None) -> bool:
    if not settings.basic_auth_enabled:
        return True
    if not authorization_header:
        return False
    try:
        scheme, encoded = authorization_header.split(" ", 1)
    except ValueError:
        return False
    if scheme.lower() != "basic":
        return False
    try:
        decoded = base64.b64decode(encoded.strip(), validate=True).decode("utf-8")
    except binascii.Error, UnicodeDecodeError:
        return False
    if ":" not in decoded:
        return False
    username, password = decoded.split(":", 1)
    return bool(
        settings.basic_auth_username
        and settings.basic_auth_password
        and secrets.compare_digest(username, settings.basic_auth_username)
        and secrets.compare_digest(password, settings.basic_auth_password)
    )


def create_app(settings: Settings, services: Services | None = None) -> FastAPI:
    """The app, with its routes bound to these settings and services."""
    svc = services or build_services(settings)
    config = svc.settings
    storage = svc.storage
    logger.info(
        "Configured Qdrant collections: %s",
        ",".join(config.rag_collections) if config.rag_collections else "(none)",
    )
    logger.info(
        "Chat LLM at %s, default model=%s",
        config.llm_url,
        config.llm_model or "(the server's first)",
    )
    if config.rerank_url:
        logger.info(
            "Reranking with model=%s at %s (api=%s)",
            config.rerank_model,
            config.rerank_url,
            config.rerank_api,
        )
    else:
        logger.info("Reranking is off; set RERANK_URL to turn it on")
    if bool(config.basic_auth_username) ^ bool(config.basic_auth_password):
        logger.warning(
            "Basic auth partially configured; set both BASIC_AUTH_USERNAME and "
            "BASIC_AUTH_PASSWORD to enable authentication."
        )

    app = FastAPI(title="Chat Hacksman", version=__version__)
    app.state.services = svc
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    def available_collections(enabled: list[str]) -> list[str]:
        allowed = set(config.rag_collections)
        return [collection for collection in enabled if collection in allowed]

    @app.middleware("http")
    async def optional_basic_auth(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if not config.basic_auth_enabled:
            return await call_next(request)
        if request.url.path in PUBLIC_PATHS or request.method == "OPTIONS":
            return await call_next(request)
        if basic_auth_valid(config, request.headers.get("Authorization")):
            return await call_next(request)
        logger.warning(
            "Unauthorized request blocked path=%s method=%s client_ip=%s",
            request.url.path,
            request.method,
            _client_ip_from_request(request),
        )
        return Response(
            content="Unauthorized",
            status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="Chat Hacksman"'},
        )

    @app.get("/")
    async def root() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/meta", response_model=MetaResponse)
    async def get_meta() -> MetaResponse:
        return MetaResponse(app_name="Chat Hacksman", version=__version__, repo_url=config.repo_url)

    @app.get("/api/settings", response_model=AppSettings)
    async def get_settings() -> AppSettings:
        settings = storage.get_settings()
        if not settings.enabled_rag_collections and config.rag_collections:
            # Backward-compatible default for older stored settings payloads.
            settings.enabled_rag_collections = list(config.rag_collections)
            storage.save_settings(settings)
        return settings

    @app.put("/api/settings", response_model=AppSettings)
    async def put_settings(update: SettingsUpdate) -> AppSettings:
        settings = AppSettings.model_validate(update.model_dump())
        settings.enabled_rag_collections = available_collections(settings.enabled_rag_collections)
        storage.save_settings(settings)
        logger.info(
            "Saved settings model=%s enabled_rag_collections=%s",
            settings.model or "(default)",
            ",".join(settings.enabled_rag_collections) or "(none)",
        )
        return settings

    @app.get("/api/llm/status", response_model=LlmStatusResponse)
    async def get_llm_status() -> LlmStatusResponse:
        url = config.llm_url
        available = await _check_llm_availability(url, config.llm_api_key)
        now = datetime.now(UTC)
        previous = storage.get_llm_status(url)
        last_changed_at = now
        if previous is not None:
            previous_available, previous_changed_at, _previous_checked_at = previous
            if previous_available == available:
                last_changed_at = previous_changed_at
            else:
                logger.info(
                    "LLM availability changed url=%s ip=%s available=%s previous=%s",
                    url,
                    _resolve_host_ip(url),
                    available,
                    previous_available,
                )
        storage.upsert_llm_status(
            url=url, is_available=available, last_changed_at=last_changed_at, last_checked_at=now
        )
        return LlmStatusResponse(
            url=url, available=available, last_changed_at=last_changed_at, last_checked_at=now
        )

    @app.get("/api/rag/collections", response_model=RagCollectionsResponse)
    async def get_rag_collections() -> RagCollectionsResponse:
        settings = storage.get_settings()
        return RagCollectionsResponse(
            available_collections=list(config.rag_collections),
            enabled_collections=available_collections(settings.enabled_rag_collections),
        )

    @app.get("/api/models", response_model=ModelsResponse)
    async def get_models() -> ModelsResponse:
        try:
            models = await svc.llm.list_models()
        except LlmError as exc:
            logger.warning("Failed to list models url=%s error=%s", config.llm_url, exc)
            return ModelsResponse(default_model=config.llm_model, models=[], error=str(exc))
        return ModelsResponse(
            default_model=config.llm_model, models=sorted(models, key=str.casefold)
        )

    @app.get("/api/topics")
    async def get_topics() -> dict[str, list[str]]:
        try:
            topics = await svc.topics.get_topics()
        except TopicsFetchError as exc:
            logger.warning("Topics feed request failed: %s", exc)
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        events = svc.topics.get_cached_events()
        occurrences = svc.topics.get_cached_occurrences()
        if events:
            logger.info("Stored events from feed count=%d", storage.upsert_events(events))
        if occurrences:
            logger.info(
                "Stored occurrences from feed count=%d", storage.upsert_occurrences(occurrences)
            )
        return topics

    @app.get("/api/events")
    async def get_events(limit: int = Query(default=200, ge=1, le=1000)) -> list[EventRecord]:
        return storage.get_events(limit=limit)

    @app.get("/api/occurrences")
    async def get_occurrences(
        limit: int = Query(default=500, ge=1, le=5000),
    ) -> list[OccurrenceRecord]:
        return storage.get_occurrences(limit=limit)

    @app.get("/api/history")
    async def get_history(limit: int = Query(default=50, ge=1, le=500)) -> list[HistoryRecord]:
        return storage.get_history(limit=limit)

    @app.get("/api/feedback")
    async def get_feedback(limit: int = Query(default=50, ge=1, le=500)) -> list[FeedbackRecord]:
        return storage.get_feedback(limit=limit)

    @app.post("/api/feedback")
    async def post_feedback(
        request: FeedbackCreateRequest, http_request: Request
    ) -> dict[str, int]:
        feedback_id = storage.append_feedback(request)
        logger.info(
            "Recorded feedback id=%s rating=%s history_id=%s client_ip=%s",
            feedback_id,
            request.rating,
            request.history_id,
            _client_ip_from_request(http_request),
        )
        return {"id": feedback_id}

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
    async def post_chat(request: ChatRequest, http_request: Request) -> ChatResponse:
        request_started = perf_counter()
        client_ip = _client_ip_from_request(http_request)
        settings = storage.get_settings()
        enabled_collections = available_collections(settings.enabled_rag_collections)
        rag_started = perf_counter()
        if request.use_rag:
            logger.info(
                "Chat request using RAG=%s enabled_collections=%s client_ip=%s",
                request.use_rag,
                ",".join(enabled_collections) or "(none)",
                client_ip,
            )
            context = await svc.rag.retrieve(
                request.question,
                enabled_collections=enabled_collections,
            )
            if (
                "events" in enabled_collections
                and _is_events_schedule_query(request.question)
                and not any(chunk.collection == "events" for chunk in context)
            ):
                context.extend(_build_fallback_events_context(svc, limit=3))
        else:
            logger.info("Chat request with RAG disabled client_ip=%s", client_ip)
            context = []
        rag_latency_ms = (perf_counter() - rag_started) * 1000.0

        try:
            llm_result = await svc.llm.chat(
                settings=settings,
                question=request.question,
                context=context,
            )
        except LlmError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        total_latency_ms = (perf_counter() - request_started) * 1000.0

        # Metrics are a free-form JSON object returned to the UI and stored.
        metrics: dict[str, Any] = {
            "total_latency_ms": round(total_latency_ms, 2),
            "rag_latency_ms": round(rag_latency_ms, 2),
            "llm_latency_ms": round(llm_result.llm_latency_ms, 2),
            "prompt_tokens": llm_result.input_tokens,
            "completion_tokens": llm_result.output_tokens,
            "total_tokens": llm_result.total_tokens,
            "tokens_per_second": (
                round(llm_result.tokens_per_second, 2)
                if isinstance(llm_result.tokens_per_second, int | float)
                else None
            ),
            "context_chunks_used": len(context),
            "temporary_chat": request.temporary_chat,
            "history_saved": not request.temporary_chat,
            "model": llm_result.model,
            "provider_metrics": llm_result.provider_metrics,
        }

        history_id: int | None = None
        if request.temporary_chat:
            logger.info("Temporary chat mode enabled; skipping history persistence")
        else:
            history_id = storage.append_history(
                provider="openai",
                model=llm_result.model,
                system_prompt=settings.system_prompt,
                question=request.question,
                answer=llm_result.answer,
                rag_collections=[chunk.collection for chunk in context],
                rag_hits=len(context),
                config_snapshot={**settings.model_dump(), "last_metrics": metrics},
            )
        return ChatResponse(
            answer=llm_result.answer,
            context=context,
            metrics=metrics,
            history_id=history_id or None,
        )

    return app
