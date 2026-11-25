"""The FastAPI app: chat, settings, history, feedback, topics and model endpoints.

create_app() builds it from Settings; nothing here reads the environment.
"""

import hashlib
import json
import logging
import re
import socket
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import perf_counter

# Any: feed payloads and chat metrics are untyped JSON.
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from chat_hacksman import __version__
from chat_hacksman.auth import (
    SESSION_COOKIE,
    OidcClient,
    OidcError,
    hash_token,
    local_login_valid,
    login_page,
    new_session_token,
    safe_next,
)
from chat_hacksman.config import Settings
from chat_hacksman.llm import LlmError, LlmResult, LlmService, StreamState, citations
from chat_hacksman.models import (
    AppSettings,
    ChatRequest,
    ChatResponse,
    ChatTurn,
    ContextChunk,
    EventRecord,
    FeedbackCreateRequest,
    FeedbackRecord,
    HistoryRecord,
    LlmStatusResponse,
    MetaResponse,
    ModelsResponse,
    NegativeFeedbackItem,
    NegativeFeedbackReport,
    OccurrenceRecord,
    RagCollectionsResponse,
    SettingsUpdate,
    TimingStatsResponse,
    UnansweredGroup,
    UnansweredQuestion,
    UnansweredReport,
    WhoAmIResponse,
)
from chat_hacksman.rag import RagService
from chat_hacksman.rerank import Reranker
from chat_hacksman.storage import Storage
from chat_hacksman.timing import EVENTS, FINISH, LLM, REWRITE, TTFT, StageTimer
from chat_hacksman.topics import TopicsFetchError, TopicsService

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
# Reachable without logging in.
PUBLIC_PATHS = frozenset(
    {"/health", "/healthz", "/login", "/logout", "/auth/oidc/login", "/auth/oidc/callback"}
)


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
            candidates=settings.rag_candidates,
            top_n=settings.rag_top_n,
            min_score=settings.rag_min_score,
            dense_vector=settings.rag_dense_vector,
            sparse_vector=settings.rag_sparse_vector,
            sparse_model=settings.rag_sparse_model,
            allowed_tiers=settings.rag_allowed_tiers,
            authority_weight=settings.rag_authority_weight,
            embedding_timeout_seconds=settings.embedding_timeout_seconds,
            embedding_api_key=settings.embedding_api_key,
            embedding_query_prefix=settings.embedding_query_prefix,
            reranker=build_reranker(settings),
            rerank_min_score=settings.rerank_min_score,
        ),
        llm=LlmService(
            base_url=settings.llm_url,
            api_key=settings.llm_api_key,
            default_model=settings.llm_model,
            request_timeout_seconds=settings.llm_timeout_seconds,
            retry_attempts=settings.llm_retry_attempts,
            retry_backoff_seconds=settings.llm_retry_backoff_seconds,
            extra_body=settings.llm_extra_body,
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


def _is_https(request: Request) -> bool:
    forwarded = request.headers.get("x-forwarded-proto", "").split(",", 1)[0].strip().lower()
    return request.url.scheme == "https" or forwarded == "https"


def _external_url(request: Request, url: str) -> str:
    """url as the browser sees it: https when a TLS proxy says so."""
    if _is_https(request) and url.startswith("http://"):
        return "https://" + url.removeprefix("http://")
    return url


def asset_version() -> str:
    """A hash of the UI's scripts, so a changed script is never served from cache."""
    digest = hashlib.sha256()
    for path in sorted(STATIC_DIR.glob("*.js")):
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def _index_html() -> str:
    page = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return re.sub(r'(\.js)\?v=[^"]*"', rf'\1?v={asset_version()}"', page)


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
                collection="events feed",
                score=1.0,
                text=text,
                metadata={
                    "source": "event",
                    "title": title,
                    "tier": "member",
                    "authority": 3,
                    "updated_at": datetime.now(tz=UTC).isoformat(timespec="minutes"),
                    "start_time": payload["occurs_at"],
                    "duration": payload.get("duration"),
                },
            )
        )
    return out


@dataclass(slots=True)
class Prepared:
    """A question made ready to answer: what was searched and found."""

    search_query: str
    history: list[ChatTurn]
    context: list[ContextChunk]
    rewrite_latency_ms: float
    rag_latency_ms: float
    timer: StageTimer = field(default_factory=StageTimer)


def _doc_ids(context: list[ContextChunk]) -> list[str]:
    ids = (
        f"{chunk.metadata.get('source', chunk.collection)}:{chunk.metadata['doc_id']}"
        for chunk in context
        if isinstance(chunk.metadata.get("doc_id"), str | int)
    )
    return list(dict.fromkeys(ids))


def _topics(context: list[ContextChunk]) -> list[str]:
    topics: dict[str, None] = {}
    for chunk in context:
        values = chunk.metadata.get("topics")
        if isinstance(values, list):
            topics.update((str(value), None) for value in values if str(value).strip())
    return list(topics)


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
    oidc = (
        OidcClient(
            issuer=config.oidc_issuer,
            client_id=config.oidc_client_id,
            client_secret=config.oidc_client_secret,
            scopes=config.oidc_scopes,
        )
        if config.oidc_issuer and config.oidc_client_id
        else None
    )
    if not config.login_required:
        logger.warning(
            "No login is configured, so anyone who can reach the app can use it; set "
            "LOCAL_LOGIN_USERNAME and LOCAL_LOGIN_PASSWORD, or OIDC_ISSUER and OIDC_CLIENT_ID"
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

    def current_user(request: Request) -> str | None:
        token = request.cookies.get(SESSION_COOKIE)
        return storage.get_login_session(hash_token(token)) if token else None

    def render_login(
        next_path: str, error: str | None = None, status_code: int = 200, local_open: bool = False
    ) -> HTMLResponse:
        page = login_page(
            oidc_label=config.oidc_login_label if oidc else None,
            local_enabled=config.local_login_enabled,
            next_path=next_path,
            error=error,
            local_open=local_open,
        )
        return HTMLResponse(page, status_code=status_code)

    def logged_in(request: Request, username: str, next_path: str) -> Response:
        """A redirect to next_path that sets a new session cookie."""
        token = new_session_token()
        max_age = config.session_days * 86400
        storage.create_login_session(
            hash_token(token), username, datetime.now(UTC) + timedelta(seconds=max_age)
        )
        response = RedirectResponse(next_path, status_code=303)
        response.set_cookie(
            SESSION_COOKIE,
            token,
            max_age=max_age,
            httponly=True,
            samesite="lax",
            secure=_is_https(request),
        )
        return response

    @app.middleware("http")
    async def require_login(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if not config.login_required:
            return await call_next(request)
        if request.url.path in PUBLIC_PATHS or request.method == "OPTIONS":
            return await call_next(request)
        if current_user(request) is not None:
            return await call_next(request)
        if request.url.path.startswith("/api/"):
            return JSONResponse({"detail": "Not logged in"}, status_code=401)
        next_path = request.url.path + (f"?{request.url.query}" if request.url.query else "")
        return RedirectResponse("/login?" + urlencode({"next": next_path}), status_code=303)

    @app.get("/login", response_model=None)
    async def get_login(request: Request, next: str | None = None) -> Response:
        next_path = safe_next(next)
        if not config.login_required or current_user(request) is not None:
            return RedirectResponse(next_path, status_code=303)
        return render_login(next_path)

    @app.post("/login", response_model=None)
    async def post_login(request: Request) -> Response:
        form = parse_qs((await request.body()).decode("utf-8", errors="replace"))
        next_path = safe_next(form.get("next", ["/"])[0])
        username = form.get("username", [""])[0]
        password = form.get("password", [""])[0]
        if not config.local_login_enabled:
            return render_login(next_path, "Local login is turned off.", 403)
        if not local_login_valid(
            config.local_login_username, config.local_login_password, username, password
        ):
            logger.warning("Failed local login client_ip=%s", _client_ip_from_request(request))
            return render_login(next_path, "Wrong username or password.", 401, local_open=True)
        logger.info("Local login succeeded")
        return logged_in(request, username, next_path)

    @app.get("/auth/oidc/login", response_model=None)
    async def oidc_login(request: Request, next: str | None = None) -> Response:
        next_path = safe_next(next)
        if oidc is None:
            return render_login(next_path, "Single sign-on is turned off.", 404)
        redirect_uri = config.oidc_redirect_uri or _external_url(
            request, str(request.url_for("oidc_callback"))
        )
        try:
            url = await oidc.authorization_url(redirect_uri, next_path)
        except OidcError as exc:
            logger.warning("OIDC login failed to start: %s", exc)
            return render_login(next_path, "Single sign-on is unavailable right now.", 502)
        return RedirectResponse(url, status_code=303)

    @app.get("/auth/oidc/callback", response_model=None)
    async def oidc_callback(
        request: Request,
        state: str = "",
        code: str = "",
        error: str | None = None,
        error_description: str | None = None,
    ) -> Response:
        if oidc is None:
            return render_login("/", "Single sign-on is turned off.", 404)
        if error:
            logger.warning("OIDC provider refused the login: %s %s", error, error_description)
            return render_login("/", "The single sign-on provider refused the login.", 401)
        try:
            username, next_path = await oidc.finish(state, code)
        except OidcError as exc:
            logger.warning("OIDC login failed: %s", exc)
            # The reason goes to the log only, not to the browser.
            return render_login("/", "Single sign-on failed; try again.", 401)
        logger.info("OIDC login succeeded")
        return logged_in(request, username, next_path)

    @app.post("/logout", response_model=None)
    async def logout(request: Request) -> Response:
        token = request.cookies.get(SESSION_COOKIE)
        if token:
            storage.delete_login_session(hash_token(token))
        response = RedirectResponse("/login" if config.login_required else "/", status_code=303)
        response.delete_cookie(SESSION_COOKIE, httponly=True, samesite="lax")
        return response

    @app.get("/api/whoami", response_model=WhoAmIResponse)
    async def whoami(request: Request) -> WhoAmIResponse:
        return WhoAmIResponse(login_required=config.login_required, username=current_user(request))

    index_html = _index_html()

    @app.get("/")
    async def root() -> HTMLResponse:
        return HTMLResponse(index_html)

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
    async def post_feedback(request: FeedbackCreateRequest) -> dict[str, int]:
        feedback_id = storage.append_feedback(request)
        logger.info(
            "Recorded feedback id=%s rating=%s history_id=%s",
            feedback_id,
            request.rating,
            request.history_id,
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

    @app.get("/api/timings", response_model=TimingStatsResponse)
    async def get_timings() -> TimingStatsResponse:
        """Average time per chat stage since the tally began (or was reset)."""
        return storage.get_timing_stats()

    @app.delete("/api/timings", response_model=TimingStatsResponse)
    async def delete_timings() -> TimingStatsResponse:
        """Start the tally again; saved chats keep their own timings."""
        storage.reset_timing_stats()
        logger.info("Reset chat timing averages")
        return storage.get_timing_stats()

    def record_timings(timer: StageTimer, error: bool = False) -> dict[str, float]:
        """This chat's timings, after adding them to the running tally."""
        report = timer.report()
        storage.record_timings(report, error=error)
        logger.info(
            "Chat timings%s %s",
            " (failed)" if error else "",
            " ".join(f"{name}={ms:.0f}ms" for name, ms in report.items()),
        )
        return report

    def failure(timer: StageTimer, detail: str) -> dict[str, Any]:
        """The body of a failed chat: why, and how long it took."""
        timings = record_timings(timer, error=True)
        return {
            "detail": detail,
            "timings": timings,
            "timing_stats": storage.get_timing_stats().model_dump(mode="json"),
        }

    @app.get("/api/reports/unanswered", response_model=UnansweredReport)
    async def get_unanswered_report(days: int = Query(default=7, ge=1, le=365)) -> UnansweredReport:
        """Questions not answered from sources, grouped by the first topic of
        what was found: the to-do list for the wiki."""
        since = datetime.now(UTC) - timedelta(days=days)
        rows = storage.get_unanswered(since=since)
        groups: dict[str, list[UnansweredQuestion]] = {}
        for history_id, created_at, question, topics in rows:
            groups.setdefault(topics[0] if topics else "unclassified", []).append(
                UnansweredQuestion(history_id=history_id, created_at=created_at, question=question)
            )
        ordered = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
        return UnansweredReport(
            since=since,
            total=len(rows),
            groups=[UnansweredGroup(topic=topic, questions=items) for topic, items in ordered],
        )

    @app.get("/api/reports/negative-feedback", response_model=NegativeFeedbackReport)
    async def get_negative_feedback_report(
        days: int = Query(default=7, ge=1, le=365),
    ) -> NegativeFeedbackReport:
        """Thumbs-down answers, with the documents they were answered from."""
        since = datetime.now(UTC) - timedelta(days=days)
        return NegativeFeedbackReport(
            since=since,
            items=[
                NegativeFeedbackItem(
                    feedback_id=feedback_id,
                    created_at=created_at,
                    note=note,
                    question=question,
                    answer=answer,
                    doc_ids=doc_ids,
                )
                for feedback_id, created_at, note, question, answer, doc_ids in (
                    storage.get_negative_feedback(since=since)
                )
            ],
        )

    async def prepare(request: ChatRequest, settings: AppSettings) -> Prepared:
        """Steps 1-4: rewrite the question with the conversation, then
        search, rerank and pick the sources."""
        timer = StageTimer()
        turns = config.chat_history_turns
        history = request.history[-turns:] if turns > 0 else []
        with timer.stage(REWRITE):
            search_query = await svc.llm.rewrite_query(
                settings=settings, question=request.question, history=history
            )
        rewrite_latency_ms = timer.stages[REWRITE]
        rag_started = perf_counter()
        enabled = available_collections(settings.enabled_rag_collections)
        context = await svc.rag.retrieve(search_query, enabled_collections=enabled, timer=timer)
        # Live facts come from the events feed, not the index.
        if _is_events_schedule_query(search_query) and not any(
            chunk.metadata.get("source") == "event" for chunk in context
        ):
            with timer.stage(EVENTS):
                context.extend(_build_fallback_events_context(svc, limit=3))
        logger.info(
            "Chat request history_turns=%d rewritten=%s sources=%d",
            len(history),
            search_query != request.question,
            len(context),
        )
        return Prepared(
            search_query=search_query,
            history=history,
            context=context,
            rewrite_latency_ms=rewrite_latency_ms,
            rag_latency_ms=(perf_counter() - rag_started) * 1000.0,
            timer=timer,
        )

    def finish(
        request: ChatRequest,
        settings: AppSettings,
        prepared: Prepared,
        answer: str,
        llm_result: LlmResult | None,
    ) -> ChatResponse:
        """Step 6 and the log: check citations, then record the chat and its
        timings. The timings stop before the database writes, so a saved
        chat holds the same timings as the response."""
        timer = prepared.timer
        if llm_result is not None and llm_result.first_token_ms is not None:
            timer.add(TTFT, llm_result.first_token_ms)
        with timer.stage(FINISH):
            response = _response(request, prepared, answer, llm_result)
        response.timings = record_timings(timer)
        response.timing_stats = storage.get_timing_stats()
        if not request.temporary_chat:
            context = prepared.context
            # No identity of any kind is stored with a chat.
            response.history_id = (
                storage.append_history(
                    provider="openai",
                    model=llm_result.model if llm_result is not None else "",
                    system_prompt=settings.system_prompt,
                    question=request.question,
                    answer=answer,
                    rag_collections=[chunk.collection for chunk in context],
                    rag_hits=len(context),
                    config_snapshot={**settings.model_dump(), "last_metrics": response.metrics},
                    doc_ids=_doc_ids(context),
                    topics=_topics(context),
                    answered=response.answered,
                    timings=response.timings,
                )
                or None
            )
        return response

    def _response(
        request: ChatRequest,
        prepared: Prepared,
        answer: str,
        llm_result: LlmResult | None,
    ) -> ChatResponse:
        context = prepared.context
        cited = citations(answer, len(context))
        answered = bool(context) and bool(cited)
        metrics: dict[str, Any] = {
            "total_latency_ms": round(prepared.timer.elapsed_ms(), 2),
            "rewrite_latency_ms": round(prepared.rewrite_latency_ms, 2),
            "rag_latency_ms": round(prepared.rag_latency_ms, 2),
            "context_chunks_used": len(context),
            "temporary_chat": request.temporary_chat,
            "history_saved": not request.temporary_chat,
            "answered": answered,
        }
        if llm_result is not None:
            metrics.update(
                {
                    "llm_latency_ms": round(llm_result.llm_latency_ms, 2),
                    "first_token_ms": (
                        round(llm_result.first_token_ms, 2)
                        if llm_result.first_token_ms is not None
                        else None
                    ),
                    "prompt_tokens": llm_result.input_tokens,
                    "completion_tokens": llm_result.output_tokens,
                    "total_tokens": llm_result.total_tokens,
                    "tokens_per_second": (
                        round(llm_result.tokens_per_second, 2)
                        if llm_result.tokens_per_second is not None
                        else None
                    ),
                    "model": llm_result.model,
                    "provider_metrics": llm_result.provider_metrics,
                }
            )
        logger.info("Chat answered=%s sources=%d cited=%d", answered, len(context), len(cited))
        return ChatResponse(
            answer=answer,
            search_query=prepared.search_query,
            context=context,
            citations=cited,
            answered=answered,
            metrics=metrics,
        )

    async def stream_chat(
        request: ChatRequest, settings: AppSettings, prepared: Prepared
    ) -> AsyncIterator[bytes]:
        def line(event: dict[str, Any]) -> bytes:
            return (json.dumps(event) + "\n").encode()

        state = StreamState()
        try:
            with prepared.timer.stage(LLM):
                async for piece in svc.llm.chat_stream(
                    settings=settings,
                    question=request.question,
                    context=prepared.context,
                    history=prepared.history,
                    state=state,
                ):
                    yield line({"type": "delta", "text": piece})
        except LlmError as exc:
            logger.warning("Streamed chat failed: %s", exc)
            yield line({"type": "error", **failure(prepared.timer, str(exc))})
            return
        response = finish(request, settings, prepared, "".join(state.parts), state.result)
        yield line({"type": "done", **response.model_dump(mode="json")})

    @app.post("/api/chat", response_model=ChatResponse)
    async def post_chat(request: ChatRequest) -> Response:
        settings = storage.get_settings()
        prepared = await prepare(request, settings)
        if not prepared.context:
            # Nothing to answer from: say so and give next steps, without the model.
            response = finish(request, settings, prepared, config.no_answer_message, None)
            if request.stream:
                done = {"type": "done", **response.model_dump(mode="json")}
                return StreamingResponse(
                    iter([json.dumps(done) + "\n"]), media_type="application/x-ndjson"
                )
            return Response(response.model_dump_json(), media_type="application/json")
        if request.stream:
            return StreamingResponse(
                stream_chat(request, settings, prepared), media_type="application/x-ndjson"
            )
        try:
            with prepared.timer.stage(LLM):
                llm_result = await svc.llm.chat(
                    settings=settings,
                    question=request.question,
                    context=prepared.context,
                    history=prepared.history,
                )
        except LlmError as exc:
            logger.warning("Chat failed: %s", exc)
            return JSONResponse(failure(prepared.timer, str(exc)), status_code=502)
        response = finish(request, settings, prepared, llm_result.answer, llm_result)
        return Response(response.model_dump_json(), media_type="application/json")

    return app
