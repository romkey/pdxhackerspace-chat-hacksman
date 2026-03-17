from __future__ import annotations

import os
from dataclasses import dataclass


def _split_collections(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(slots=True)
class AppConfig:
    db_path: str
    app_host: str
    app_port: int
    topics_url: str
    topics_cache_ttl_seconds: int
    qdrant_url: str
    qdrant_api_key: str | None
    rag_collections: list[str]
    rag_top_k: int
    rag_top_k_events: int
    rag_top_k_slack: int
    rag_top_k_calibre: int
    rag_min_score: float
    embedding_url: str
    embedding_model: str
    embedding_context_length: int
    embedding_timeout_seconds: float
    llm_timeout_seconds: float
    llm_retry_attempts: int
    llm_retry_backoff_seconds: float
    default_provider: str
    default_llm_base_url: str
    default_model: str
    default_system_prompt: str
    basic_auth_username: str | None
    basic_auth_password: str | None
    sentry_dsn: str | None
    log_level: str
    repo_url: str


def load_config() -> AppConfig:
    collections = _split_collections(os.getenv("RAG_COLLECTIONS", ""))
    for key in (
        "RAG_COLLECTION_1",
        "RAG_COLLECTION_2",
        "RAG_COLLECTION_3",
        "RAG_COLLECTION_4",
        "RAG_COLLECTION_5",
    ):
        value = os.getenv(key, "").strip()
        if value:
            collections.append(value)

    # Keep order but remove duplicates.
    deduped_collections = list(dict.fromkeys(collections))

    return AppConfig(
        db_path=os.getenv("CHAT_HACKSMAN_DB_PATH", "./data/chat_hacksman.db"),
        app_host=os.getenv("APP_HOST", "0.0.0.0"),
        app_port=int(os.getenv("APP_PORT", "8000")),
        topics_url=os.getenv("RAG_TOPICS_URL", "https://members.pdxhackerspace.org/rag.json"),
        topics_cache_ttl_seconds=int(os.getenv("TOPICS_CACHE_TTL_SECONDS", "300")),
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY") or None,
        rag_collections=deduped_collections,
        rag_top_k=int(os.getenv("RAG_TOP_K", "4")),
        rag_top_k_events=int(os.getenv("RAG_TOP_K_EVENTS", "3")),
        rag_top_k_slack=int(os.getenv("RAG_TOP_K_SLACK", "3")),
        rag_top_k_calibre=int(os.getenv("RAG_TOP_K_CALIBRE", "2")),
        rag_min_score=float(os.getenv("RAG_MIN_SCORE", "0.5")),
        embedding_url=os.getenv("EMBEDDING_URL", "http://localhost:11434"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
        embedding_context_length=int(os.getenv("EMBEDDING_CONTEXT_LENGTH", "8192")),
        embedding_timeout_seconds=float(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "30")),
        llm_timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "180")),
        llm_retry_attempts=int(os.getenv("LLM_RETRY_ATTEMPTS", "2")),
        llm_retry_backoff_seconds=float(os.getenv("LLM_RETRY_BACKOFF_SECONDS", "0.5")),
        default_provider=os.getenv("DEFAULT_PROVIDER", "ollama"),
        default_llm_base_url=os.getenv("DEFAULT_LLM_BASE_URL", "http://localhost:11434"),
        default_model=os.getenv("DEFAULT_MODEL", "llama3.2:latest"),
        default_system_prompt=os.getenv(
            "DEFAULT_SYSTEM_PROMPT",
            "You are Chat Hacksman, an expert assistant for hackerspace members.",
        ),
        basic_auth_username=os.getenv("BASIC_AUTH_USERNAME") or None,
        basic_auth_password=os.getenv("BASIC_AUTH_PASSWORD") or None,
        sentry_dsn=os.getenv("SENTRY_DSN") or None,
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        repo_url=os.getenv(
            "REPO_URL",
            "https://github.com/romkey/pdxhackerspace-chat-hacksman",
        ),
    )
