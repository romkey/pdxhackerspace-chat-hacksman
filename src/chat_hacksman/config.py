"""Settings, read once at startup from environment variables.

Every variable is documented in docs/configuration.md and listed, with a
safe example value, in .env.example.
"""

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from chat_hacksman.rerank import RERANK_APIS, RerankApi

# Variables from before the chat API became OpenAI-compatible, and what
# replaced them. Setting one stops the app, so a stale .env can't silently
# point the chat at the wrong server.
REMOVED_VARIABLES: dict[str, str] = {
    "DEFAULT_PROVIDER": "LLM_URL (any OpenAI-compatible server)",
    "DEFAULT_LLM_BASE_URL": "LLM_URL, which includes /v1",
    "DEFAULT_MODEL": "LLM_MODEL",
    "OLLAMA_API_KEY": "LLM_API_KEY",
    # 0.9.0: one retrieval pipeline over the shared hs_* collections, which
    # are no longer treated differently by name.
    "RAG_COLLECTION_1": "RAG_COLLECTIONS",
    "RAG_COLLECTION_2": "RAG_COLLECTIONS",
    "RAG_COLLECTION_3": "RAG_COLLECTIONS",
    "RAG_COLLECTION_4": "RAG_COLLECTIONS",
    "RAG_COLLECTION_5": "RAG_COLLECTIONS",
    "RAG_TOP_K": "RAG_CANDIDATES and RAG_TOP_N",
    "RAG_TOP_K_EVENTS": "RAG_CANDIDATES and RAG_TOP_N",
    "RAG_TOP_K_SLACK": "RAG_CANDIDATES and RAG_TOP_N",
    "RAG_TOP_K_CALIBRE": "RAG_CANDIDATES and RAG_TOP_N",
    "RERANK_CANDIDATES": "RAG_CANDIDATES",
    "RERANK_TOP_N": "RAG_TOP_N",
}

DEFAULT_COLLECTIONS: tuple[str, ...] = ("hs_member", "hs_catalog")

# The member bot's own instructions; the answering rules shared by both bots
# are fixed in llm.ANSWERING_RULES and always come first.
DEFAULT_SYSTEM_PROMPT = (
    "You are Chat Hacksman, the PDX Hackerspace assistant for logged-in members. "
    "Help members find resources, policies and procedures, and explain how the "
    "space's own devices are built.\n"
    "- Entering and leaving: explain fob use at the main entrance and the back gate, "
    "what the camera display is for, and how to read the exit dashboard. Always add "
    "that the dashboard's sensors can be wrong, so members should check doors "
    "physically before leaving.\n"
    '- "Where is" questions: give the location and the map link from the context.\n'
    "- Library questions: answer from the catalog entries, merging ebook, audiobook "
    "and hardcopy formats of the same work into one answer.\n"
    "- When you can't answer, suggest the Slack channel or wiki page the context "
    "points to, or asking at the open house."
)

DEFAULT_NO_ANSWER_MESSAGE = (
    "I couldn't find that in anything I can search: the wiki, policies, signs, "
    "manuals or the library catalog. Try searching the wiki, asking in the space's "
    "Slack, or asking at the next open house. Infrastructure admins handle anything "
    "about changing or reconfiguring the space's systems."
)

DEFAULT_EMBEDDING_MODEL = "nomic-ai/modernbert-embed-base"
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

# Query prefixes for the embedding models we know, matched by name (case
# ignored, with or without the "org/" part). Taken from each model card on
# Hugging Face; documents are embedded by the ingester, so only the query
# side is applied here. See the table in .env.example.
_NOMIC_QUERY_PREFIX = "search_query: "
_E5_QUERY_PREFIX = "query: "
KNOWN_QUERY_PREFIXES: dict[str, str] = {
    "nomic-ai/modernbert-embed-base": _NOMIC_QUERY_PREFIX,
    "nomic-ai/nomic-embed-text-v1.5": _NOMIC_QUERY_PREFIX,
    "nomic-ai/nomic-embed-text-v1": _NOMIC_QUERY_PREFIX,
    "intfloat/e5-large-v2": _E5_QUERY_PREFIX,
    "intfloat/multilingual-e5-large": _E5_QUERY_PREFIX,
    "mixedbread-ai/mxbai-embed-large-v1": (
        "Represent this sentence for searching relevant passages: "
    ),
    "qwen/qwen3-embedding-0.6b": (
        "Instruct: Given a web search query, retrieve relevant passages that answer the query"
        "\nQuery:"
    ),
    "openai/text-embedding-3-small": "",
    "openai/text-embedding-3-large": "",
}


class ConfigError(ValueError):
    """An environment variable is set to something unusable."""


def known_query_prefix(model: str) -> str:
    """The query prefix for a model in KNOWN_QUERY_PREFIXES, or "" if unknown."""
    name = model.strip().lower()
    # Tags such as ":latest" and GGUF-style suffixes don't change the prefix.
    name = name.split(":", 1)[0]
    if name in KNOWN_QUERY_PREFIXES:
        return KNOWN_QUERY_PREFIXES[name]
    short = name.rsplit("/", 1)[-1]
    for known, prefix in KNOWN_QUERY_PREFIXES.items():
        if known.rsplit("/", 1)[-1] == short:
            return prefix
    return ""


def parse_prefix(value: str) -> str:
    """An EMBEDDING_QUERY_PREFIX value as text to put in front of a query.

    "none" means no prefix, a literal backslash-n is a newline, and a space
    is added if the prefix doesn't already end in whitespace.
    """
    if value.strip().lower() == "none":
        return ""
    prefix = value.replace("\\n", "\n")
    if prefix and not prefix[-1].isspace():
        prefix += " "
    return prefix


def _int(env: Mapping[str, str], name: str, default: int, minimum: int = 0) -> int:
    text = env.get(name, "").strip()
    if not text:
        return default
    try:
        value = int(text)
    except ValueError:
        raise ConfigError(f"{name} must be a whole number, not {text!r}") from None
    if value < minimum:
        raise ConfigError(f"{name} must be at least {minimum}, not {value}")
    return value


def _float(env: Mapping[str, str], name: str, default: float, minimum: float = 0.0) -> float:
    text = env.get(name, "").strip()
    if not text:
        return default
    try:
        value = float(text)
    except ValueError:
        raise ConfigError(f"{name} must be a number, not {text!r}") from None
    if value < minimum:
        raise ConfigError(f"{name} must be at least {minimum}, not {value}")
    return value


def _optional(env: Mapping[str, str], name: str) -> str | None:
    return env.get(name, "").strip() or None


def _url(env: Mapping[str, str], name: str, default: str) -> str:
    value = env.get(name, "").strip() or default
    if not value.startswith(("http://", "https://")):
        raise ConfigError(f"{name} must be an http:// or https:// URL, not {value!r}")
    return value.rstrip("/")


def _optional_url(env: Mapping[str, str], name: str) -> str | None:
    return _url(env, name, "") if env.get(name, "").strip() else None


def _name(env: Mapping[str, str], name: str, default: str) -> str:
    """A vector name: unset is the default, "none" is empty (no name)."""
    value = env.get(name, "").strip()
    if not value:
        return default
    return "" if value.lower() == "none" else value


def _optional_float(env: Mapping[str, str], name: str) -> float | None:
    return _float(env, name, 0.0, minimum=float("-inf")) if env.get(name, "").strip() else None


def _rerank_api(env: Mapping[str, str]) -> RerankApi:
    value = env.get("RERANK_API", "").strip().lower() or "jina"
    for api in RERANK_APIS:
        if value == api:
            return api
    raise ConfigError(f"RERANK_API must be one of {', '.join(RERANK_APIS)}, not {value!r}")


def _split(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _list(env: Mapping[str, str], name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    """A comma-separated list, in order without duplicates; unset or empty is the default."""
    items = _split(env.get(name, ""))
    return tuple(dict.fromkeys(items)) if items else default


@dataclass(frozen=True)
class Settings:
    db_path: Path = Path("data/chat_hacksman.db")
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"
    repo_url: str = "https://github.com/romkey/pdxhackerspace-chat-hacksman"
    topics_url: str = "https://members.pdxhackerspace.org/rag.json"
    topics_cache_ttl_seconds: int = 300
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    rag_collections: tuple[str, ...] = DEFAULT_COLLECTIONS
    # Hits per collection from the (hybrid) search, before reranking.
    rag_candidates: int = 50
    # Chunks given to the model, after reranking and the authority pass.
    rag_top_n: int = 8
    # Only applied to dense-only search; fused hybrid scores are rank-based.
    rag_min_score: float = 0.5
    # Named vectors in each collection. Empty dense name: the unnamed vector.
    # Empty sparse name: dense-only search.
    rag_dense_vector: str = "dense"
    rag_sparse_vector: str = "sparse"
    rag_sparse_model: str = "qdrant/bm25"
    # Points with any other tier are never used, whatever collection they're in.
    rag_allowed_tiers: tuple[str, ...] = ("public", "member")
    # How much authority (1-5) can lift a chunk whose relevance is close to
    # another's, on relevance scaled to 0-1 over the candidates.
    rag_authority_weight: float = 0.1
    # Earlier turns of the conversation sent with each question.
    chat_history_turns: int = 6
    no_answer_message: str = DEFAULT_NO_ANSWER_MESSAGE
    embedding_url: str = "http://localhost:8080/v1"
    embedding_api_key: str | None = None
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_query_prefix: str = _NOMIC_QUERY_PREFIX
    embedding_context_length: int = 8192
    embedding_timeout_seconds: float = 30.0
    rerank_url: str | None = None
    rerank_api: RerankApi = "jina"
    rerank_api_key: str | None = None
    rerank_model: str = DEFAULT_RERANK_MODEL
    rerank_min_score: float | None = None
    rerank_context_length: int = 8192
    rerank_timeout_seconds: float = 30.0
    llm_timeout_seconds: float = 180.0
    llm_retry_attempts: int = 2
    llm_retry_backoff_seconds: float = 0.5
    llm_url: str = "http://localhost:8080/v1"
    llm_api_key: str | None = None
    # Empty: the first model the server lists.
    llm_model: str = ""
    default_system_prompt: str = DEFAULT_SYSTEM_PROMPT
    basic_auth_username: str | None = None
    basic_auth_password: str | None = None
    sentry_dsn: str | None = None

    @property
    def basic_auth_enabled(self) -> bool:
        return bool(self.basic_auth_username and self.basic_auth_password)

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> Settings:
        port = _int(env, "APP_PORT", cls.port, minimum=1)
        if port > 65535:
            raise ConfigError(f"APP_PORT must be 1-65535, not {port}")

        log_level = env.get("LOG_LEVEL", cls.log_level).strip().upper() or cls.log_level
        if log_level not in logging.getLevelNamesMapping():
            raise ConfigError(f"LOG_LEVEL {log_level!r} is not a logging level")

        for name, replacement in REMOVED_VARIABLES.items():
            if env.get(name, "").strip():
                raise ConfigError(f"{name} is no longer used; set {replacement} instead")

        tiers = _list(env, "RAG_ALLOWED_TIERS", cls.rag_allowed_tiers)
        if "admin" in (tier.lower() for tier in tiers):
            raise ConfigError("RAG_ALLOWED_TIERS must never include admin")

        embedding_model = env.get("EMBEDDING_MODEL", "").strip() or cls.embedding_model
        prefix_text = env.get("EMBEDDING_QUERY_PREFIX")
        query_prefix = (
            known_query_prefix(embedding_model)
            if prefix_text is None or prefix_text == ""
            else parse_prefix(prefix_text)
        )

        return cls(
            db_path=Path(env.get("CHAT_HACKSMAN_DB_PATH", "").strip() or cls.db_path),
            host=env.get("APP_HOST", "").strip() or cls.host,
            port=port,
            log_level=log_level,
            repo_url=env.get("REPO_URL", "").strip() or cls.repo_url,
            topics_url=_url(env, "RAG_TOPICS_URL", cls.topics_url),
            topics_cache_ttl_seconds=_int(
                env, "TOPICS_CACHE_TTL_SECONDS", cls.topics_cache_ttl_seconds
            ),
            qdrant_url=_url(env, "QDRANT_URL", cls.qdrant_url),
            qdrant_api_key=_optional(env, "QDRANT_API_KEY"),
            rag_collections=_list(env, "RAG_COLLECTIONS", cls.rag_collections),
            rag_candidates=_int(env, "RAG_CANDIDATES", cls.rag_candidates, minimum=1),
            rag_top_n=_int(env, "RAG_TOP_N", cls.rag_top_n, minimum=1),
            rag_min_score=_float(env, "RAG_MIN_SCORE", cls.rag_min_score, minimum=-1.0),
            rag_dense_vector=_name(env, "RAG_DENSE_VECTOR", cls.rag_dense_vector),
            rag_sparse_vector=_name(env, "RAG_SPARSE_VECTOR", cls.rag_sparse_vector),
            rag_sparse_model=env.get("RAG_SPARSE_MODEL", "").strip() or cls.rag_sparse_model,
            rag_allowed_tiers=tiers,
            rag_authority_weight=_float(
                env, "RAG_AUTHORITY_WEIGHT", cls.rag_authority_weight, minimum=0.0
            ),
            chat_history_turns=_int(env, "CHAT_HISTORY_TURNS", cls.chat_history_turns),
            no_answer_message=(env.get("NO_ANSWER_MESSAGE", "").strip() or cls.no_answer_message),
            embedding_url=_url(env, "EMBEDDING_URL", cls.embedding_url),
            embedding_api_key=_optional(env, "EMBEDDING_API_KEY"),
            embedding_model=embedding_model,
            embedding_query_prefix=query_prefix,
            embedding_context_length=_int(
                env, "EMBEDDING_CONTEXT_LENGTH", cls.embedding_context_length, minimum=1
            ),
            embedding_timeout_seconds=_float(
                env, "EMBEDDING_TIMEOUT_SECONDS", cls.embedding_timeout_seconds
            ),
            rerank_url=_optional_url(env, "RERANK_URL"),
            rerank_api=_rerank_api(env),
            rerank_api_key=_optional(env, "RERANK_API_KEY"),
            rerank_model=env.get("RERANK_MODEL", "").strip() or cls.rerank_model,
            rerank_min_score=_optional_float(env, "RERANK_MIN_SCORE"),
            rerank_context_length=_int(
                env, "RERANK_CONTEXT_LENGTH", cls.rerank_context_length, minimum=1
            ),
            rerank_timeout_seconds=_float(
                env, "RERANK_TIMEOUT_SECONDS", cls.rerank_timeout_seconds
            ),
            llm_timeout_seconds=_float(env, "LLM_TIMEOUT_SECONDS", cls.llm_timeout_seconds),
            llm_retry_attempts=_int(env, "LLM_RETRY_ATTEMPTS", cls.llm_retry_attempts),
            llm_retry_backoff_seconds=_float(
                env, "LLM_RETRY_BACKOFF_SECONDS", cls.llm_retry_backoff_seconds
            ),
            llm_url=_url(env, "LLM_URL", cls.llm_url),
            llm_api_key=_optional(env, "LLM_API_KEY"),
            llm_model=env.get("LLM_MODEL", "").strip(),
            default_system_prompt=(
                env.get("DEFAULT_SYSTEM_PROMPT", "").strip() or cls.default_system_prompt
            ),
            basic_auth_username=_optional(env, "BASIC_AUTH_USERNAME"),
            basic_auth_password=_optional(env, "BASIC_AUTH_PASSWORD"),
            sentry_dsn=_optional(env, "SENTRY_DSN"),
        )
