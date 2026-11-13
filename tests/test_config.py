from pathlib import Path

import pytest

from chat_hacksman.config import ConfigError, Settings, known_query_prefix, parse_prefix


def test_defaults() -> None:
    settings = Settings.from_env({})
    assert settings == Settings()
    assert settings.db_path == Path("data/chat_hacksman.db")
    assert settings.embedding_url == "http://localhost:8080/v1"
    assert settings.embedding_model == "nomic-ai/modernbert-embed-base"
    assert settings.embedding_query_prefix == "search_query: "
    assert settings.embedding_api_key is None
    assert not settings.basic_auth_enabled


def test_supports_rag_collection_1_to_5_after_list() -> None:
    env = {
        "RAG_COLLECTIONS": "zero, one",
        "RAG_COLLECTION_1": "one",
        "RAG_COLLECTION_2": "two",
        "RAG_COLLECTION_3": "three",
        "RAG_COLLECTION_4": "four",
        "RAG_COLLECTION_5": "five",
    }
    assert Settings.from_env(env).rag_collections == (
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
    )


def test_reads_secrets_and_optional_values() -> None:
    env = {
        "BASIC_AUTH_USERNAME": "demo",
        "BASIC_AUTH_PASSWORD": "dummy-password",
        "LLM_API_KEY": "dummy-llm-key",
        "EMBEDDING_API_KEY": "dummy-embedding-key",
        "QDRANT_API_KEY": "",
        "SENTRY_DSN": "https://public@example.ingest.sentry.io/1",
    }
    settings = Settings.from_env(env)
    assert settings.basic_auth_enabled
    assert settings.llm_api_key == "dummy-llm-key"
    assert settings.embedding_api_key == "dummy-embedding-key"
    assert settings.qdrant_api_key is None
    assert settings.sentry_dsn == "https://public@example.ingest.sentry.io/1"


def test_reads_numbers() -> None:
    env = {
        "APP_PORT": "9000",
        "RAG_TOP_K_EVENTS": "7",
        "RAG_TOP_K_SLACK": "6",
        "RAG_TOP_K_CALIBRE": "5",
        "RAG_MIN_SCORE": "0.25",
        "EMBEDDING_CONTEXT_LENGTH": "2048",
        "LLM_TIMEOUT_SECONDS": "60",
    }
    settings = Settings.from_env(env)
    assert settings.port == 9000
    assert (settings.rag_top_k_events, settings.rag_top_k_slack, settings.rag_top_k_calibre) == (
        7,
        6,
        5,
    )
    assert settings.rag_min_score == 0.25
    assert settings.embedding_context_length == 2048
    assert settings.llm_timeout_seconds == 60.0


def test_strips_trailing_slash_from_urls() -> None:
    assert Settings.from_env({"EMBEDDING_URL": "https://api.openai.com/v1/"}).embedding_url == (
        "https://api.openai.com/v1"
    )


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("APP_PORT", "eighty", "APP_PORT must be a whole number"),
        ("APP_PORT", "0", "APP_PORT must be at least 1"),
        ("APP_PORT", "65536", "APP_PORT must be 1-65535"),
        ("RAG_TOP_K", "0", "RAG_TOP_K must be at least 1"),
        ("RAG_MIN_SCORE", "high", "RAG_MIN_SCORE must be a number"),
        ("LLM_TIMEOUT_SECONDS", "-1", "LLM_TIMEOUT_SECONDS must be at least"),
        ("LOG_LEVEL", "loud", "LOG_LEVEL"),
        ("LLM_URL", "localhost:8080", "LLM_URL must be an http"),
        ("EMBEDDING_URL", "localhost:8080", "EMBEDDING_URL must be an http"),
    ],
)
def test_rejects_bad_values(name: str, value: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        Settings.from_env({name: value})


@pytest.mark.parametrize(
    ("model", "prefix"),
    [
        ("nomic-ai/modernbert-embed-base", "search_query: "),
        ("modernbert-embed-base", "search_query: "),
        ("nomic-embed-text-v1.5", "search_query: "),
        ("nomic-ai/nomic-embed-text-v1.5:latest", "search_query: "),
        ("intfloat/e5-large-v2", "query: "),
        ("intfloat/multilingual-e5-large", "query: "),
        (
            "mixedbread-ai/mxbai-embed-large-v1",
            "Represent this sentence for searching relevant passages: ",
        ),
        (
            "Qwen/Qwen3-Embedding-0.6B",
            "Instruct: Given a web search query, retrieve relevant passages that answer the query"
            "\nQuery:",
        ),
        ("text-embedding-3-small", ""),
        ("text-embedding-3-large", ""),
        ("some/unknown-model", ""),
    ],
)
def test_known_query_prefix(model: str, prefix: str) -> None:
    assert known_query_prefix(model) == prefix


def test_query_prefix_follows_model_unless_set() -> None:
    assert Settings.from_env(
        {"EMBEDDING_MODEL": "intfloat/e5-large-v2"}
    ).embedding_query_prefix == ("query: ")
    env = {"EMBEDDING_MODEL": "intfloat/e5-large-v2", "EMBEDDING_QUERY_PREFIX": "none"}
    assert Settings.from_env(env).embedding_query_prefix == ""
    env = {"EMBEDDING_MODEL": "intfloat/e5-large-v2", "EMBEDDING_QUERY_PREFIX": ""}
    assert Settings.from_env(env).embedding_query_prefix == "query: "


@pytest.mark.parametrize(
    ("value", "prefix"),
    [
        ("none", ""),
        ("NONE", ""),
        ("search_query: ", "search_query: "),
        ("search_query:", "search_query: "),
        ("Instruct: find\\nQuery:", "Instruct: find\nQuery: "),
        ("Line\\n", "Line\n"),
    ],
)
def test_parse_prefix(value: str, prefix: str) -> None:
    assert parse_prefix(value) == prefix


def test_reranking_is_off_by_default() -> None:
    settings = Settings.from_env({})
    assert settings.rerank_url is None
    assert settings.rerank_api == "jina"
    assert settings.rerank_model == "BAAI/bge-reranker-v2-m3"
    assert settings.rerank_candidates == 10
    assert settings.rerank_top_n is None
    assert settings.rerank_min_score is None


def test_reads_rerank_settings() -> None:
    env = {
        "RERANK_URL": "http://tei.local:8080/",
        "RERANK_API": "TEI",
        "RERANK_API_KEY": "dummy-rerank-key",
        "RERANK_MODEL": "Alibaba-NLP/gte-reranker-modernbert-base",
        "RERANK_CANDIDATES": "25",
        "RERANK_TOP_N": "6",
        "RERANK_MIN_SCORE": "0.1",
        "RERANK_CONTEXT_LENGTH": "512",
        "RERANK_TIMEOUT_SECONDS": "10",
    }
    settings = Settings.from_env(env)
    assert settings.rerank_url == "http://tei.local:8080"
    assert settings.rerank_api == "tei"
    assert settings.rerank_api_key == "dummy-rerank-key"
    assert settings.rerank_model == "Alibaba-NLP/gte-reranker-modernbert-base"
    assert (settings.rerank_candidates, settings.rerank_top_n) == (25, 6)
    assert settings.rerank_min_score == 0.1
    assert settings.rerank_context_length == 512
    assert settings.rerank_timeout_seconds == 10.0


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("RERANK_API", "cohere", "RERANK_API must be one of jina, tei"),
        ("RERANK_URL", "tei.local", "RERANK_URL must be an http"),
        ("RERANK_CANDIDATES", "0", "RERANK_CANDIDATES must be at least 1"),
        ("RERANK_TOP_N", "0", "RERANK_TOP_N must be at least 1"),
        ("RERANK_MIN_SCORE", "low", "RERANK_MIN_SCORE must be a number"),
    ],
)
def test_rejects_bad_rerank_values(name: str, value: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        Settings.from_env({name: value})


def test_llm_settings() -> None:
    defaults = Settings.from_env({})
    assert (defaults.llm_url, defaults.llm_api_key, defaults.llm_model) == (
        "http://localhost:8080/v1",
        None,
        "",
    )
    settings = Settings.from_env(
        {"LLM_URL": "https://api.openai.com/v1/", "LLM_MODEL": " gpt-4o-mini "}
    )
    assert settings.llm_url == "https://api.openai.com/v1"
    assert settings.llm_model == "gpt-4o-mini"


@pytest.mark.parametrize(
    ("name", "replacement"),
    [
        ("DEFAULT_PROVIDER", "LLM_URL"),
        ("DEFAULT_LLM_BASE_URL", "LLM_URL"),
        ("DEFAULT_MODEL", "LLM_MODEL"),
        ("OLLAMA_API_KEY", "LLM_API_KEY"),
    ],
)
def test_rejects_removed_variables(name: str, replacement: str) -> None:
    with pytest.raises(ConfigError, match=f"{name} is no longer used; set {replacement}"):
        Settings.from_env({name: "x"})
    # Empty counts as unset, as everywhere else.
    Settings.from_env({name: ""})
