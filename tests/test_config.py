from __future__ import annotations

from app.config import load_config


def test_load_config_supports_rag_collection_4_and_5(monkeypatch) -> None:
    monkeypatch.setenv("RAG_COLLECTIONS", "")
    monkeypatch.setenv("RAG_COLLECTION_1", "one")
    monkeypatch.setenv("RAG_COLLECTION_2", "two")
    monkeypatch.setenv("RAG_COLLECTION_3", "three")
    monkeypatch.setenv("RAG_COLLECTION_4", "four")
    monkeypatch.setenv("RAG_COLLECTION_5", "five")

    config = load_config()
    assert config.rag_collections == ["one", "two", "three", "four", "five"]


def test_load_config_reads_basic_auth_env(monkeypatch) -> None:
    monkeypatch.setenv("BASIC_AUTH_USERNAME", "demo")
    monkeypatch.setenv("BASIC_AUTH_PASSWORD", "secret")

    config = load_config()
    assert config.basic_auth_username == "demo"
    assert config.basic_auth_password == "secret"


def test_load_config_reads_collection_specific_top_k_env(monkeypatch) -> None:
    monkeypatch.setenv("RAG_TOP_K_EVENTS", "7")
    monkeypatch.setenv("RAG_TOP_K_SLACK", "6")
    monkeypatch.setenv("RAG_TOP_K_CALIBRE", "5")

    config = load_config()
    assert config.rag_top_k_events == 7
    assert config.rag_top_k_slack == 6
    assert config.rag_top_k_calibre == 5


def test_load_config_reads_embedding_context_length(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_CONTEXT_LENGTH", "2048")
    config = load_config()
    assert config.embedding_context_length == 2048


def test_load_config_reads_sentry_dsn(monkeypatch) -> None:
    monkeypatch.setenv("SENTRY_DSN", "https://public@example.ingest.sentry.io/1")
    config = load_config()
    assert config.sentry_dsn == "https://public@example.ingest.sentry.io/1"
