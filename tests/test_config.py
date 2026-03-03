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
