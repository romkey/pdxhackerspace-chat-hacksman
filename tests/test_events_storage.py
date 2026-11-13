import json
import sqlite3
from pathlib import Path

from chat_hacksman.models import AppSettings
from chat_hacksman.storage import Storage


def test_storage_upsert_and_get_events(tmp_path: Path) -> None:
    db_path = str(tmp_path / "events.db")
    settings = AppSettings(
        system_prompt="test",
    )
    storage = Storage(db_path=db_path, default_settings=settings)
    events = [
        {
            "id": 10,
            "slug": "dorkbot",
            "title": "Dorkbot",
            "description": "People doing strange things with electricity.",
            "more_info_url": "https://dorkbotpdx.org",
            "visibility": "public",
            "open_to": "public",
            "recurrence_type": "monthly",
            "start_time": "2026-01-01T00:00:00-08:00",
            "duration": 180,
        }
    ]
    assert storage.upsert_events(events) == 1
    loaded = storage.get_events(limit=10)
    assert len(loaded) == 1
    assert loaded[0].event_id == 10
    assert loaded[0].title == "Dorkbot"


def test_storage_upsert_and_get_occurrences(tmp_path: Path) -> None:
    db_path = str(tmp_path / "occurrences.db")
    settings = AppSettings(
        system_prompt="test",
    )
    storage = Storage(db_path=db_path, default_settings=settings)
    occurrences = [
        {
            "id": 101,
            "slug": "dorkbot-2026-03-09",
            "occurs_at": "2026-03-09T19:30:00-07:00",
            "occurs_at_unix": 1773109800,
            "ends_at_unix": 1773120600,
            "duration": 180,
            "is_cancelled": False,
            "is_postponed": False,
            "in_progress": False,
            "postponed_until": None,
            "open_to": "public",
            "event": {"id": 16, "title": "Dorkbot"},
        }
    ]
    assert storage.upsert_occurrences(occurrences) == 1
    loaded = storage.get_occurrences(limit=10)
    assert len(loaded) == 1
    assert loaded[0].occurrence_id == 101
    assert loaded[0].event_id == 16


def test_storage_history_links_chat_to_prompt_history(tmp_path: Path) -> None:
    db_path = str(tmp_path / "history-prompts.db")
    settings = AppSettings(
        system_prompt="test",
    )
    storage = Storage(db_path=db_path, default_settings=settings)

    first_id = storage.append_history(
        provider="openai",
        model="test-model",
        system_prompt="Prompt A",
        question="q1",
        answer="a1",
        rag_collections=["events"],
        rag_hits=1,
        config_snapshot={},
    )
    second_id = storage.append_history(
        provider="openai",
        model="test-model",
        system_prompt="Prompt A",
        question="q2",
        answer="a2",
        rag_collections=["events"],
        rag_hits=1,
        config_snapshot={},
    )
    assert first_id > 0
    assert second_id > 0

    rows = storage.get_history(limit=10)
    assert len(rows) == 2
    assert rows[0].system_prompt == "Prompt A"
    assert rows[1].system_prompt == "Prompt A"
    assert isinstance(rows[0].prompt_id, int)
    assert isinstance(rows[1].prompt_id, int)
    assert rows[0].prompt_id == rows[1].prompt_id


def test_version_1_settings_lose_the_old_provider_fields(tmp_path: Path) -> None:
    db_path = tmp_path / "old.db"
    old_payload = {
        "provider": "ollama",
        "llm_base_url": "http://localhost:11434",
        "model": "llama3.2:latest",
        "system_prompt": "kept",
        "enabled_rag_collections": ["docs"],
        "tweaks": {"max_tokens": 100, "num_ctx": 4096, "repeat_penalty": 1.1, "seed": 3},
    }
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE app_settings (id INTEGER PRIMARY KEY, payload_json TEXT, updated_at TEXT)"
        )
        conn.execute("INSERT INTO app_settings VALUES (1, ?, 'then')", (json.dumps(old_payload),))
        conn.execute("CREATE TABLE llm_base_url_history (url TEXT PRIMARY KEY, last_used_at TEXT)")
        conn.execute("PRAGMA user_version = 1")
    conn.close()

    storage = Storage(db_path=db_path, default_settings=AppSettings(system_prompt="default"))
    settings = storage.get_settings()
    assert settings.model == ""
    assert settings.system_prompt == "kept"
    assert settings.enabled_rag_collections == ["docs"]
    assert (settings.tweaks.max_tokens, settings.tweaks.seed) == (100, 3)
    with sqlite3.connect(db_path) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        stored = json.loads(conn.execute("SELECT payload_json FROM app_settings").fetchone()[0])
    conn.close()
    assert "llm_base_url_history" not in tables
    assert version == 2
    assert "provider" not in stored
    assert "num_ctx" not in stored["tweaks"]

    # Settings saved after the migration are left alone on the next start.
    storage.save_settings(settings.model_copy(update={"model": "picked"}))
    reopened = Storage(db_path=db_path, default_settings=AppSettings(system_prompt="default"))
    assert reopened.get_settings().model == "picked"
