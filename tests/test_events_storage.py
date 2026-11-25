import json
import sqlite3
from pathlib import Path

from chat_hacksman.models import AppSettings
from chat_hacksman.storage import OLD_DEFAULT_SYSTEM_PROMPT, SCHEMA_VERSION, Storage


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
    # Schema 3 resets the old per-source collections to the default.
    assert settings.enabled_rag_collections == []
    assert (settings.tweaks.max_tokens, settings.tweaks.seed) == (100, 3)
    with sqlite3.connect(db_path) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        stored = json.loads(conn.execute("SELECT payload_json FROM app_settings").fetchone()[0])
    conn.close()
    assert "llm_base_url_history" not in tables
    assert version == SCHEMA_VERSION
    assert "provider" not in stored
    assert "num_ctx" not in stored["tweaks"]

    # Settings saved after the migration are left alone on the next start.
    storage.save_settings(settings.model_copy(update={"model": "picked"}))
    reopened = Storage(db_path=db_path, default_settings=AppSettings(system_prompt="default"))
    assert reopened.get_settings().model == "picked"


def test_version_2_drops_the_old_default_prompt_and_adds_answer_columns(tmp_path: Path) -> None:
    db_path = tmp_path / "v2.db"
    Storage(db_path=db_path, default_settings=AppSettings(system_prompt="x"))
    with sqlite3.connect(db_path) as conn:
        conn.execute("ALTER TABLE chat_history DROP COLUMN doc_ids_json")
        conn.execute("ALTER TABLE chat_history DROP COLUMN topics_json")
        conn.execute("ALTER TABLE chat_history DROP COLUMN answered")
        conn.execute("ALTER TABLE chat_history DROP COLUMN timings_json")
        payload = {"system_prompt": OLD_DEFAULT_SYSTEM_PROMPT, "enabled_rag_collections": ["slack"]}
        conn.execute("INSERT INTO app_settings VALUES (1, ?, 'then')", (json.dumps(payload),))
        conn.execute("PRAGMA user_version = 2")
    conn.close()

    storage = Storage(db_path=db_path, default_settings=AppSettings(system_prompt="member bot"))
    settings = storage.get_settings()
    assert settings.system_prompt == "member bot"
    assert settings.enabled_rag_collections == []
    storage.append_history(
        provider="openai",
        model="m",
        system_prompt="p",
        question="q",
        answer="a",
        rag_collections=[],
        rag_hits=0,
        config_snapshot={},
    )
    (record,) = storage.get_history()
    assert (record.doc_ids, record.topics, record.answered) == ([], [], None)


def _append(storage: Storage, **extra: object) -> int:
    return storage.append_history(
        provider="openai",
        model="m",
        system_prompt="p",
        question="q",
        answer="a",
        rag_collections=[],
        rag_hits=0,
        config_snapshot={},
        **extra,  # type: ignore[arg-type]
    )


def test_version_3_adds_timings_to_saved_chats(tmp_path: Path) -> None:
    db_path = tmp_path / "v3.db"
    _append(Storage(db_path=db_path, default_settings=AppSettings(system_prompt="x")))
    with sqlite3.connect(db_path) as conn:
        conn.execute("ALTER TABLE chat_history DROP COLUMN timings_json")
        conn.execute("DROP TABLE chat_timing_stats")
        conn.execute("DROP TABLE chat_timing_counts")
        conn.execute("PRAGMA user_version = 3")
    conn.close()

    storage = Storage(db_path=db_path, default_settings=AppSettings(system_prompt="x"))
    _append(storage, timings={"qdrant": 12.5, "total": 40.0})
    newer, older = storage.get_history()
    assert newer.timings == {"qdrant": 12.5, "total": 40.0}
    assert older.timings == {}
    assert storage.get_timing_stats().chats == 0


def test_timing_tally_survives_a_restart_and_resets(tmp_path: Path) -> None:
    db_path = tmp_path / "timings.db"
    settings = AppSettings(system_prompt="x")
    storage = Storage(db_path=db_path, default_settings=settings)
    assert storage.get_timing_stats().model_dump() == {
        "chats": 0,
        "errors": 0,
        "since": None,
        "stages": [],
    }
    storage.record_timings({"qdrant": 10.0, "total": 30.0})
    storage.record_timings({"rerank": 5.0, "qdrant": 20.0, "total": 40.0}, error=True)

    stats = Storage(db_path=db_path, default_settings=settings).get_timing_stats()

    assert (stats.chats, stats.errors) == (2, 1)
    assert stats.since is not None
    assert [row.stage for row in stats.stages] == ["qdrant", "rerank", "total"]
    qdrant = stats.stages[0]
    assert (qdrant.count, qdrant.average_ms, qdrant.min_ms, qdrant.max_ms, qdrant.last_ms) == (
        2,
        15.0,
        10.0,
        20.0,
        20.0,
    )
    assert stats.stages[1].count == 1

    _append(storage, timings={"total": 1.0})
    storage.reset_timing_stats()
    assert storage.get_timing_stats().chats == 0
    # Saved chats keep their own timings.
    assert storage.get_history()[0].timings == {"total": 1.0}
