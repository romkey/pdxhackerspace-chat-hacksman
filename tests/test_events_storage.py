from __future__ import annotations

from pathlib import Path

from app.models import AppSettings
from app.storage import Storage


def test_storage_upsert_and_get_events(tmp_path: Path) -> None:
    db_path = str(tmp_path / "events.db")
    settings = AppSettings(
        provider="ollama",
        llm_base_url="http://localhost:11434",
        model="llama3.2:latest",
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
        provider="ollama",
        llm_base_url="http://localhost:11434",
        model="llama3.2:latest",
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
        provider="ollama",
        llm_base_url="http://localhost:11434",
        model="llama3.2:latest",
        system_prompt="test",
    )
    storage = Storage(db_path=db_path, default_settings=settings)

    first_id = storage.append_history(
        provider="ollama",
        model="llama3.2:latest",
        system_prompt="Prompt A",
        question="q1",
        answer="a1",
        rag_collections=["events"],
        rag_hits=1,
        config_snapshot={},
    )
    second_id = storage.append_history(
        provider="ollama",
        model="llama3.2:latest",
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
