from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.models import (
    AppSettings,
    EventRecord,
    FeedbackCreateRequest,
    FeedbackRecord,
    HistoryRecord,
    OccurrenceRecord,
)


class Storage:
    def __init__(self, db_path: str, default_settings: AppSettings) -> None:
        self.db_path = db_path
        self.default_settings = default_settings
        self._ensure_parent_dir()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_parent_dir(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS app_settings (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    system_prompt TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    rag_collections_json TEXT NOT NULL,
                    rag_hits INTEGER NOT NULL,
                    config_snapshot_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feed_events (
                    event_id INTEGER PRIMARY KEY,
                    slug TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    more_info_url TEXT NOT NULL,
                    visibility TEXT NOT NULL,
                    open_to TEXT NOT NULL,
                    recurrence_type TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    duration INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feed_occurrences (
                    occurrence_id INTEGER PRIMARY KEY,
                    slug TEXT NOT NULL,
                    occurs_at TEXT NOT NULL,
                    occurs_at_unix INTEGER NOT NULL,
                    ends_at_unix INTEGER NOT NULL,
                    duration INTEGER NOT NULL,
                    is_cancelled INTEGER NOT NULL,
                    is_postponed INTEGER NOT NULL,
                    in_progress INTEGER NOT NULL,
                    postponed_until TEXT,
                    open_to TEXT NOT NULL,
                    event_id INTEGER,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    rating TEXT NOT NULL,
                    text TEXT NOT NULL,
                    history_id INTEGER,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_base_url_history (
                    url TEXT PRIMARY KEY,
                    last_used_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_base_url_status (
                    url TEXT PRIMARY KEY,
                    is_available INTEGER NOT NULL,
                    last_changed_at TEXT NOT NULL,
                    last_checked_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def get_settings(self) -> AppSettings:
        with self._connect() as conn:
            row = conn.execute("SELECT payload_json FROM app_settings WHERE id = 1").fetchone()
            if not row:
                return self.default_settings
            payload = json.loads(row["payload_json"])
            if "enabled_rag_collections" not in payload:
                payload["enabled_rag_collections"] = self.default_settings.enabled_rag_collections
            return AppSettings.model_validate(payload)

    def save_settings(self, settings: AppSettings) -> AppSettings:
        now = datetime.now(UTC).isoformat()
        payload_json = settings.model_dump_json()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO app_settings (id, payload_json, updated_at)
                VALUES (1, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (payload_json, now),
            )
            conn.commit()
        return settings

    def remember_llm_base_url(self, llm_base_url: str) -> None:
        normalized = llm_base_url.strip()
        if not normalized:
            return
        last_used_at = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO llm_base_url_history (url, last_used_at)
                VALUES (?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    last_used_at = excluded.last_used_at
                """,
                (normalized, last_used_at),
            )
            conn.commit()

    def get_llm_base_urls(self, *, limit: int = 200) -> list[str]:
        safe_limit = max(1, min(limit, 1000))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT url
                FROM llm_base_url_history
                ORDER BY LOWER(url) ASC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()
        return [str(row["url"]) for row in rows]

    def get_llm_base_url_status(
        self, llm_base_url: str
    ) -> tuple[bool, datetime, datetime] | None:
        normalized = llm_base_url.strip()
        if not normalized:
            return None
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT is_available, last_changed_at, last_checked_at
                FROM llm_base_url_status
                WHERE url = ?
                """,
                (normalized,),
            ).fetchone()
        if not row:
            return None
        return (
            bool(row["is_available"]),
            datetime.fromisoformat(row["last_changed_at"]),
            datetime.fromisoformat(row["last_checked_at"]),
        )

    def upsert_llm_base_url_status(
        self,
        *,
        llm_base_url: str,
        is_available: bool,
        last_changed_at: datetime,
        last_checked_at: datetime,
    ) -> None:
        normalized = llm_base_url.strip()
        if not normalized:
            return
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO llm_base_url_status (
                    url,
                    is_available,
                    last_changed_at,
                    last_checked_at
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    is_available = excluded.is_available,
                    last_changed_at = excluded.last_changed_at,
                    last_checked_at = excluded.last_checked_at
                """,
                (
                    normalized,
                    int(is_available),
                    last_changed_at.isoformat(),
                    last_checked_at.isoformat(),
                ),
            )
            conn.commit()

    def append_history(
        self,
        *,
        provider: str,
        model: str,
        system_prompt: str,
        question: str,
        answer: str,
        rag_collections: list[str],
        rag_hits: int,
        config_snapshot: dict[str, Any],
    ) -> int:
        created_at = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO chat_history (
                    created_at,
                    provider,
                    model,
                    system_prompt,
                    question,
                    answer,
                    rag_collections_json,
                    rag_hits,
                    config_snapshot_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    provider,
                    model,
                    system_prompt,
                    question,
                    answer,
                    json.dumps(rag_collections),
                    rag_hits,
                    json.dumps(config_snapshot),
                ),
            )
            conn.commit()
            row_id = cursor.lastrowid
            return int(row_id) if row_id is not None else 0

    def get_history(self, *, limit: int = 100) -> list[HistoryRecord]:
        safe_limit = max(1, min(limit, 500))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    id,
                    created_at,
                    provider,
                    model,
                    system_prompt,
                    question,
                    answer,
                    rag_collections_json,
                    rag_hits,
                    config_snapshot_json
                FROM chat_history
                ORDER BY id DESC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()

        records: list[HistoryRecord] = []
        for row in rows:
            records.append(
                HistoryRecord(
                    id=row["id"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    provider=row["provider"],
                    model=row["model"],
                    system_prompt=row["system_prompt"],
                    question=row["question"],
                    answer=row["answer"],
                    rag_collections=json.loads(row["rag_collections_json"]),
                    rag_hits=row["rag_hits"],
                    config_snapshot=json.loads(row["config_snapshot_json"]),
                )
            )
        return records

    def clear_history(self) -> int:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM chat_history")
            conn.commit()
            return int(cursor.rowcount or 0)

    def delete_latest_history(self) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                DELETE FROM chat_history
                WHERE id = (
                    SELECT id
                    FROM chat_history
                    ORDER BY id DESC
                    LIMIT 1
                )
                """
            )
            conn.commit()
            return int(cursor.rowcount or 0)

    def append_feedback(self, feedback: FeedbackCreateRequest) -> int:
        created_at = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO chat_feedback (
                    created_at,
                    rating,
                    text,
                    history_id,
                    question,
                    answer,
                    provider,
                    model
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    feedback.rating,
                    feedback.text.strip(),
                    feedback.history_id,
                    feedback.question.strip(),
                    feedback.answer.strip(),
                    feedback.provider.strip(),
                    feedback.model.strip(),
                ),
            )
            conn.commit()
            row_id = cursor.lastrowid
            return int(row_id) if row_id is not None else 0

    def get_feedback(self, *, limit: int = 100) -> list[FeedbackRecord]:
        safe_limit = max(1, min(limit, 500))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    id,
                    created_at,
                    rating,
                    text,
                    history_id,
                    question,
                    answer,
                    provider,
                    model
                FROM chat_feedback
                ORDER BY id DESC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()

        records: list[FeedbackRecord] = []
        for row in rows:
            rating = "up" if row["rating"] == "up" else "down"
            records.append(
                FeedbackRecord(
                    id=row["id"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    rating=rating,
                    text=row["text"],
                    history_id=row["history_id"],
                    question=row["question"],
                    answer=row["answer"],
                    provider=row["provider"],
                    model=row["model"],
                )
            )
        return records

    def upsert_events(self, events: list[dict[str, Any]]) -> int:
        if not events:
            return 0

        updated_at = datetime.now(UTC).isoformat()
        upserted = 0
        with self._connect() as conn:
            for item in events:
                event_id = item.get("id")
                if not isinstance(event_id, int):
                    continue
                slug = str(item.get("slug", ""))
                title = str(item.get("title", ""))
                description = str(item.get("description", ""))
                more_info_url = str(item.get("more_info_url", ""))
                visibility = str(item.get("visibility", ""))
                open_to = str(item.get("open_to", ""))
                recurrence_type = str(item.get("recurrence_type", ""))
                start_time = str(item.get("start_time", ""))
                duration = item.get("duration")
                if not isinstance(duration, int):
                    duration = 0

                conn.execute(
                    """
                    INSERT INTO feed_events (
                        event_id,
                        slug,
                        title,
                        description,
                        more_info_url,
                        visibility,
                        open_to,
                        recurrence_type,
                        start_time,
                        duration,
                        payload_json,
                        updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(event_id) DO UPDATE SET
                        slug = excluded.slug,
                        title = excluded.title,
                        description = excluded.description,
                        more_info_url = excluded.more_info_url,
                        visibility = excluded.visibility,
                        open_to = excluded.open_to,
                        recurrence_type = excluded.recurrence_type,
                        start_time = excluded.start_time,
                        duration = excluded.duration,
                        payload_json = excluded.payload_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        event_id,
                        slug,
                        title,
                        description,
                        more_info_url,
                        visibility,
                        open_to,
                        recurrence_type,
                        start_time,
                        duration,
                        json.dumps(item),
                        updated_at,
                    ),
                )
                upserted += 1
            conn.commit()
        return upserted

    def get_events(self, *, limit: int = 200) -> list[EventRecord]:
        safe_limit = max(1, min(limit, 1000))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    event_id,
                    slug,
                    title,
                    description,
                    more_info_url,
                    visibility,
                    open_to,
                    recurrence_type,
                    start_time,
                    duration,
                    payload_json
                FROM feed_events
                ORDER BY start_time ASC, event_id ASC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()

        result: list[EventRecord] = []
        for row in rows:
            result.append(
                EventRecord(
                    event_id=row["event_id"],
                    slug=row["slug"],
                    title=row["title"],
                    description=row["description"],
                    more_info_url=row["more_info_url"],
                    visibility=row["visibility"],
                    open_to=row["open_to"],
                    recurrence_type=row["recurrence_type"],
                    start_time=row["start_time"],
                    duration=row["duration"],
                    payload=json.loads(row["payload_json"]),
                )
            )
        return result

    def upsert_occurrences(self, occurrences: list[dict[str, Any]]) -> int:
        if not occurrences:
            return 0

        updated_at = datetime.now(UTC).isoformat()
        upserted = 0
        with self._connect() as conn:
            for item in occurrences:
                occurrence_id = item.get("id")
                if not isinstance(occurrence_id, int):
                    continue
                slug = str(item.get("slug", ""))
                occurs_at = str(item.get("occurs_at", ""))
                occurs_at_unix = item.get("occurs_at_unix")
                ends_at_unix = item.get("ends_at_unix")
                duration = item.get("duration")
                if not isinstance(occurs_at_unix, int):
                    occurs_at_unix = 0
                if not isinstance(ends_at_unix, int):
                    ends_at_unix = 0
                if not isinstance(duration, int):
                    duration = 0
                is_cancelled = bool(item.get("is_cancelled", False))
                is_postponed = bool(item.get("is_postponed", False))
                in_progress = bool(item.get("in_progress", False))
                postponed_until = item.get("postponed_until")
                postponed_until_str = str(postponed_until) if postponed_until is not None else None
                open_to = str(item.get("open_to", ""))

                event_info = item.get("event")
                event_id = None
                if isinstance(event_info, dict) and isinstance(event_info.get("id"), int):
                    event_id = event_info.get("id")

                conn.execute(
                    """
                    INSERT INTO feed_occurrences (
                        occurrence_id,
                        slug,
                        occurs_at,
                        occurs_at_unix,
                        ends_at_unix,
                        duration,
                        is_cancelled,
                        is_postponed,
                        in_progress,
                        postponed_until,
                        open_to,
                        event_id,
                        payload_json,
                        updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(occurrence_id) DO UPDATE SET
                        slug = excluded.slug,
                        occurs_at = excluded.occurs_at,
                        occurs_at_unix = excluded.occurs_at_unix,
                        ends_at_unix = excluded.ends_at_unix,
                        duration = excluded.duration,
                        is_cancelled = excluded.is_cancelled,
                        is_postponed = excluded.is_postponed,
                        in_progress = excluded.in_progress,
                        postponed_until = excluded.postponed_until,
                        open_to = excluded.open_to,
                        event_id = excluded.event_id,
                        payload_json = excluded.payload_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        occurrence_id,
                        slug,
                        occurs_at,
                        occurs_at_unix,
                        ends_at_unix,
                        duration,
                        int(is_cancelled),
                        int(is_postponed),
                        int(in_progress),
                        postponed_until_str,
                        open_to,
                        event_id,
                        json.dumps(item),
                        updated_at,
                    ),
                )
                upserted += 1
            conn.commit()
        return upserted

    def get_occurrences(self, *, limit: int = 500) -> list[OccurrenceRecord]:
        safe_limit = max(1, min(limit, 5000))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    occurrence_id,
                    slug,
                    occurs_at,
                    occurs_at_unix,
                    ends_at_unix,
                    duration,
                    is_cancelled,
                    is_postponed,
                    in_progress,
                    postponed_until,
                    open_to,
                    event_id,
                    payload_json
                FROM feed_occurrences
                ORDER BY occurs_at_unix ASC, occurrence_id ASC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()

        result: list[OccurrenceRecord] = []
        for row in rows:
            result.append(
                OccurrenceRecord(
                    occurrence_id=row["occurrence_id"],
                    slug=row["slug"],
                    occurs_at=row["occurs_at"],
                    occurs_at_unix=row["occurs_at_unix"],
                    ends_at_unix=row["ends_at_unix"],
                    duration=row["duration"],
                    is_cancelled=bool(row["is_cancelled"]),
                    is_postponed=bool(row["is_postponed"]),
                    in_progress=bool(row["in_progress"]),
                    postponed_until=row["postponed_until"],
                    open_to=row["open_to"],
                    event_id=row["event_id"],
                    payload=json.loads(row["payload_json"]),
                )
            )
        return result
