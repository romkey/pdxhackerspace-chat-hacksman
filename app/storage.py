from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.models import AppSettings, HistoryRecord


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
            conn.commit()

    def get_settings(self) -> AppSettings:
        with self._connect() as conn:
            row = conn.execute("SELECT payload_json FROM app_settings WHERE id = 1").fetchone()
            if not row:
                return self.default_settings
            payload = json.loads(row["payload_json"])
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
    ) -> None:
        created_at = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
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
