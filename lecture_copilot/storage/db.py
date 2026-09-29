from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from lecture_copilot.config import DB_PATH, ensure_dirs


SCHEMA = """
CREATE TABLE IF NOT EXISTS lectures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    subject TEXT,
    topic TEXT,
    source_language TEXT,
    target_language TEXT,
    date TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    cost_usd REAL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS transcript_segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lecture_id INTEGER NOT NULL,
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    original_text TEXT NOT NULL,
    translated_text TEXT,
    FOREIGN KEY (lecture_id) REFERENCES lectures(id)
);

CREATE TABLE IF NOT EXISTS slides (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lecture_id INTEGER NOT NULL,
    timestamp REAL NOT NULL,
    image_path TEXT,
    ocr_text TEXT,
    summary TEXT,
    title TEXT,
    formulas TEXT,
    code TEXT,
    important INTEGER DEFAULT 0,
    FOREIGN KEY (lecture_id) REFERENCES lectures(id)
);

CREATE TABLE IF NOT EXISTS slide_points (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    slide_id INTEGER NOT NULL,
    content TEXT NOT NULL,
    importance INTEGER DEFAULT 0,
    FOREIGN KEY (slide_id) REFERENCES slides(id)
);
"""


def connect(path: Path | None = None) -> sqlite3.Connection:
    ensure_dirs()
    db = path or DB_PATH
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def get_conn(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    conn = connect(path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(path: Path | None = None) -> None:
    with get_conn(path) as conn:
        conn.executescript(SCHEMA)
        # Migrate older databases created before cost tracking existed.
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(lectures)")}
        if "cost_usd" not in columns:
            conn.execute("ALTER TABLE lectures ADD COLUMN cost_usd REAL DEFAULT 0")


def create_lecture(
    title: str,
    subject: str,
    topic: str,
    source_language: str,
    target_language: str,
) -> int:
    now = datetime.now(timezone.utc)
    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO lectures (title, subject, topic, source_language, target_language, date, started_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                title,
                subject,
                topic,
                source_language,
                target_language,
                now.date().isoformat(),
                now.isoformat(),
            ),
        )
        return int(cur.lastrowid)


def end_lecture(lecture_id: int, cost_usd: float = 0.0) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE lectures SET ended_at = ?, cost_usd = ? WHERE id = ?",
            (datetime.now(timezone.utc).isoformat(), cost_usd, lecture_id),
        )


def insert_segment(
    lecture_id: int,
    start_time: float,
    end_time: float,
    original_text: str,
    translated_text: str | None,
) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO transcript_segments
                (lecture_id, start_time, end_time, original_text, translated_text)
            VALUES (?, ?, ?, ?, ?)
            """,
            (lecture_id, start_time, end_time, original_text, translated_text),
        )
        return int(cur.lastrowid)


def update_segment_translation(segment_id: int, translated_text: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE transcript_segments SET translated_text = ? WHERE id = ?",
            (translated_text, segment_id),
        )


def list_segments(lecture_id: int) -> list[sqlite3.Row]:
    with get_conn() as conn:
        cur = conn.execute(
            """
            SELECT start_time, end_time, original_text, translated_text
            FROM transcript_segments
            WHERE lecture_id = ?
            ORDER BY start_time
            """,
            (lecture_id,),
        )
        return list(cur.fetchall())


def get_lecture(lecture_id: int) -> sqlite3.Row | None:
    with get_conn() as conn:
        cur = conn.execute("SELECT * FROM lectures WHERE id = ?", (lecture_id,))
        return cur.fetchone()


def list_lectures() -> list[sqlite3.Row]:
    with get_conn() as conn:
        cur = conn.execute("SELECT * FROM lectures ORDER BY id DESC")
        return list(cur.fetchall())


def insert_slide(
    lecture_id: int,
    timestamp: float,
    image_path: str | None,
    ocr_text: str,
    summary: str,
    title: str,
    formulas: str,
    code: str,
    important: int = 0,
) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO slides
                (lecture_id, timestamp, image_path, ocr_text, summary, title, formulas, code, important)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (lecture_id, timestamp, image_path, ocr_text, summary, title, formulas, code, important),
        )
        return int(cur.lastrowid)


def insert_slide_point(slide_id: int, content: str, importance: int = 0) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO slide_points (slide_id, content, importance) VALUES (?, ?, ?)",
            (slide_id, content, importance),
        )
        return int(cur.lastrowid)


def list_slides(lecture_id: int) -> list[sqlite3.Row]:
    with get_conn() as conn:
        cur = conn.execute(
            """
            SELECT id, timestamp, image_path, ocr_text, summary, title, formulas, code, important
            FROM slides
            WHERE lecture_id = ?
            ORDER BY timestamp
            """,
            (lecture_id,),
        )
        return list(cur.fetchall())


def list_slide_points(slide_id: int) -> list[sqlite3.Row]:
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT content, importance FROM slide_points WHERE slide_id = ? ORDER BY id",
            (slide_id,),
        )
        return list(cur.fetchall())
