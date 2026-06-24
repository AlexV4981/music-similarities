"""
db.py — SQLite operations for the music library

Schema:
    songs table — one row per indexed song
        id          INTEGER PRIMARY KEY
        filepath    TEXT UNIQUE        — absolute path on disk
        file_hash   TEXT               — MD5 of file, detects moves/renames
        title       TEXT
        artist      TEXT
        album       TEXT
        duration    REAL               — seconds
        indexed_at  TEXT               — ISO timestamp

All writes go through this module. FAISS holds the vectors;
SQLite holds the metadata. They stay in sync via song.id == FAISS row index.
"""

import hashlib
import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from config import DB_PATH


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # lets callers do row["title"] instead of row[2]
    return conn


# ---------------------------------------------------------------------------
# Schema init
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create tables if they don't exist. Safe to call multiple times."""
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS songs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                filepath    TEXT    NOT NULL UNIQUE,
                file_hash   TEXT    NOT NULL,
                title       TEXT    NOT NULL DEFAULT 'Unknown Title',
                artist      TEXT    NOT NULL DEFAULT 'Unknown Artist',
                album       TEXT    NOT NULL DEFAULT 'Unknown Album',
                duration    REAL    NOT NULL DEFAULT 0.0,
                indexed_at  TEXT    NOT NULL
            )
        """)
        conn.commit()


# ---------------------------------------------------------------------------
# File hashing — used to detect changed/moved files
# ---------------------------------------------------------------------------

def hash_file(filepath: str, chunk_size: int = 65536) -> str:
    """Return MD5 hex digest of a file. Reads in chunks to handle large files."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------

def insert_song(
    filepath: str,
    file_hash: str,
    title: str,
    artist: str,
    album: str,
    duration: float,
) -> int:
    """
    Insert a new song row. Returns the new row id.
    Raises sqlite3.IntegrityError if filepath already exists — callers should
    check is_indexed() first or use upsert_song() for re-index flows.
    """
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO songs (filepath, file_hash, title, artist, album, duration, indexed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (filepath, file_hash, title, artist, album, duration, now),
        )
        conn.commit()
        return cursor.lastrowid


def update_song_hash(song_id: int, file_hash: str) -> None:
    """Update the stored hash after a file has been re-indexed."""
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            "UPDATE songs SET file_hash = ?, indexed_at = ? WHERE id = ?",
            (file_hash, now, song_id),
        )
        conn.commit()


def delete_song(song_id: int) -> None:
    """Remove a stale song row (file no longer exists on disk)."""
    with _connect() as conn:
        conn.execute("DELETE FROM songs WHERE id = ?", (song_id,))
        conn.commit()


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------

def is_indexed(filepath: str) -> Optional[sqlite3.Row]:
    """
    Return the song row if this filepath is already in the DB, else None.
    Callers use the returned row to check if the hash has changed.
    """
    with _connect() as conn:
        return conn.execute(
            "SELECT * FROM songs WHERE filepath = ?", (filepath,)
        ).fetchone()


def get_song_by_id(song_id: int) -> Optional[sqlite3.Row]:
    with _connect() as conn:
        return conn.execute(
            "SELECT * FROM songs WHERE id = ?", (song_id,)
        ).fetchone()


def get_all_songs() -> list:
    """Return all songs ordered by artist then title."""
    with _connect() as conn:
        return conn.execute(
            "SELECT * FROM songs ORDER BY artist, title"
        ).fetchall()


def get_song_count() -> int:
    with _connect() as conn:
        row = conn.execute("SELECT COUNT(*) FROM songs").fetchone()
        return row[0]


def get_all_filepaths() -> set:
    """Return set of all indexed filepaths — used to detect stale entries."""
    with _connect() as conn:
        rows = conn.execute("SELECT filepath FROM songs").fetchall()
        return {row["filepath"] for row in rows}
