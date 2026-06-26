"""
db.py — SQLite operations for the music library

Schema:
    songs table — one row per indexed song
        id          INTEGER PRIMARY KEY
        filepath    TEXT UNIQUE
        file_hash   TEXT
        title       TEXT
        artist      TEXT
        album       TEXT
        duration    REAL
        indexed_at  TEXT — ISO timestamp
        valence     REAL  0.0-1.0 emotional positivity (NULL if not yet computed)
        energy      REAL  0.0-1.0 intensity/loudness
        danceability REAL 0.0-1.0 how groovy/danceable
        bpm         REAL  beats per minute
        key         INTEGER 0-11 (C=0 ... B=11)
        mode        INTEGER 1=major 0=minor

Uses a persistent connection during indexing sessions to avoid
hammering the OS with open/close calls for 500+ songs.
"""

import hashlib
import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from config import DB_PATH, DATA_DIR


# ---------------------------------------------------------------------------
# Persistent connection (used by indexer for bulk operations)
# ---------------------------------------------------------------------------

_persistent_conn: Optional[sqlite3.Connection] = None


def open_session() -> None:
    """
    Open a persistent DB connection for a bulk indexing session.
    Call this at the start of indexer.run_index(), close_session() at the end.
    Avoids opening/closing hundreds of connections for large libraries.
    """
    global _persistent_conn
    _ensure_data_dir()
    conn = sqlite3.connect(DB_PATH, timeout=60, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # safe concurrent reads during write
    conn.execute("PRAGMA synchronous=NORMAL") # faster writes, still safe
    conn.execute("PRAGMA cache_size=-64000")  # 64MB cache for large libraries
    _persistent_conn = conn


def close_session() -> None:
    """Close the persistent connection after a bulk indexing session."""
    global _persistent_conn
    if _persistent_conn:
        _persistent_conn.commit()
        _persistent_conn.close()
        _persistent_conn = None


# ---------------------------------------------------------------------------
# Connection helper — uses persistent conn if open, else opens a new one
# ---------------------------------------------------------------------------

def _ensure_data_dir() -> None:
    """Make sure the data directory exists and is writable."""
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
    except OSError as e:
        raise sqlite3.OperationalError(
            f"Cannot create data directory '{DATA_DIR}': {e}\n"
            f"Try: sudo mkdir -p {DATA_DIR} && sudo chown $(whoami) {DATA_DIR}"
        )
    if not os.access(DATA_DIR, os.W_OK):
        raise sqlite3.OperationalError(
            f"Data directory is not writable: '{DATA_DIR}'\n"
            f"Try: sudo chown -R $(whoami) {DATA_DIR}"
        )


def _connect() -> sqlite3.Connection:
    """
    Return the active connection.
    Uses the persistent session connection if one is open (indexer mode),
    otherwise opens a fresh short-lived connection (Flask API mode).
    """
    if _persistent_conn is not None:
        return _persistent_conn

    _ensure_data_dir()
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _is_session_conn(conn: sqlite3.Connection) -> bool:
    return conn is _persistent_conn


# ---------------------------------------------------------------------------
# Schema init
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create tables if they don't exist. Migrates existing DBs to add new columns."""
    _ensure_data_dir()
    conn = _connect()

    # Create table with full schema
    conn.execute("""
        CREATE TABLE IF NOT EXISTS songs (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            filepath     TEXT    NOT NULL UNIQUE,
            file_hash    TEXT    NOT NULL,
            title        TEXT    NOT NULL DEFAULT 'Unknown Title',
            artist       TEXT    NOT NULL DEFAULT 'Unknown Artist',
            album        TEXT    NOT NULL DEFAULT 'Unknown Album',
            duration     REAL    NOT NULL DEFAULT 0.0,
            indexed_at   TEXT    NOT NULL,
            valence      REAL,
            energy       REAL,
            danceability REAL,
            bpm          REAL,
            key          INTEGER,
            mode         INTEGER
        )
    """)

    # Migration: add new columns to existing DBs that don't have them yet
    # ALTER TABLE ADD COLUMN is safe to run even if migration was partial
    existing_cols = {
        row[1] for row in conn.execute("PRAGMA table_info(songs)").fetchall()
    }
    new_cols = {
        "valence":      "REAL",
        "energy":       "REAL",
        "danceability": "REAL",
        "bpm":          "REAL",
        "key":          "INTEGER",
        "mode":         "INTEGER",
    }
    for col, col_type in new_cols.items():
        if col not in existing_cols:
            conn.execute(f"ALTER TABLE songs ADD COLUMN {col} {col_type}")
            print(f"[db] Migrated: added column '{col}'")

    conn.commit()
    if not _is_session_conn(conn):
        conn.close()


# ---------------------------------------------------------------------------
# File hashing
# ---------------------------------------------------------------------------

def hash_file(filepath: str, chunk_size: int = 65536) -> str:
    """Return MD5 hex digest of a file."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------

def insert_song(
    filepath:     str,
    file_hash:    str,
    title:        str,
    artist:       str,
    album:        str,
    duration:     float,
    valence:      float | None = None,
    energy:       float | None = None,
    danceability: float | None = None,
    bpm:          float | None = None,
    key:          int   | None = None,
    mode:         int   | None = None,
) -> int:
    """Insert a new song row. Returns the new row id."""
    now = datetime.now(timezone.utc).isoformat()
    conn = _connect()
    cursor = conn.execute(
        """
        INSERT INTO songs
            (filepath, file_hash, title, artist, album, duration, indexed_at,
             valence, energy, danceability, bpm, key, mode)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (filepath, file_hash, title, artist, album, duration, now,
         valence, energy, danceability, bpm, key, mode),
    )
    conn.commit()
    row_id = cursor.lastrowid
    if not _is_session_conn(conn):
        conn.close()
    return row_id


def update_song_hash(song_id: int, file_hash: str) -> None:
    """Update stored hash after a file has been re-indexed."""
    now = datetime.now(timezone.utc).isoformat()
    conn = _connect()
    conn.execute(
        "UPDATE songs SET file_hash = ?, indexed_at = ? WHERE id = ?",
        (file_hash, now, song_id),
    )
    conn.commit()
    if not _is_session_conn(conn):
        conn.close()


def update_song_features(
    song_id:      int,
    valence:      float,
    energy:       float,
    danceability: float,
    bpm:          float,
    key:          int,
    mode:         int,
) -> None:
    """Update audio features for an existing song row."""
    conn = _connect()
    conn.execute(
        """UPDATE songs
           SET valence=?, energy=?, danceability=?, bpm=?, key=?, mode=?
           WHERE id=?""",
        (valence, energy, danceability, bpm, key, mode, song_id),
    )
    conn.commit()
    if not _is_session_conn(conn):
        conn.close()


def delete_song(song_id: int) -> None:
    """Remove a stale song row."""
    conn = _connect()
    conn.execute("DELETE FROM songs WHERE id = ?", (song_id,))
    conn.commit()
    if not _is_session_conn(conn):
        conn.close()


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------

def is_indexed(filepath: str) -> Optional[sqlite3.Row]:
    """Return the song row if this filepath is in the DB, else None."""
    conn = _connect()
    row = conn.execute(
        "SELECT * FROM songs WHERE filepath = ?", (filepath,)
    ).fetchone()
    if not _is_session_conn(conn):
        conn.close()
    return row


def get_song_by_id(song_id: int) -> Optional[sqlite3.Row]:
    conn = _connect()
    row = conn.execute(
        "SELECT * FROM songs WHERE id = ?", (song_id,)
    ).fetchone()
    if not _is_session_conn(conn):
        conn.close()
    return row


def get_all_songs() -> list:
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM songs ORDER BY artist, title"
    ).fetchall()
    if not _is_session_conn(conn):
        conn.close()
    return rows


def get_song_count() -> int:
    conn = _connect()
    row = conn.execute("SELECT COUNT(*) FROM songs").fetchone()
    if not _is_session_conn(conn):
        conn.close()
    return row[0]


def get_all_filepaths() -> set:
    """Return set of all indexed filepaths — used to detect stale entries."""
    conn = _connect()
    rows = conn.execute("SELECT filepath FROM songs").fetchall()
    if not _is_session_conn(conn):
        conn.close()
    return {row["filepath"] for row in rows}
