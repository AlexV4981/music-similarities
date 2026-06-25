"""
indexer.py — Batch music library indexer

Architecture (based on SO/batch processing best practices):
  - Files are split into chunks of BATCH_SIZE using a generator
  - Each batch: embed all songs → bulk insert to SQLite via executemany → 
    add vectors to FAISS → save checkpoint to disk
  - Only one batch lives in memory at a time
  - A crash mid-run loses at most one batch (50 songs), not everything

Run modes:
  python indexer.py /path/to/music            # full index
  python indexer.py /path/to/music --update   # skip unchanged files
  python indexer.py /path/to/music --clean    # remove deleted files from index
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Generator

import faiss
import numpy as np
from mutagen import File as MutagenFile
from tqdm import tqdm

import db
from config import (
    EMBEDDING_DIM,
    FAISS_ID_MAP_PATH,
    FAISS_INDEX_PATH,
    SUPPORTED_FORMATS,
)
from extractor import (
    AudioLoadError,
    AudioTooShortError,
    SilentAudioError,
    UnsupportedFormatError,
    get_embedding,
)

BATCH_SIZE = 50   # process → index → save to disk every N songs


# ---------------------------------------------------------------------------
# Batch item dataclass — holds everything for one song through the pipeline
# ---------------------------------------------------------------------------

@dataclass
class SongItem:
    filepath:  str
    title:     str
    artist:    str
    album:     str
    duration:  float
    file_hash: str
    vector:    np.ndarray
    db_id:     int | None = None   # set after DB insert


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def _extract_metadata(filepath: str) -> dict:
    """Pull tags from file. Falls back to filename / 'Unknown' — never raises."""
    meta = {
        "title":    os.path.splitext(os.path.basename(filepath))[0],
        "artist":   "Unknown Artist",
        "album":    "Unknown Album",
        "duration": 0.0,
    }
    try:
        tags = MutagenFile(filepath, easy=True)
        if tags is None:
            return meta
        def _first(key):
            val = tags.get(key)
            return val[0].strip() if val else None
        meta["title"]    = _first("title")  or meta["title"]
        meta["artist"]   = _first("artist") or meta["artist"]
        meta["album"]    = _first("album")  or meta["album"]
        if hasattr(tags, "info") and hasattr(tags.info, "length"):
            meta["duration"] = float(tags.info.length)
    except Exception:
        pass
    return meta


# ---------------------------------------------------------------------------
# FAISS helpers
# ---------------------------------------------------------------------------

def _new_index() -> faiss.IndexFlatIP:
    return faiss.IndexFlatIP(EMBEDDING_DIM)

def _load_index() -> faiss.IndexFlatIP:
    if os.path.exists(FAISS_INDEX_PATH):
        return faiss.read_index(FAISS_INDEX_PATH)
    return _new_index()

def _save_index(index: faiss.IndexFlatIP) -> None:
    faiss.write_index(index, FAISS_INDEX_PATH)

def _load_id_map() -> dict:
    if os.path.exists(FAISS_ID_MAP_PATH):
        with open(FAISS_ID_MAP_PATH) as f:
            return json.load(f)
    return {}

def _save_id_map(id_map: dict) -> None:
    with open(FAISS_ID_MAP_PATH, "w") as f:
        json.dump(id_map, f)


# ---------------------------------------------------------------------------
# File collection
# ---------------------------------------------------------------------------

def _collect_audio_files(music_dir: str) -> list[str]:
    found = []
    for root, _, files in os.walk(music_dir):
        for fname in files:
            if os.path.splitext(fname)[1].lower() in SUPPORTED_FORMATS:
                found.append(os.path.join(root, fname))
    return sorted(found)


# ---------------------------------------------------------------------------
# Batch generator
# Yields lists of BATCH_SIZE filepaths at a time — only one batch in memory
# Pattern from: https://stackoverflow.com/a/8991553
# ---------------------------------------------------------------------------

def _batched(items: list, size: int) -> Generator[list, None, None]:
    """Split a list into chunks of `size`. Yields each chunk as a list."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


# ---------------------------------------------------------------------------
# Single-batch processor
# Embed → bulk DB insert via executemany → add to FAISS → checkpoint to disk
# ---------------------------------------------------------------------------

def _process_batch(
    filepaths:   list[str],
    index:       faiss.IndexFlatIP,
    id_map:      dict,
    update_only: bool,
    pbar:        tqdm,
) -> tuple[int, int, int, list]:
    """
    Process one batch of up to BATCH_SIZE songs.

    Returns (added, skipped, errors, error_log_entries)
    """
    added    = 0
    skipped  = 0
    errors   = 0
    err_log  = []

    # ── Phase 1: filter + embed ───────────────────────────────────────────────
    # Build list of SongItems for songs that need embedding.
    # Songs that are unchanged (--update) are skipped here.
    to_embed:      list[tuple[str, object | None]] = []   # (filepath, existing_row)
    skip_filepaths: set[str] = set()

    for filepath in filepaths:
        existing = db.is_indexed(filepath)
        if update_only and existing:
            current_hash = db.hash_file(filepath)
            if current_hash == existing["file_hash"]:
                skip_filepaths.add(filepath)
                skipped += 1
                pbar.update(1)
                continue
        to_embed.append((filepath, existing))

    # ── Phase 2: extract embeddings ───────────────────────────────────────────
    embedded: list[SongItem] = []

    for filepath, existing in to_embed:
        try:
            vector = get_embedding(filepath)
        except (AudioLoadError, AudioTooShortError,
                SilentAudioError, UnsupportedFormatError) as e:
            errors += 1
            err_log.append((filepath, str(e)))
            pbar.update(1)
            continue
        except Exception as e:
            errors += 1
            err_log.append((filepath, f"Unexpected: {e}"))
            pbar.update(1)
            continue

        meta      = _extract_metadata(filepath)
        file_hash = db.hash_file(filepath)

        item = SongItem(
            filepath  = filepath,
            title     = meta["title"],
            artist    = meta["artist"],
            album     = meta["album"],
            duration  = meta["duration"],
            file_hash = file_hash,
            vector    = vector,
            db_id     = existing["id"] if existing else None,
        )
        embedded.append(item)

    if not embedded:
        return added, skipped, errors, err_log

    # ── Phase 3: bulk DB write (executemany — much faster than one-by-one) ───
    # Pattern: https://remusao.github.io/posts/few-tips-sqlite-perf.html
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()

    new_items      = [it for it in embedded if it.db_id is None]
    updated_items  = [it for it in embedded if it.db_id is not None]

    # Bulk insert new songs
    if new_items:
        rows = [
            (it.filepath, it.file_hash, it.title, it.artist,
             it.album, it.duration, now)
            for it in new_items
        ]
        conn = db._connect()
        cursor = conn.executemany(
            """INSERT OR IGNORE INTO songs
               (filepath, file_hash, title, artist, album, duration, indexed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        conn.commit()

        # Fetch the IDs that were just inserted
        placeholders = ",".join("?" * len(new_items))
        id_rows = conn.execute(
            f"SELECT id, filepath FROM songs WHERE filepath IN ({placeholders})",
            [it.filepath for it in new_items],
        ).fetchall()
        path_to_id = {r["filepath"]: r["id"] for r in id_rows}
        for it in new_items:
            it.db_id = path_to_id.get(it.filepath)

    # Bulk update changed files
    if updated_items:
        conn = db._connect()
        conn.executemany(
            "UPDATE songs SET file_hash = ?, indexed_at = ? WHERE id = ?",
            [(it.file_hash, now, it.db_id) for it in updated_items],
        )
        conn.commit()

    # ── Phase 4: add vectors to FAISS + update id_map ────────────────────────
    valid = [it for it in embedded if it.db_id is not None]
    if valid:
        vectors = np.stack([it.vector for it in valid]).astype(np.float32)
        index.add(vectors)   # add whole batch at once — faster than one-by-one
        for it in valid:
            faiss_row = index.ntotal - len(valid) + valid.index(it)
            id_map[str(faiss_row)] = it.db_id
        added += len(valid)

    # ── Phase 5: checkpoint to disk after every batch ─────────────────────────
    _save_index(index)
    _save_id_map(id_map)

    for _ in embedded:
        pbar.update(1)

    return added, skipped, errors, err_log


# ---------------------------------------------------------------------------
# Stale entry cleanup
# ---------------------------------------------------------------------------

def clean_stale_entries() -> int:
    all_paths = db.get_all_filepaths()
    stale = [p for p in all_paths if not os.path.exists(p)]
    for path in stale:
        row = db.is_indexed(path)
        if row:
            db.delete_song(row["id"])
    if stale:
        print(f"[indexer] Removed {len(stale)} stale entries.")
    return len(stale)


# ---------------------------------------------------------------------------
# Main indexer
# ---------------------------------------------------------------------------

def run_index(music_dir: str, update_only: bool = False) -> None:
    if not os.path.isdir(music_dir):
        print(f"[indexer] ERROR: '{music_dir}' is not a directory.")
        sys.exit(1)

    db.init_db()
    db.open_session()

    try:
        audio_files = _collect_audio_files(music_dir)
        if not audio_files:
            print(f"[indexer] No supported audio files found in '{music_dir}'.")
            return

        total = len(audio_files)
        batches = list(_batched(audio_files, BATCH_SIZE))
        n_batches = len(batches)

        print(f"[indexer] Found {total} audio files → {n_batches} batches of up to {BATCH_SIZE}")

        if update_only:
            index  = _load_index()
            id_map = _load_id_map()
            print(f"[indexer] --update mode: existing index has {index.ntotal} vectors.")
        else:
            index  = _new_index()
            id_map = {}
            print("[indexer] Full re-index: building from scratch.")

        total_added   = 0
        total_skipped = 0
        total_errors  = 0
        all_errors    = []

        with tqdm(total=total, desc="Indexing", unit="song") as pbar:
            for batch_num, batch in enumerate(batches, 1):
                pbar.set_description(f"Batch {batch_num}/{n_batches}")

                added, skipped, errors, err_log = _process_batch(
                    filepaths   = batch,
                    index       = index,
                    id_map      = id_map,
                    update_only = update_only,
                    pbar        = pbar,
                )

                total_added   += added
                total_skipped += skipped
                total_errors  += errors
                all_errors    += err_log

                tqdm.write(
                    f"  Batch {batch_num}/{n_batches} — "
                    f"added: {added}  skipped: {skipped}  errors: {errors}  "
                    f"[index total: {index.ntotal}]"
                )

    finally:
        db.close_session()

    print(f"\n[indexer] Done.")
    print(f"  Added/updated : {total_added}")
    print(f"  Skipped       : {total_skipped}")
    print(f"  Errors        : {total_errors}")
    print(f"  Total in index: {index.ntotal}")

    if all_errors:
        print(f"\n[indexer] Files that failed:")
        for path, reason in all_errors:
            print(f"  {os.path.basename(path)}: {reason}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index your music library.")
    parser.add_argument("music_dir", help="Path to your music folder")
    parser.add_argument("--update", action="store_true",
                        help="Only process new or changed files")
    parser.add_argument("--clean",  action="store_true",
                        help="Remove DB entries for deleted files, then exit")
    args = parser.parse_args()

    if args.clean:
        db.init_db()
        db.open_session()
        removed = clean_stale_entries()
        db.close_session()
        print(f"Cleaned {removed} stale entries.")
        sys.exit(0)

    run_index(music_dir=args.music_dir, update_only=args.update)
