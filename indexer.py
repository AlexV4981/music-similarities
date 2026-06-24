"""
indexer.py — Bulk library indexer

Walks a music directory, extracts CLAP embeddings, and stores:
  - Metadata → SQLite (via db.py)
  - Embedding vectors → FAISS index (vectors.index)
  - FAISS row ID → DB song ID map → id_map.json

Run modes:
  python indexer.py /path/to/music          # full index
  python indexer.py /path/to/music --update # skip already-indexed unchanged files
  python indexer.py /path/to/music --clean  # remove stale DB entries for deleted files

The FAISS index uses IndexFlatIP (inner product on L2-normalised vectors = cosine similarity).
Vectors are appended in DB insertion order so FAISS row i == songs.id i.
The id_map bridges the two: id_map[faiss_row] = db_song_id.
"""

import argparse
import json
import os
import sys

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


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def _extract_metadata(filepath: str) -> dict:
    """
    Pull title, artist, album, duration from file tags via mutagen.
    Falls back to filename / 'Unknown' if tags are missing.
    Never raises — bad metadata is not a reason to skip a song.
    """
    meta = {
        "title": os.path.splitext(os.path.basename(filepath))[0],
        "artist": "Unknown Artist",
        "album": "Unknown Album",
        "duration": 0.0,
    }

    try:
        tags = MutagenFile(filepath, easy=True)
        if tags is None:
            return meta

        def _first(key):
            val = tags.get(key)
            return val[0].strip() if val else None

        meta["title"]  = _first("title")  or meta["title"]
        meta["artist"] = _first("artist") or meta["artist"]
        meta["album"]  = _first("album")  or meta["album"]

        if hasattr(tags, "info") and hasattr(tags.info, "length"):
            meta["duration"] = float(tags.info.length)

    except Exception:
        pass   # silently fall back to defaults

    return meta


# ---------------------------------------------------------------------------
# FAISS index helpers
# ---------------------------------------------------------------------------

def _new_index() -> faiss.IndexFlatIP:
    """Create a fresh FAISS inner-product index."""
    return faiss.IndexFlatIP(EMBEDDING_DIM)


def _load_index() -> faiss.IndexFlatIP:
    """Load existing index from disk, or create a new one if not found."""
    if os.path.exists(FAISS_INDEX_PATH):
        return faiss.read_index(FAISS_INDEX_PATH)
    return _new_index()


def _save_index(index: faiss.IndexFlatIP) -> None:
    faiss.write_index(index, FAISS_INDEX_PATH)


def _load_id_map() -> dict:
    """id_map: { str(faiss_row): db_song_id }"""
    if os.path.exists(FAISS_ID_MAP_PATH):
        with open(FAISS_ID_MAP_PATH) as f:
            return json.load(f)
    return {}


def _save_id_map(id_map: dict) -> None:
    with open(FAISS_ID_MAP_PATH, "w") as f:
        json.dump(id_map, f)


# ---------------------------------------------------------------------------
# Stale entry cleanup
# ---------------------------------------------------------------------------

def clean_stale_entries() -> int:
    """
    Remove DB rows whose files no longer exist on disk.
    Returns count of removed entries.
    Note: does NOT rebuild the FAISS index — run a full re-index after cleaning
    if you want FAISS to shrink too (IndexFlatIP doesn't support deletion).
    """
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
# Core indexer
# ---------------------------------------------------------------------------

def _collect_audio_files(music_dir: str) -> list[str]:
    """Walk music_dir recursively and return all supported audio file paths."""
    found = []
    for root, _, files in os.walk(music_dir):
        for fname in files:
            if os.path.splitext(fname)[1].lower() in SUPPORTED_FORMATS:
                found.append(os.path.join(root, fname))
    return sorted(found)


def run_index(music_dir: str, update_only: bool = False) -> None:
    """
    Main indexing routine.

    Args:
        music_dir:   Root directory of your music library.
        update_only: If True, skip files already in DB whose hash hasn't changed.
                     If False, re-index everything (rebuilds FAISS from scratch).
    """
    if not os.path.isdir(music_dir):
        print(f"[indexer] ERROR: '{music_dir}' is not a directory.")
        sys.exit(1)

    db.init_db()

    audio_files = _collect_audio_files(music_dir)
    if not audio_files:
        print(f"[indexer] No supported audio files found in '{music_dir}'.")
        sys.exit(0)

    print(f"[indexer] Found {len(audio_files)} audio files.")

    if update_only:
        # Load existing index and id_map to append to
        index  = _load_index()
        id_map = _load_id_map()
        print(f"[indexer] --update mode: existing index has {index.ntotal} vectors.")
    else:
        # Full rebuild — start fresh
        index  = _new_index()
        id_map = {}
        print("[indexer] Full re-index: building from scratch.")

    skipped   = 0
    added     = 0
    errors    = 0
    error_log = []

    for filepath in tqdm(audio_files, desc="Indexing", unit="song"):
        existing = db.is_indexed(filepath)

        if update_only and existing:
            # Check if the file has changed since last index
            current_hash = db.hash_file(filepath)
            if current_hash == existing["file_hash"]:
                skipped += 1
                continue
            # File changed — re-embed but reuse existing DB row
            # (Can't update FAISS in place; we add a new vector and update the map)
            try:
                vector = get_embedding(filepath)
            except (AudioLoadError, AudioTooShortError, SilentAudioError, UnsupportedFormatError) as e:
                errors += 1
                error_log.append((filepath, str(e)))
                continue

            faiss_row = index.ntotal
            index.add(np.expand_dims(vector, 0))
            id_map[str(faiss_row)] = existing["id"]
            db.update_song_hash(existing["id"], current_hash)
            added += 1
            continue

        if existing and not update_only:
            # Full rebuild — existing row will be re-inserted below;
            # but we need to skip duplicate inserts. Just re-embed.
            pass

        # New file — extract embedding + metadata
        try:
            vector = get_embedding(filepath)
        except (AudioLoadError, AudioTooShortError, SilentAudioError, UnsupportedFormatError) as e:
            errors += 1
            error_log.append((filepath, str(e)))
            continue
        except Exception as e:
            errors += 1
            error_log.append((filepath, f"Unexpected: {e}"))
            continue

        meta      = _extract_metadata(filepath)
        file_hash = db.hash_file(filepath)

        if existing and not update_only:
            # Full rebuild: update hash, reuse id
            db.update_song_hash(existing["id"], file_hash)
            song_id = existing["id"]
        else:
            try:
                song_id = db.insert_song(
                    filepath  = filepath,
                    file_hash = file_hash,
                    title     = meta["title"],
                    artist    = meta["artist"],
                    album     = meta["album"],
                    duration  = meta["duration"],
                )
            except Exception as e:
                errors += 1
                error_log.append((filepath, f"DB insert failed: {e}"))
                continue

        faiss_row = index.ntotal
        index.add(np.expand_dims(vector, 0))   # shape must be (1, EMBEDDING_DIM)
        id_map[str(faiss_row)] = song_id
        added += 1

    # Persist
    _save_index(index)
    _save_id_map(id_map)

    # Summary
    print(f"\n[indexer] Done.")
    print(f"  Added/updated : {added}")
    print(f"  Skipped       : {skipped}")
    print(f"  Errors        : {errors}")
    print(f"  Total in index: {index.ntotal}")

    if error_log:
        print(f"\n[indexer] Files that failed:")
        for path, reason in error_log:
            print(f"  {os.path.basename(path)}: {reason}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index your music library for similarity search.")
    parser.add_argument("music_dir", help="Path to your music folder")
    parser.add_argument(
        "--update",
        action="store_true",
        help="Only process new or changed files (faster for incremental updates)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove DB entries for files that no longer exist, then exit",
    )
    args = parser.parse_args()

    if args.clean:
        db.init_db()
        removed = clean_stale_entries()
        print(f"Cleaned {removed} stale entries. Run without --clean to re-index.")
        sys.exit(0)

    run_index(music_dir=args.music_dir, update_only=args.update)
