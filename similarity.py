"""
similarity.py — FAISS-backed similarity search

Loads the FAISS index once at startup (singleton) and exposes
find_similar() which takes a query embedding and returns the
top N most similar songs with metadata from SQLite.

FAISS IndexFlatIP + L2-normalised vectors = cosine similarity.
Scores range 0.0 - 1.0 (1.0 = identical).
"""

import json
import os
import threading
from dataclasses import dataclass

import faiss
import numpy as np

import db
from config import (
    DEFAULT_TOP_N,
    EMBEDDING_DIM,
    FAISS_ID_MAP_PATH,
    FAISS_INDEX_PATH,
    LOW_CONFIDENCE_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    rank:           int
    song_id:        int
    filepath:       str
    title:          str
    artist:         str
    album:          str
    duration:       float
    score:          float   # cosine similarity 0.0 - 1.0
    low_confidence: bool    # True if score < LOW_CONFIDENCE_THRESHOLD

    def to_dict(self) -> dict:
        return {
            "rank":           self.rank,
            "song_id":        self.song_id,
            "title":          self.title,
            "artist":         self.artist,
            "album":          self.album,
            "duration":       round(self.duration, 1),
            "score":          round(self.score, 4),
            "score_pct":      f"{self.score * 100:.1f}%",
            "low_confidence": self.low_confidence,
            "filepath":       self.filepath,
        }


# ---------------------------------------------------------------------------
# Index singleton
# ---------------------------------------------------------------------------

_index:   faiss.IndexFlatIP | None = None
_id_map:  dict                     = {}   # str(faiss_row) -> db_song_id
_index_lock = threading.Lock()


class IndexNotReadyError(Exception):
    """Raised when FAISS index hasn't been built yet."""


def _load() -> None:
    """Load FAISS index and id_map from disk. Thread-safe."""
    global _index, _id_map

    with _index_lock:
        if _index is not None:
            return

        if not os.path.exists(FAISS_INDEX_PATH):
            raise IndexNotReadyError(
                "FAISS index not found. Run the indexer first:\n"
                "  python indexer.py /path/to/music"
            )
        if not os.path.exists(FAISS_ID_MAP_PATH):
            raise IndexNotReadyError(
                "id_map.json not found. Re-run the indexer to rebuild it."
            )

        _index = faiss.read_index(FAISS_INDEX_PATH)

        with open(FAISS_ID_MAP_PATH) as f:
            _id_map = json.load(f)

        # Sanity check: FAISS count vs id_map count
        if _index.ntotal != len(_id_map):
            print(
                f"[similarity] WARNING: FAISS has {_index.ntotal} vectors "
                f"but id_map has {len(_id_map)} entries. "
                "Index may be out of sync — consider re-running the indexer."
            )


def reload() -> None:
    """Force reload of the index from disk (call after re-indexing)."""
    global _index, _id_map
    with _index_lock:
        _index  = None
        _id_map = {}
    _load()


def index_size() -> int:
    """Return number of vectors in the loaded index, or 0 if not loaded."""
    return _index.ntotal if _index is not None else 0


def is_ready() -> bool:
    return os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_ID_MAP_PATH)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def find_similar(
    query_vector: np.ndarray,
    top_n: int = DEFAULT_TOP_N,
    exclude_filepath: str | None = None,
) -> list[SearchResult]:
    """
    Find the top_n most similar songs to a query embedding.

    Args:
        query_vector:     L2-normalised float32 ndarray of shape (EMBEDDING_DIM,)
        top_n:            Number of results to return
        exclude_filepath: Optional filepath to exclude from results
                          (used when the uploaded song is already in the library
                          so it doesn't return itself as #1)

    Returns:
        List of SearchResult sorted by score descending.

    Raises:
        IndexNotReadyError if the index hasn't been built.
        ValueError on bad query vector shape/dtype.
    """
    _load()

    # Input validation
    if query_vector.ndim != 1 or query_vector.shape[0] != EMBEDDING_DIM:
        raise ValueError(
            f"query_vector must be shape ({EMBEDDING_DIM},), "
            f"got {query_vector.shape}"
        )
    if query_vector.dtype != np.float32:
        query_vector = query_vector.astype(np.float32)

    # Re-normalise defensively (extractor already does this, but be safe)
    norm = np.linalg.norm(query_vector)
    if norm > 0:
        query_vector = query_vector / norm

    # FAISS expects shape (n_queries, dim)
    q = np.expand_dims(query_vector, 0)

    # Fetch extra results in case we need to filter out the query song itself
    fetch_n = min(top_n + 5, _index.ntotal)
    scores, faiss_rows = _index.search(q, fetch_n)

    results = []
    for score, faiss_row in zip(scores[0], faiss_rows[0]):
        if faiss_row == -1:
            continue   # FAISS pads with -1 when fewer results than requested

        db_id = _id_map.get(str(faiss_row))
        if db_id is None:
            continue   # id_map out of sync — skip

        song = db.get_song_by_id(db_id)
        if song is None:
            continue   # DB row deleted since last index load

        # Filter out the query song if it's in the library
        if exclude_filepath and os.path.abspath(song["filepath"]) == os.path.abspath(exclude_filepath):
            continue

        # Clamp score to [0, 1] — floating point can give tiny negatives
        clamped_score = float(max(0.0, min(1.0, score)))

        results.append(SearchResult(
            rank           = len(results) + 1,
            song_id        = db_id,
            filepath       = song["filepath"],
            title          = song["title"],
            artist         = song["artist"],
            album          = song["album"],
            duration       = song["duration"],
            score          = clamped_score,
            low_confidence = clamped_score < LOW_CONFIDENCE_THRESHOLD,
        ))

        if len(results) >= top_n:
            break

    # Re-assign ranks after filtering
    for i, r in enumerate(results):
        r.rank = i + 1

    return results
