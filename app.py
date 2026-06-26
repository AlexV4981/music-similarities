"""
app.py — Flask API

Endpoints:
    POST /upload        receive audio file, extract embedding, return similar songs
    GET  /library       all indexed songs with metadata
    GET  /status        index health check
    POST /reindex       trigger re-index of the music library
"""

import os
import threading
import uuid
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

import db
import similarity
from valence import format_key
from config import (
    FLASK_PORT,
    MAX_UPLOAD_BYTES,
    SUPPORTED_FORMATS,
    UPLOAD_DIR,
    DEFAULT_TOP_N,
)
from extractor import (
    AudioLoadError,
    AudioTooShortError,
    SilentAudioError,
    UnsupportedFormatError,
    get_embedding,
)
from similarity import IndexNotReadyError

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

# One lock so concurrent uploads don't fight over the CLAP model
_extraction_lock = threading.Lock()

# Re-index state (runs in background thread)
_reindex_state = {
    "running": False,
    "last_result": None,   # "success" | "error: <msg>"
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in SUPPORTED_FORMATS


def _error(msg: str, code: int = 400):
    return jsonify({"error": msg}), code


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    """Serve the frontend with the correct API URL injected."""
    import socket
    # Get the host the browser used to reach this page
    # so the JS points back to the same machine, not hardcoded localhost
    host = request.host  # e.g. "192.168.1.45:5000" or "localhost:5000"
    api_url = f"{request.scheme}://{host}"

    frontend_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "frontend", "index.html"
    )
    with open(frontend_path, "r") as f:
        html = f.read()

    # Replace the hardcoded localhost API URL with the actual server address
    html = html.replace(
        "const API = 'http://localhost:5000'",
        f"const API = '{api_url}'"
    )
    return html, 200, {"Content-Type": "text/html"}

@app.route("/status", methods=["GET"])
def status():
    """
    Returns index health: whether it exists, how many songs, DB count.
    Frontend checks this on load to warn the user if no index exists.
    """
    db.init_db()
    db_count    = db.get_song_count()
    index_ready = similarity.is_ready()
    index_size  = similarity.index_size()

    in_sync = True
    if index_ready and index_size > 0:
        in_sync = abs(index_size - db_count) <= 2   # allow tiny drift

    return jsonify({
        "index_ready":   index_ready,
        "index_size":    index_size,
        "db_song_count": db_count,
        "in_sync":       in_sync,
        "reindex_running": _reindex_state["running"],
        "last_reindex_result": _reindex_state["last_result"],
    })


@app.route("/library", methods=["GET"])
def library():
    """Return all indexed songs, ordered by artist then title."""
    db.init_db()
    songs = db.get_all_songs()
    return jsonify({
        "count": len(songs),
        "songs": [
            {
                "id":          s["id"],
                "title":       s["title"],
                "artist":      s["artist"],
                "album":       s["album"],
                "duration":    round(s["duration"], 1),
                "valence":     round(s["valence"], 3) if s["valence"] is not None else None,
                "energy":      round(s["energy"], 3)  if s["energy"]  is not None else None,
                "danceability":round(s["danceability"], 3) if s["danceability"] is not None else None,
                "bpm":         round(s["bpm"], 1)     if s["bpm"]     is not None else None,
                "key_label":   format_key(s["key"], s["mode"]) if s["key"] is not None else None,
            }
            for s in songs
        ],
    })


@app.route("/upload", methods=["POST"])
def upload():
    """
    Receive an audio file, extract its CLAP embedding,
    search the index, and return ranked similar songs.

    Form fields:
        file    — the audio file (required)
        top_n   — number of results to return (optional, default DEFAULT_TOP_N)
    """
    # --- Validate index exists before doing expensive work ---
    if not similarity.is_ready():
        return _error(
            "No index found. Run the indexer on your music library first.", 503
        )

    # --- File presence check ---
    if "file" not in request.files:
        return _error("No file field in request.")
    f = request.files["file"]
    if not f or f.filename == "":
        return _error("No file selected.")

    # --- Extension check ---
    if not _allowed(f.filename):
        return _error(
            f"Unsupported format '{Path(f.filename).suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )

    # --- top_n param ---
    try:
        top_n = int(request.form.get("top_n", DEFAULT_TOP_N))
        top_n = max(1, min(top_n, 50))
    except ValueError:
        top_n = DEFAULT_TOP_N

    # --- Save to temp file ---
    safe_name = f"{uuid.uuid4().hex}_{secure_filename(f.filename)}"
    tmp_path  = os.path.join(UPLOAD_DIR, safe_name)
    try:
        f.save(tmp_path)
    except Exception as e:
        return _error(f"Failed to save uploaded file: {e}", 500)

    # --- Extract embedding (serialised via lock — CLAP not thread-safe) ---
    try:
        with _extraction_lock:
            vector = get_embedding(tmp_path)
    except UnsupportedFormatError as e:
        return _error(str(e))
    except AudioTooShortError as e:
        return _error(str(e))
    except SilentAudioError as e:
        return _error(str(e))
    except AudioLoadError as e:
        return _error(str(e))
    except Exception as e:
        return _error(f"Extraction failed unexpectedly: {e}", 500)
    finally:
        # Always clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # --- Search ---
    try:
        results = similarity.find_similar(vector, top_n=top_n)
    except IndexNotReadyError as e:
        return _error(str(e), 503)
    except Exception as e:
        return _error(f"Search failed: {e}", 500)

    any_low_confidence = any(r.low_confidence for r in results)

    return jsonify({
        "query_filename": f.filename,
        "result_count":   len(results),
        "low_confidence_warning": any_low_confidence,
        "results": [r.to_dict() for r in results],
    })


@app.route("/taste", methods=["POST"])
def taste():
    """
    Multi-song taste profile search.
    Accepts 2-5 songs, embeds each one, averages the vectors into a
    single centroid, then searches the index against that centroid.
    This gives a much better signal of actual taste than a single song.

    Form fields:
        files   — 2-5 audio files (field name: "files")
        top_n   — number of results (optional, default DEFAULT_TOP_N)
    """
    if not similarity.is_ready():
        return _error("No index found. Run the indexer first.", 503)

    files = request.files.getlist("files")
    if not files or len(files) < 2:
        return _error("Upload at least 2 songs to build a taste profile.")
    if len(files) > 5:
        return _error("Maximum 5 songs for a taste profile.")

    for f in files:
        if not _allowed(f.filename):
            return _error(f"Unsupported format in '{f.filename}'.")

    try:
        top_n = int(request.form.get("top_n", DEFAULT_TOP_N))
        top_n = max(1, min(top_n, 50))
    except ValueError:
        top_n = DEFAULT_TOP_N

    # Save all uploads to temp files
    tmp_paths = []
    for f in files:
        safe_name = f"{uuid.uuid4().hex}_{secure_filename(f.filename)}"
        tmp_path  = os.path.join(UPLOAD_DIR, safe_name)
        f.save(tmp_path)
        tmp_paths.append((f.filename, tmp_path))

    vectors    = []
    failed     = []
    succeeded  = []

    try:
        with _extraction_lock:
            for filename, tmp_path in tmp_paths:
                try:
                    vec = get_embedding(tmp_path)
                    vectors.append(vec)
                    succeeded.append(filename)
                except (AudioLoadError, AudioTooShortError,
                        SilentAudioError, UnsupportedFormatError) as e:
                    failed.append({"file": filename, "reason": str(e)})
                except Exception as e:
                    failed.append({"file": filename, "reason": f"Unexpected: {e}"})
    finally:
        for _, tmp_path in tmp_paths:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    if len(vectors) < 2:
        return _error(
            f"Not enough songs embedded successfully (need 2, got {len(vectors)}). "
            f"Failures: {[f['reason'] for f in failed]}"
        )

    # Average all vectors into a single taste centroid then re-normalise
    # so it sits on the unit sphere and cosine similarity still works correctly
    import numpy as np
    centroid = np.mean(np.stack(vectors), axis=0).astype(np.float32)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm

    try:
        results = similarity.find_similar(centroid, top_n=top_n)
    except IndexNotReadyError as e:
        return _error(str(e), 503)
    except Exception as e:
        return _error(f"Search failed: {e}", 500)

    any_low_confidence = any(r.low_confidence for r in results)

    return jsonify({
        "mode":               "taste_profile",
        "songs_used":         succeeded,
        "songs_failed":       failed,
        "result_count":       len(results),
        "low_confidence_warning": any_low_confidence,
        "results":            [r.to_dict() for r in results],
    })


@app.route("/reindex", methods=["POST"])
def reindex():
    """
    Trigger a background re-index of the music library.
    Expects JSON body: { "music_dir": "/path/to/music", "update": true/false }
    Returns immediately; poll /status for completion.
    """
    if _reindex_state["running"]:
        return _error("Re-index already in progress.", 409)

    body = request.get_json(silent=True) or {}
    music_dir = body.get("music_dir", "").strip()
    update_only = bool(body.get("update", True))

    if not music_dir:
        return _error("music_dir is required in the request body.")
    if not os.path.isdir(music_dir):
        return _error(f"'{music_dir}' is not a valid directory.")

    def _run():
        _reindex_state["running"] = True
        _reindex_state["last_result"] = None
        try:
            from indexer import run_index
            run_index(music_dir=music_dir, update_only=update_only)
            similarity.reload()   # hot-reload the new index
            _reindex_state["last_result"] = "success"
        except Exception as e:
            _reindex_state["last_result"] = f"error: {e}"
        finally:
            _reindex_state["running"] = False

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return jsonify({
        "message": f"Re-index started for '{music_dir}' (update={update_only}). Poll /status for progress."
    }), 202


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(413)
def too_large(_):
    mb = MAX_UPLOAD_BYTES // (1024 * 1024)
    return _error(f"File too large. Maximum upload size is {mb}MB.", 413)


@app.errorhandler(404)
def not_found(_):
    return _error("Endpoint not found.", 404)


@app.errorhandler(405)
def method_not_allowed(_):
    return _error("Method not allowed.", 405)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _startup_checks():
    """Run once at startup regardless of server (Flask dev or Gunicorn)."""
    db.init_db()
    print(f"[app] Index ready: {similarity.is_ready()}")
    print(f"[app] Songs in DB: {db.get_song_count()}")

_startup_checks()

if __name__ == "__main__":
    # Direct python run — dev only, use start.sh for production (Gunicorn)
    print(f"[app] Starting Flask dev server on http://localhost:{FLASK_PORT}")
    print(f"[app] For better performance run: bash start.sh")
    app.run(host="0.0.0.0", port=FLASK_PORT, debug=False)
