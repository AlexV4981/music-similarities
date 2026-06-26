"""
extractor.py — CLAP embedding pipeline

Improvements over v1:
  1. torch.compile fix — import was missing at compile scope
  2. Audio peak normalisation — quiet/loud recordings of same song embed consistently
  3. Smart windowing — 3 x 30s windows at 20/50/75% through song, averaged
     captures verse+chorus+bridge instead of one averaged blob
  4. Chunk embedding for long songs — splits into 30s chunks, embeds each,
     averages; CLAP was trained on short clips not 5min blobs
  5. Embedding cache — keyed by MD5 hash, skips recomputation for repeated files
"""

import hashlib
import os
import subprocess
import tempfile
import threading
from collections import OrderedDict

import librosa
import numpy as np

from config import (
    CLAP_MODEL_ID,
    EMBEDDING_DIM,
    MAX_DURATION_SECONDS,
    MIN_DURATION_SECONDS,
    MIN_RMS_ENERGY,
    NEEDS_CONVERSION,
    SAMPLE_RATE,
    SUPPORTED_FORMATS,
)

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

WINDOW_SECONDS   = 30     # length of each analysis window
CHUNK_SECONDS    = 30     # chunk size for long-song embedding
WINDOW_POSITIONS = [0.20, 0.50, 0.75]   # where in the song to sample windows
PEAK_TARGET_DB   = -3.0  # normalise peaks to this level before embedding
CACHE_MAX_SIZE   = 64    # max embeddings held in memory cache


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class AudioLoadError(Exception):      pass
class AudioTooShortError(Exception):  pass
class SilentAudioError(Exception):    pass
class UnsupportedFormatError(Exception): pass


# ---------------------------------------------------------------------------
# Embedding cache — LRU, keyed by file MD5
# ---------------------------------------------------------------------------

_cache: OrderedDict = OrderedDict()
_cache_lock = threading.Lock()


def _cache_get(key: str) -> np.ndarray | None:
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key].copy()
    return None


def _cache_put(key: str, vector: np.ndarray) -> None:
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
        else:
            if len(_cache) >= CACHE_MAX_SIZE:
                _cache.popitem(last=False)   # evict oldest
            _cache[key] = vector.copy()


def _file_hash(filepath: str) -> str:
    """MD5 of first 1MB — fast enough for cache keying without reading whole file."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        h.update(f.read(1024 * 1024))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------

_model     = None
_processor = None
_model_lock = threading.Lock()


def _load_model() -> None:
    """Lazy-load CLAP. Thread-safe via double-checked locking."""
    global _model, _processor
    if _model is not None:
        return

    with _model_lock:
        if _model is not None:
            return

        import torch
        from transformers import ClapModel, ClapProcessor

        # Thread count — physical cores only, avoids hyperthreading contention
        cpu_cores = os.cpu_count() or 4
        torch.set_num_threads(max(1, cpu_cores // 2))
        torch.set_num_interop_threads(2)

        print(f"[extractor] Loading CLAP model '{CLAP_MODEL_ID}'…")
        print(f"[extractor] First run downloads ~900MB — progress shown below.")

        # local_files_only=False + no env override = HuggingFace shows tqdm bars
        # for each file being downloaded (config, tokenizer, weights, etc.)
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import are_progress_bars_disabled
        import huggingface_hub

        # Force progress bars on even if something disabled them
        huggingface_hub.utils.enable_progress_bars()

        _processor = ClapProcessor.from_pretrained(
            CLAP_MODEL_ID,
            local_files_only=False,
        )
        _model = ClapModel.from_pretrained(
            CLAP_MODEL_ID,
            local_files_only=False,
        )
        _model.eval()

        # Fix: torch must be imported at this scope for compile to work
        try:
            _model = torch.compile(_model, mode="reduce-overhead")
            print("[extractor] torch.compile active.")
        except Exception as e:
            print(f"[extractor] torch.compile skipped ({e}) — eager mode.")

        print("[extractor] CLAP model ready.")


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _convert_to_wav(filepath: str) -> str:
    """Convert unsupported format to temp WAV via ffmpeg."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", filepath,
             "-ar", str(SAMPLE_RATE), "-ac", "1", tmp.name],
            capture_output=True, timeout=120,
        )
        if result.returncode != 0:
            raise AudioLoadError(
                f"ffmpeg failed for '{filepath}': {result.stderr.decode()[:200]}"
            )
    except FileNotFoundError:
        raise AudioLoadError("ffmpeg not installed. Run: sudo apt install ffmpeg")
    return tmp.name


def _peak_normalise(audio: np.ndarray, target_db: float = PEAK_TARGET_DB) -> np.ndarray:
    """
    Normalise audio peak to target_db.
    Ensures quiet and loud recordings of the same song embed consistently.
    Skips if audio is silent to avoid divide-by-zero.
    """
    peak = np.max(np.abs(audio))
    if peak < 1e-8:
        return audio
    target_amplitude = 10 ** (target_db / 20.0)
    return (audio * (target_amplitude / peak)).astype(np.float32)


def _load_audio(filepath: str) -> np.ndarray:
    """
    Load audio file → mono float32 at SAMPLE_RATE.
    Applies format conversion, duration checks, silence check,
    and peak normalisation.
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in SUPPORTED_FORMATS:
        raise UnsupportedFormatError(
            f"Format '{ext}' not supported. Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )

    tmp_path  = None
    load_path = filepath

    try:
        if ext in NEEDS_CONVERSION:
            tmp_path  = _convert_to_wav(filepath)
            load_path = tmp_path

        try:
            audio, _ = librosa.load(
                load_path,
                sr=SAMPLE_RATE,
                mono=True,
                duration=MAX_DURATION_SECONDS,
            )
        except Exception as e:
            raise AudioLoadError(f"Could not decode '{filepath}': {e}")

        duration = len(audio) / SAMPLE_RATE
        if duration < MIN_DURATION_SECONDS:
            raise AudioTooShortError(
                f"Audio is {duration:.1f}s — minimum is {MIN_DURATION_SECONDS}s"
            )

        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < MIN_RMS_ENERGY:
            raise SilentAudioError(f"Audio is silent (RMS: {rms:.2e})")

        # Peak normalise — makes quiet/loud versions of same song embed similarly
        audio = _peak_normalise(audio)

        return audio.astype(np.float32)

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Core embedding — single audio array → normalised vector
# ---------------------------------------------------------------------------

def _embed_audio(audio: np.ndarray) -> np.ndarray:
    """
    Run CLAP on a raw audio array.
    Returns L2-normalised float32 vector of shape (EMBEDDING_DIM,).
    Audio must be at SAMPLE_RATE, mono, float32.
    """
    import torch

    _load_model()

    try:
        inputs = _processor(
            audio=[audio],
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        )
    except TypeError:
        inputs = _processor(
            audios=[audio],
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        )

    audio_keys = {k: v for k, v in inputs.items()
                  if k in ("input_features", "is_longer")}

    with torch.inference_mode():
        output = _model.get_audio_features(**audio_keys)

    if isinstance(output, torch.Tensor):
        tensor = output
    elif hasattr(output, "pooler_output") and output.pooler_output is not None:
        tensor = output.pooler_output
    elif hasattr(output, "last_hidden_state"):
        tensor = output.last_hidden_state.mean(dim=1)
    else:
        raise RuntimeError(f"Unrecognised model output type: {type(output)}")

    vector = tensor.squeeze().cpu().numpy().astype(np.float32)
    if vector.ndim > 1:
        vector = vector[0]
    if vector.ndim == 0:
        raise RuntimeError("Embedding collapsed to scalar")

    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm

    if vector.shape[0] != EMBEDDING_DIM:
        raise RuntimeError(
            f"Wrong embedding dim: got {vector.shape[0]}, expected {EMBEDDING_DIM}"
        )
    if np.isnan(vector).any():
        raise RuntimeError("Embedding contains NaN")

    return vector


# ---------------------------------------------------------------------------
# Windowed embedding — the main quality improvement
#
# Strategy (based on MIR best practices):
#   Short songs  (< 2x WINDOW_SECONDS): embed whole song as one chunk
#   Medium songs (< 4x WINDOW_SECONDS): embed 3 fixed windows at 20/50/75%
#   Long songs   (>= 4x WINDOW_SECONDS): chunk into CHUNK_SECONDS pieces,
#                                         embed each, average
#
# All per-window embeddings are averaged then re-normalised.
# This captures verse + chorus + bridge instead of one averaged blob,
# and avoids feeding CLAP audio it wasn't trained on (>60s clips).
# ---------------------------------------------------------------------------

def _windowed_embed(audio: np.ndarray) -> np.ndarray:
    """
    Extract a taste-representative embedding by sampling multiple
    windows across the song rather than embedding the full audio blob.
    """
    duration    = len(audio) / SAMPLE_RATE
    window_samp = int(WINDOW_SECONDS * SAMPLE_RATE)
    chunk_samp  = int(CHUNK_SECONDS  * SAMPLE_RATE)

    # ── Short song: embed whole thing ────────────────────────────────────────
    if duration < WINDOW_SECONDS * 2:
        return _embed_audio(audio)

    # ── Medium song: 3 strategic windows ─────────────────────────────────────
    if duration < WINDOW_SECONDS * 4:
        vectors = []
        for pos in WINDOW_POSITIONS:
            start = int(pos * len(audio))
            end   = min(start + window_samp, len(audio))
            chunk = audio[start:end]
            # Skip windows that are too short or silent
            if len(chunk) / SAMPLE_RATE < MIN_DURATION_SECONDS:
                continue
            if np.sqrt(np.mean(chunk ** 2)) < MIN_RMS_ENERGY:
                continue
            vectors.append(_embed_audio(chunk))

        if not vectors:
            return _embed_audio(audio[:window_samp])

        centroid = np.mean(np.stack(vectors), axis=0).astype(np.float32)
        norm = np.linalg.norm(centroid)
        return centroid / norm if norm > 0 else centroid

    # ── Long song: chunk + average ────────────────────────────────────────────
    # Skip first and last 15s — usually silence/fade
    skip_samp = int(15 * SAMPLE_RATE)
    trimmed   = audio[skip_samp : len(audio) - skip_samp]

    chunks  = [trimmed[i:i + chunk_samp]
               for i in range(0, len(trimmed), chunk_samp)]
    vectors = []
    for chunk in chunks:
        if len(chunk) / SAMPLE_RATE < MIN_DURATION_SECONDS:
            continue
        if np.sqrt(np.mean(chunk ** 2)) < MIN_RMS_ENERGY:
            continue
        vectors.append(_embed_audio(chunk))

    if not vectors:
        return _embed_audio(audio[:window_samp])

    centroid = np.mean(np.stack(vectors), axis=0).astype(np.float32)
    norm = np.linalg.norm(centroid)
    return centroid / norm if norm > 0 else centroid


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_embedding(filepath: str) -> np.ndarray:
    """
    Extract a CLAP embedding from an audio file.

    Returns:
        np.ndarray shape (EMBEDDING_DIM,) — L2-normalised float32

    Raises:
        AudioLoadError, AudioTooShortError, SilentAudioError,
        UnsupportedFormatError, RuntimeError
    """
    # Check cache first — avoids recomputing for repeated files in taste mode
    try:
        cache_key = _file_hash(filepath)
        cached    = _cache_get(cache_key)
        if cached is not None:
            return cached
    except OSError:
        cache_key = None

    audio  = _load_audio(filepath)
    vector = _windowed_embed(audio)

    if cache_key:
        _cache_put(cache_key, vector)

    return vector


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python extractor.py <audio_file>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"Testing extractor on: {path}")
    try:
        vec = get_embedding(path)
        print(f"  Shape:    {vec.shape}")
        print(f"  Dtype:    {vec.dtype}")
        print(f"  L2 norm:  {np.linalg.norm(vec):.6f}  (should be ~1.0)")
        print(f"  Min/Max:  {vec.min():.4f} / {vec.max():.4f}")
        print(f"  Any NaN:  {np.isnan(vec).any()}")
        print("  PASS")
    except (AudioLoadError, AudioTooShortError,
            SilentAudioError, UnsupportedFormatError) as e:
        print(f"  FAIL (expected): {e}")
        sys.exit(1)
    except Exception as e:
        print(f"  FAIL (unexpected): {e}")
        sys.exit(2)
