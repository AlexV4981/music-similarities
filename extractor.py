"""
extractor.py — CLAP embedding pipeline

Loads the CLAP model once at module level (singleton pattern) so Flask
and the indexer both reuse the same loaded weights instead of loading
them fresh on every call.

Usage:
    from extractor import get_embedding, AudioLoadError, AudioTooShortError, SilentAudioError
    vector = get_embedding("/path/to/song.mp3")   # returns np.ndarray shape (512,)
"""

import os
import subprocess
import tempfile
import threading
import numpy as np
import librosa

from config import (
    CLAP_MODEL_ID,
    SAMPLE_RATE,
    MAX_DURATION_SECONDS,
    MIN_DURATION_SECONDS,
    MIN_RMS_ENERGY,
    NEEDS_CONVERSION,
    SUPPORTED_FORMATS,
    EMBEDDING_DIM,
)

# ---------------------------------------------------------------------------
# Custom exceptions — callers catch these for clean error messages
# ---------------------------------------------------------------------------

class AudioLoadError(Exception):
    """File could not be decoded as audio."""

class AudioTooShortError(Exception):
    """Audio clip is below the minimum duration threshold."""

class SilentAudioError(Exception):
    """Audio is essentially silent / blank."""

class UnsupportedFormatError(Exception):
    """File extension is not in the supported list."""


# ---------------------------------------------------------------------------
# Model singleton — load once, reuse everywhere
# ---------------------------------------------------------------------------

_model = None
_processor = None
_model_lock = threading.Lock()   # guards concurrent load attempts


def _load_model():
    """Lazy-load CLAP model and processor. Thread-safe."""
    global _model, _processor
    if _model is not None:
        return

    with _model_lock:
        if _model is not None:   # double-checked locking
            return
        try:
            from transformers import ClapModel, ClapProcessor
        except ImportError:
            raise RuntimeError(
                "transformers package not installed. Run: pip install transformers torch torchaudio"
            )

        print(f"[extractor] Loading CLAP model '{CLAP_MODEL_ID}' — this takes ~30s on first run...")
        _processor = ClapProcessor.from_pretrained(CLAP_MODEL_ID)
        _model = ClapModel.from_pretrained(CLAP_MODEL_ID)
        _model.eval()   # inference mode, no dropout
        print("[extractor] CLAP model loaded.")


# ---------------------------------------------------------------------------
# Audio loading helpers
# ---------------------------------------------------------------------------

def _convert_to_wav(filepath: str) -> str:
    """
    Use ffmpeg to convert an unsupported format to a temp WAV file.
    Returns the path to the temp file — caller is responsible for cleanup.
    Raises AudioLoadError if ffmpeg fails or is not installed.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", filepath, "-ar", str(SAMPLE_RATE), "-ac", "1", tmp.name],
            capture_output=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise AudioLoadError(
                f"ffmpeg conversion failed for '{filepath}': {result.stderr.decode()[:200]}"
            )
    except FileNotFoundError:
        raise AudioLoadError(
            "ffmpeg is not installed. Install it with: sudo apt install ffmpeg"
        )
    return tmp.name


def _load_audio(filepath: str) -> np.ndarray:
    """
    Load audio file to a mono float32 numpy array at SAMPLE_RATE.

    Handles:
    - Format conversion via ffmpeg for NEEDS_CONVERSION types
    - Capping duration at MAX_DURATION_SECONDS
    - Minimum duration check
    - Silent audio check

    Returns np.ndarray of shape (n_samples,)
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext not in SUPPORTED_FORMATS:
        raise UnsupportedFormatError(
            f"Format '{ext}' is not supported. Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )

    tmp_path = None
    load_path = filepath

    try:
        # Convert formats that librosa struggles with
        if ext in NEEDS_CONVERSION:
            tmp_path = _convert_to_wav(filepath)
            load_path = tmp_path

        try:
            audio, sr = librosa.load(
                load_path,
                sr=SAMPLE_RATE,
                mono=True,
                duration=MAX_DURATION_SECONDS,
            )
        except Exception as e:
            raise AudioLoadError(f"Could not decode audio file '{filepath}': {e}")

        # Duration check
        duration = len(audio) / SAMPLE_RATE
        if duration < MIN_DURATION_SECONDS:
            raise AudioTooShortError(
                f"Audio is only {duration:.1f}s — minimum is {MIN_DURATION_SECONDS}s"
            )

        # Silence check
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < MIN_RMS_ENERGY:
            raise SilentAudioError(
                f"Audio appears to be silent (RMS energy: {rms:.2e})"
            )

        return audio.astype(np.float32)

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def get_embedding(filepath: str) -> np.ndarray:
    """
    Extract a CLAP audio embedding from the given file.

    Args:
        filepath: Absolute path to an audio file.

    Returns:
        np.ndarray of shape (EMBEDDING_DIM,) — L2-normalised float32 vector.

    Raises:
        AudioLoadError, AudioTooShortError, SilentAudioError, UnsupportedFormatError
        on bad input; RuntimeError if the model can't be loaded.
    """
    import torch

    _load_model()

    audio = _load_audio(filepath)

    # CLAP processor expects a list of waveforms + the sample rate
    # newer transformers uses `audio`, older used `audios` — try both
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

    with torch.no_grad():
        embedding = _model.get_audio_features(**inputs)   # shape: (1, EMBEDDING_DIM)

    vector = embedding[0].cpu().numpy().astype(np.float32)

    # L2-normalise so cosine similarity == dot product (required for FAISS IndexFlatIP)
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm

    # Sanity check
    if vector.shape[0] != EMBEDDING_DIM:
        raise RuntimeError(
            f"Unexpected embedding dimension: got {vector.shape[0]}, expected {EMBEDDING_DIM}"
        )
    if np.isnan(vector).any():
        raise RuntimeError(f"Embedding contains NaN values for file: {filepath}")

    return vector


# ---------------------------------------------------------------------------
# CLI test — run directly to verify a single file works
# Usage: python extractor.py /path/to/song.mp3
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
        print(f"  Shape:     {vec.shape}")
        print(f"  Dtype:     {vec.dtype}")
        print(f"  L2 norm:   {np.linalg.norm(vec):.6f}  (should be ~1.0)")
        print(f"  Min/Max:   {vec.min():.4f} / {vec.max():.4f}")
        print(f"  Any NaN:   {np.isnan(vec).any()}")
        print("  PASS — embedding looks valid.")
    except (AudioLoadError, AudioTooShortError, SilentAudioError, UnsupportedFormatError) as e:
        print(f"  FAIL (expected error type): {e}")
        sys.exit(1)
    except Exception as e:
        print(f"  FAIL (unexpected): {e}")
        sys.exit(2)
