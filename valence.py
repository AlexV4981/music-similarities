"""
valence.py — Local valence and energy estimation using librosa

Valence (emotional positivity 0-1):
  Derived from key mode (major=positive, minor=negative), spectral brightness
  (high freq content = brighter = happier), and chroma harmony.
  Approximates Spotify's valence feature using only local audio analysis.

Energy (intensity 0-1):
  RMS loudness + spectral flux + onset rate combined.
  High energy = loud, fast, active. Low energy = quiet, slow, calm.

Danceability (0-1):
  Tempo proximity to 120-130 BPM + beat strength + rhythm regularity.
  Based on Spotify's documented danceability formula components.

Key + Mode:
  Detected via librosa chroma + key estimation.
  Mode: 1 = major (tends brighter/happier), 0 = minor (tends darker/sadder).

All values are stored in the DB at index time so search is instant.
"""

import numpy as np
import librosa


def extract_features(audio: np.ndarray, sr: int) -> dict:
    """
    Extract valence, energy, danceability, key, mode, and BPM from audio.

    Args:
        audio: mono float32 numpy array
        sr:    sample rate

    Returns dict with keys:
        valence      float  0.0–1.0  emotional positivity
        energy       float  0.0–1.0  intensity / loudness
        danceability float  0.0–1.0  how groovy / danceable
        bpm          float  beats per minute
        key          int    0–11 (C=0, C#=1 … B=11)
        mode         int    1=major, 0=minor
    """
    # Use a representative middle chunk — avoids silent intros/outros
    # biasing the feature estimates
    duration   = len(audio) / sr
    skip       = int(min(15.0, duration * 0.1) * sr)
    end        = len(audio) - skip
    chunk      = audio[skip:end] if end > skip else audio
    if len(chunk) < sr * 2:   # fallback if too short after trimming
        chunk = audio

    # ── BPM ──────────────────────────────────────────────────────────────────
    tempo, beats = librosa.beat.beat_track(y=chunk, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])

    # ── Key and mode ─────────────────────────────────────────────────────────
    # Chromagram → sum energy per pitch class → pick strongest as key
    chroma      = librosa.feature.chroma_cqt(y=chunk, sr=sr)
    chroma_mean = chroma.mean(axis=1)   # shape (12,)

    # Major and minor key profiles (Krumhansl-Kessler)
    major_profile = np.array([6.35,2.23,3.48,2.33,4.38,4.09,
                               2.52,5.19,2.39,3.66,2.29,2.88])
    minor_profile = np.array([6.33,2.68,3.52,5.38,2.60,3.53,
                               2.54,4.75,3.98,2.69,3.34,3.17])

    major_scores = np.array([
        np.corrcoef(np.roll(chroma_mean, -i), major_profile)[0,1]
        for i in range(12)
    ])
    minor_scores = np.array([
        np.corrcoef(np.roll(chroma_mean, -i), minor_profile)[0,1]
        for i in range(12)
    ])

    best_major = int(np.argmax(major_scores))
    best_minor = int(np.argmax(minor_scores))

    if major_scores[best_major] >= minor_scores[best_minor]:
        key  = best_major
        mode = 1   # major
        key_confidence = float(major_scores[best_major])
    else:
        key  = best_minor
        mode = 0   # minor
        key_confidence = float(minor_scores[best_minor])

    # ── Energy ───────────────────────────────────────────────────────────────
    # Combines RMS loudness + spectral flux (rate of change in spectrum)
    rms         = librosa.feature.rms(y=chunk)[0]
    rms_mean    = float(np.mean(rms))

    stft        = np.abs(librosa.stft(chunk))
    flux        = np.mean(np.diff(stft, axis=1) ** 2)

    # Onset density (events per second) — more onsets = more active
    onsets      = librosa.onset.onset_detect(y=chunk, sr=sr)
    onset_rate  = len(onsets) / (len(chunk) / sr)

    # Normalise each component then blend
    # RMS typically 0–0.3 for music, flux and onset_rate need soft caps
    rms_norm    = float(np.clip(rms_mean / 0.25, 0, 1))
    flux_norm   = float(np.clip(flux / 1e5, 0, 1))
    onset_norm  = float(np.clip(onset_rate / 8.0, 0, 1))

    energy = float(np.clip(
        0.5 * rms_norm + 0.3 * flux_norm + 0.2 * onset_norm, 0, 1
    ))

    # ── Valence ───────────────────────────────────────────────────────────────
    # Major key → positive valence base, minor → negative base
    mode_score  = 0.65 if mode == 1 else 0.35

    # Spectral brightness: ratio of high-freq to low-freq energy
    # Brighter = more high frequencies = generally happier sounding
    spectral_centroid = librosa.feature.spectral_centroid(y=chunk, sr=sr)[0]
    centroid_mean     = float(np.mean(spectral_centroid))
    # Typical music centroid range: 500–4000 Hz
    brightness        = float(np.clip((centroid_mean - 500) / 3500, 0, 1))

    # Harmonic-to-noise ratio proxy: harmonic content = more consonant = more positive
    harmonic, _ = librosa.effects.hpss(chunk)
    hnr         = float(np.clip(
        np.mean(harmonic ** 2) / (np.mean(chunk ** 2) + 1e-8), 0, 1
    ))

    # Chroma consonance: how much energy sits on stable intervals (3rds, 5ths)
    # Stable intervals in semitones from root: 0,4,7 (major triad) or 0,3,7 (minor)
    if mode == 1:
        consonant_intervals = [0, 4, 7]
    else:
        consonant_intervals = [0, 3, 7]
    root_chroma = np.roll(chroma_mean, -key)
    consonance  = float(np.sum(root_chroma[consonant_intervals]) /
                        (np.sum(root_chroma) + 1e-8))
    consonance  = float(np.clip(consonance, 0, 1))

    # Blend: mode is the strongest signal, brightness and consonance refine it
    valence = float(np.clip(
        0.50 * mode_score +
        0.25 * brightness +
        0.15 * consonance +
        0.10 * hnr,
        0, 1
    ))

    # ── Danceability ──────────────────────────────────────────────────────────
    # Tempo proximity to sweet spot (120-130 BPM) + beat strength + regularity

    # BPM score — peaks at 125 BPM, falls off on either side
    bpm_score = float(np.clip(1.0 - abs(bpm - 125) / 60.0, 0, 1))

    # Beat strength: how strong and consistent the detected beats are
    if len(beats) > 1:
        beat_intervals = np.diff(beats)
        regularity     = float(np.clip(
            1.0 - (np.std(beat_intervals) / (np.mean(beat_intervals) + 1e-8)),
            0, 1
        ))
    else:
        regularity = 0.0

    # Percussive energy ratio — more drums = more danceable
    _, percussive = librosa.effects.hpss(chunk)
    perc_ratio    = float(np.clip(
        np.mean(percussive ** 2) / (np.mean(chunk ** 2) + 1e-8), 0, 1
    ))

    danceability = float(np.clip(
        0.40 * bpm_score +
        0.35 * regularity +
        0.25 * perc_ratio,
        0, 1
    ))

    return {
        "valence":      round(valence,      4),
        "energy":       round(energy,       4),
        "danceability": round(danceability, 4),
        "bpm":          round(bpm,          2),
        "key":          key,
        "mode":         mode,
    }


# Key names for display
KEY_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

def format_key(key: int, mode: int) -> str:
    """Return human-readable key string e.g. 'A minor', 'C# major'"""
    return f"{KEY_NAMES[key % 12]} {'major' if mode == 1 else 'minor'}"
