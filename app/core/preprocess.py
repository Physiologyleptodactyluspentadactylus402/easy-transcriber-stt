"""Audio preprocessing utilities for the Transcriber app.

Provides FFmpeg-based helpers for:
- Decoding any audio format to a normalised mono WAV.
- Measuring integrated loudness (LUFS) via the EBU R128 loudnorm filter.
- Applying loudness normalisation (two-pass loudnorm) to a target LUFS level.

All FFmpeg calls are made via :mod:`subprocess` so that the system FFmpeg
installation is used — no Python audio library is required beyond the
standard library.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import soundfile as sf
    import torch
    import torchaudio
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    _DEMUCS_AVAILABLE = True
except ImportError:
    _DEMUCS_AVAILABLE = False

_demucs_model_cache = {}

logger = logging.getLogger("transcriber.preprocess")


def _require_ffmpeg() -> str:
    """Return the path to the ffmpeg executable, raising if not found."""
    path = shutil.which("ffmpeg")
    if path is None:
        raise RuntimeError(
            "ffmpeg not found on PATH. Please install FFmpeg: https://ffmpeg.org/download.html"
        )
    return path


# ---------------------------------------------------------------------------
# Task 1 helpers
# ---------------------------------------------------------------------------


def decode_to_wav(
    input_path: Path | str,
    output_path: Path | str,
    sample_rate: int = 48000,
) -> Path:
    """Decode *input_path* to a mono WAV file at *sample_rate* Hz.

    Uses FFmpeg to handle any audio format that FFmpeg understands (MP3, M4A,
    FLAC, OGG, …).  The output is always:
    - mono (``ac=1``)
    - PCM 16-bit little-endian (``pcm_s16le``)
    - sampled at *sample_rate* Hz (default 48 000 Hz)

    Parameters
    ----------
    input_path:
        Source audio file.  Must exist.
    output_path:
        Destination WAV path.  Parent directory must exist (or be creatable).
    sample_rate:
        Output sample rate in Hz.  Defaults to 48 000.

    Returns
    -------
    Path
        Resolved path of the written WAV file.

    Raises
    ------
    FileNotFoundError
        If *input_path* does not exist.
    RuntimeError
        If FFmpeg exits with a non-zero return code.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input audio file not found: {input_path}")

    ffmpeg = _require_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg,
        "-y",                     # overwrite without prompting
        "-i", str(input_path),
        "-ac", "1",               # mono
        "-ar", str(sample_rate),  # sample rate
        "-c:a", "pcm_s16le",      # 16-bit PCM
        str(output_path),
    ]
    logger.debug("decode_to_wav command: %s", " ".join(cmd))

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        stderr_text = result.stderr.decode(errors="replace")
        logger.error("FFmpeg decode failed (rc=%d):\n%s", result.returncode, stderr_text)
        raise RuntimeError(
            f"FFmpeg failed with return code {result.returncode}.\n{stderr_text}"
        )

    logger.info("Decoded %s → %s (%d Hz mono)", input_path.name, output_path.name, sample_rate)
    return output_path.resolve()


def analyze_lufs(wav_path: Path | str) -> dict[str, float]:
    """Measure the integrated loudness of *wav_path* using FFmpeg loudnorm pass 1.

    Runs FFmpeg with the ``loudnorm`` filter in *measurement-only* mode
    (``print_format=json``) and parses the JSON block that FFmpeg writes to
    stderr.

    Parameters
    ----------
    wav_path:
        Path to a WAV (or any FFmpeg-readable) audio file.

    Returns
    -------
    dict[str, float]
        Dictionary with keys:
        ``input_i``, ``input_tp``, ``input_lra``, ``input_thresh``,
        ``target_offset`` — all as Python :class:`float` values.

    Raises
    ------
    FileNotFoundError
        If *wav_path* does not exist.
    RuntimeError
        If FFmpeg exits with a non-zero return code or the JSON block
        cannot be found in its output.
    """
    wav_path = Path(wav_path)

    if not wav_path.exists():
        raise FileNotFoundError(f"WAV file not found: {wav_path}")

    ffmpeg = _require_ffmpeg()

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-i", str(wav_path),
        "-af", "loudnorm=I=-23:LRA=7:TP=-2:print_format=json",
        "-f", "null",
        "-",
    ]
    logger.debug("analyze_lufs command: %s", " ".join(cmd))

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # FFmpeg writes the loudnorm JSON to stderr regardless of return code on
    # some builds; check output before raising on rc.
    stderr_text = result.stderr.decode(errors="replace")

    # Extract the JSON block: FFmpeg prints it between a blank line and a closing brace.
    json_match = re.search(r"\{[^{}]*\}", stderr_text, re.DOTALL)
    if json_match is None:
        if result.returncode != 0:
            logger.error("FFmpeg loudnorm failed (rc=%d):\n%s", result.returncode, stderr_text)
            raise RuntimeError(
                f"FFmpeg failed with return code {result.returncode}.\n{stderr_text}"
            )
        raise RuntimeError(
            f"Could not find loudnorm JSON in FFmpeg output.\nstderr:\n{stderr_text}"
        )

    raw = json.loads(json_match.group())
    logger.debug("loudnorm raw JSON: %s", raw)

    keys = ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")
    stats: dict[str, float] = {}
    for key in keys:
        if key not in raw:
            raise RuntimeError(f"Missing key '{key}' in loudnorm JSON: {raw}")
        try:
            stats[key] = float(raw[key])
        except (ValueError, TypeError) as exc:
            raise RuntimeError(
                f"Cannot convert loudnorm key '{key}' value {raw[key]!r} to float"
            ) from exc

    logger.info(
        "LUFS analysis for %s: input_i=%.1f LUFS, input_tp=%.1f dBTP, input_lra=%.1f LU",
        wav_path.name,
        stats["input_i"],
        stats["input_tp"],
        stats["input_lra"],
    )
    return stats


# ---------------------------------------------------------------------------
# Task 2 — loudness normalisation
# ---------------------------------------------------------------------------


def apply_loudnorm(
    wav_path: Path | str,
    output_path: Path | str,
    target_lufs: float = -16.0,
    measured: Optional[dict[str, float]] = None,
) -> Path:
    """Normalise the loudness of *wav_path* to *target_lufs* using a two-pass loudnorm.

    If *measured* is ``None`` the function runs :func:`analyze_lufs` first
    (pass 1).  The actual normalisation (pass 2) is then applied with the
    measured stats so that FFmpeg can choose between *linear* and *dynamic*
    normalisation intelligently.

    Output characteristics:
    - 48 000 Hz sample rate
    - Mono (``ac=1``)
    - 16-bit PCM (``pcm_s16le``)

    Parameters
    ----------
    wav_path:
        Input WAV file.
    output_path:
        Destination WAV path.
    target_lufs:
        Target integrated loudness in LUFS.  Defaults to ``-16.0``.
    measured:
        Pre-computed stats dict from :func:`analyze_lufs`.  When supplied the
        internal pass-1 analysis is skipped.

    Returns
    -------
    Path
        Resolved path of the normalised WAV file.

    Raises
    ------
    FileNotFoundError
        If *wav_path* does not exist.
    RuntimeError
        If FFmpeg exits with a non-zero return code.
    """
    wav_path = Path(wav_path)
    output_path = Path(output_path)

    if not wav_path.exists():
        raise FileNotFoundError(f"Input WAV file not found: {wav_path}")

    # Pass 1 — measure if not provided
    if measured is None:
        logger.info("apply_loudnorm: running pass-1 analysis on %s", wav_path.name)
        measured = analyze_lufs(wav_path)

    ffmpeg = _require_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build the loudnorm filter string with measured stats (pass 2).
    # Using linear=true when the signal is well-behaved; FFmpeg will fall back
    # to dynamic mode if linear is not possible given the LRA.
    loudnorm_filter = (
        f"loudnorm="
        f"I={target_lufs}:"
        f"LRA=7:"
        f"TP=-1.5:"
        f"measured_I={measured['input_i']}:"
        f"measured_LRA={measured['input_lra']}:"
        f"measured_TP={measured['input_tp']}:"
        f"measured_thresh={measured['input_thresh']}:"
        f"offset={measured['target_offset']}:"
        f"linear=true:"
        f"print_format=summary"
    )

    cmd = [
        ffmpeg,
        "-y",
        "-i", str(wav_path),
        "-af", loudnorm_filter,
        "-ar", "48000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(output_path),
    ]
    logger.debug("apply_loudnorm command: %s", " ".join(cmd))

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        stderr_text = result.stderr.decode(errors="replace")
        logger.error("FFmpeg loudnorm pass-2 failed (rc=%d):\n%s", result.returncode, stderr_text)
        raise RuntimeError(
            f"FFmpeg loudnorm failed with return code {result.returncode}.\n{stderr_text}"
        )

    logger.info(
        "Loudnorm applied: %s → %s (target %.1f LUFS)",
        wav_path.name,
        output_path.name,
        target_lufs,
    )
    return output_path.resolve()


# ---------------------------------------------------------------------------
# Task 3 — voice isolation (Demucs)
# ---------------------------------------------------------------------------


def _run_demucs(wav_path: Path) -> np.ndarray:
    """Run Demucs htdemucs and return the vocals stem as numpy array at 44.1kHz."""
    if not _DEMUCS_AVAILABLE:
        raise RuntimeError("Demucs is not installed.")

    model_name = "htdemucs"
    if model_name not in _demucs_model_cache:
        model = get_model(model_name)
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            device = "xpu"
        model.to(device)
        _demucs_model_cache[model_name] = (model, device)

    model, device = _demucs_model_cache[model_name]

    # Load audio
    wav, sr = torchaudio.load(str(wav_path))
    # Demucs expects stereo — if mono, duplicate
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    # Resample to model's sample rate if needed
    if sr != model.samplerate:
        wav = torchaudio.transforms.Resample(sr, model.samplerate)(wav)

    # Add batch dim and move to device
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / ref.std()
    wav = wav.unsqueeze(0).to(device)

    with torch.no_grad():
        sources = apply_model(model, wav, device=device)

    # sources shape: (batch, num_sources, channels, samples)
    # source order: drums, bass, other, vocals
    vocals_idx = model.sources.index("vocals")
    vocals = sources[0, vocals_idx].mean(0).cpu().numpy()  # mono
    return vocals


def apply_voice_isolation(
    wav_path: Path,
    output_path: Path,
) -> Path:
    """Extract vocals using Demucs htdemucs, resample to 48kHz, save as WAV."""
    if not _DEMUCS_AVAILABLE:
        raise RuntimeError(
            "Demucs is not installed. Install it with: pip install demucs"
        )

    vocals_np = _run_demucs(wav_path)

    # Demucs outputs at 44100Hz — resample to 48000Hz
    vocals_tensor = torch.from_numpy(vocals_np).unsqueeze(0)  # (1, samples)
    resampler = torchaudio.transforms.Resample(44100, 48000)
    vocals_48k = resampler(vocals_tensor)

    # Save as 16-bit WAV using soundfile (avoids torchaudio backend quirks)
    vocals_np_48k = vocals_48k.squeeze(0).numpy()
    sf.write(str(output_path), vocals_np_48k, 48000, subtype="PCM_16")
    logger.info("Voice isolation: %s → %s", wav_path.name, output_path.name)
    return output_path
