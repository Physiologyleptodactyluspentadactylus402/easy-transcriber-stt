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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import soundfile as sf

try:
    import torch
    import torchaudio
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    _DEMUCS_AVAILABLE = True
except ImportError:
    _DEMUCS_AVAILABLE = False

_demucs_model_cache = {}

# ---------------------------------------------------------------------------
# torchaudio compatibility shims
# ---------------------------------------------------------------------------
# The XPU PyTorch build redirects torchaudio.load/save to TorchCodec which
# may not be installed.  DeepFilterNet also needs torchaudio.backend.common
# (removed in torchaudio >= 2.1).  Install soundfile-based fallbacks for all
# three, BEFORE any library that depends on torchaudio is imported.
# ---------------------------------------------------------------------------
import importlib as _importlib

if _importlib.util.find_spec("torchaudio"):
    import torchaudio as _ta

    # --- shim torchaudio.backend.common (for DeepFilterNet) ---
    if not _importlib.util.find_spec("torchaudio.backend"):
        import types, sys, collections
        _backend = types.ModuleType("torchaudio.backend")
        _common = types.ModuleType("torchaudio.backend.common")
        _common.AudioMetaData = collections.namedtuple(
            "AudioMetaData", ["sample_rate", "num_frames", "num_channels",
                              "bits_per_sample", "encoding"],
            defaults=[0, 0, 0, 0, ""],
        )
        _backend.common = _common
        sys.modules["torchaudio.backend"] = _backend
        sys.modules["torchaudio.backend.common"] = _common

    # --- shim torchaudio.info ---
    if not hasattr(_ta, "info"):
        def _ta_info(filepath, **kw):
            _fi = sf.info(str(filepath))
            _bits = 0
            if _fi.subtype:
                _m = __import__("re").search(r"\d+", _fi.subtype)
                if _m:
                    _bits = int(_m.group())
            from torchaudio.backend.common import AudioMetaData as _AMD
            return _AMD(
                sample_rate=_fi.samplerate,
                num_frames=_fi.frames,
                num_channels=_fi.channels,
                bits_per_sample=_bits,
                encoding=_fi.subtype or "",
            )
        _ta.info = _ta_info

    # --- guard torchaudio.load (fallback to soundfile) ---
    _orig_ta_load = _ta.load

    def _safe_ta_load(filepath, *args, **kw):
        try:
            return _orig_ta_load(filepath, *args, **kw)
        except (RuntimeError, OSError, ImportError):
            data, sr = sf.read(str(filepath), dtype="float32", always_2d=True)
            tensor = torch.from_numpy(data.T)
            return tensor, sr

    _ta.load = _safe_ta_load

    # --- guard torchaudio.save (fallback to soundfile) ---
    _orig_ta_save = _ta.save

    def _safe_ta_save(filepath, src, sample_rate, *args, **kw):
        try:
            return _orig_ta_save(filepath, src, sample_rate, *args, **kw)
        except (RuntimeError, OSError, ImportError):
            audio_np = src.cpu().numpy()
            if audio_np.ndim == 2:
                audio_np = audio_np.T
            sf.write(str(filepath), audio_np, sample_rate)

    _ta.save = _safe_ta_save

try:
    from df.enhance import enhance, init_df, load_audio, save_audio
    _DEEPFILTER_AVAILABLE = True
except ImportError:
    _DEEPFILTER_AVAILABLE = False

_deepfilter_cache = {}

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


def _select_device() -> str:
    """Pick the best available accelerator: CUDA > XPU > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


DEMUCS_CHUNK_SEC = 300     # 5-minute chunks
DEMUCS_OVERLAP_SEC = 5     # 5-second crossfade overlap


def _get_demucs_model():
    """Get or create cached Demucs model on the best available device."""
    if not _DEMUCS_AVAILABLE:
        raise RuntimeError("Demucs is not installed.")

    model_name = "htdemucs"
    if model_name not in _demucs_model_cache:
        model = get_model(model_name)
        device = _select_device()
        model.to(device)
        if device != "cpu":
            try:
                model = model.to(memory_format=torch.channels_last)
            except Exception:
                pass
        _demucs_model_cache[model_name] = (model, device)
    return _demucs_model_cache[model_name]


def _process_demucs_chunk(model, chunk_wav, device) -> np.ndarray:
    """Process a single stereo chunk through Demucs, with OOM fallback."""
    ref = chunk_wav.mean(0)
    chunk_wav = (chunk_wav - ref.mean()) / ref.std()
    chunk_wav = chunk_wav.unsqueeze(0).to(device)

    try:
        with torch.no_grad():
            if device != "cpu":
                with torch.autocast(device_type=device, dtype=torch.float16):
                    sources = apply_model(model, chunk_wav, device=device)
            else:
                sources = apply_model(model, chunk_wav, device=device)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            logger.warning("GPU OOM on chunk — falling back to CPU")
            if hasattr(torch, "xpu"):
                torch.xpu.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
            model_cpu = model.cpu()
            chunk_cpu = chunk_wav.cpu()
            with torch.no_grad():
                sources = apply_model(model_cpu, chunk_cpu, device="cpu")
            model.to(device)
        else:
            raise

    vocals_idx = model.sources.index("vocals")
    return sources[0, vocals_idx].mean(0).cpu().float().numpy()


def _run_demucs(wav_path: Path) -> np.ndarray:
    """Run Demucs htdemucs with chunked processing to avoid GPU OOM.

    Splits audio into 5-minute chunks with 5-second overlap, processes each
    on GPU, then reassembles with linear crossfade in the overlap regions.
    """
    model, device = _get_demucs_model()

    wav, sr = torchaudio.load(str(wav_path))
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    if sr != model.samplerate:
        wav = torchaudio.transforms.Resample(sr, model.samplerate)(wav)
    sr = model.samplerate

    total_samples = wav.shape[1]
    chunk_samples = DEMUCS_CHUNK_SEC * sr
    overlap_samples = DEMUCS_OVERLAP_SEC * sr
    step_samples = chunk_samples - overlap_samples

    # Short file — process as single chunk
    if total_samples <= chunk_samples:
        return _process_demucs_chunk(model, wav, device)

    logger.info(
        "Chunked Demucs: %d samples, chunk=%ds, overlap=%ds, ~%d chunks",
        total_samples, DEMUCS_CHUNK_SEC, DEMUCS_OVERLAP_SEC,
        (total_samples + step_samples - 1) // step_samples,
    )

    vocals = np.zeros(total_samples, dtype=np.float32)
    chunk_idx = 0

    for start in range(0, total_samples, step_samples):
        end = min(start + chunk_samples, total_samples)
        chunk = wav[:, start:end]
        chunk_vocals = _process_demucs_chunk(model, chunk, device)

        if start == 0:
            # First chunk — no crossfade needed
            vocals[start:end] = chunk_vocals[:end - start]
        else:
            # Crossfade the overlap region
            ol_len = min(overlap_samples, end - start, len(chunk_vocals))
            fade_in = np.linspace(0.0, 1.0, ol_len, dtype=np.float32)
            fade_out = 1.0 - fade_in
            vocals[start:start + ol_len] = (
                vocals[start:start + ol_len] * fade_out
                + chunk_vocals[:ol_len] * fade_in
            )
            # Copy the rest (non-overlap portion)
            rest_start = ol_len
            rest_len = (end - start) - ol_len
            if rest_len > 0:
                vocals[start + ol_len:end] = chunk_vocals[rest_start:rest_start + rest_len]

        chunk_idx += 1
        logger.debug("Demucs chunk %d done: samples %d-%d", chunk_idx, start, end)

        if end >= total_samples:
            break

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
    return output_path.resolve()


# ---------------------------------------------------------------------------
# Task 4 — noise reduction (DeepFilterNet)
# ---------------------------------------------------------------------------


def _run_deepfilter(wav_path: Path) -> np.ndarray:
    """Run DeepFilterNet and return enhanced audio as numpy array at 48kHz."""
    if not _DEEPFILTER_AVAILABLE:
        raise RuntimeError("DeepFilterNet is not installed.")

    if "model" not in _deepfilter_cache:
        # Force CPU: DeepFilterNet doesn't support XPU.  Its config()
        # reads os.environ["DEVICE"] (upper-cased option name).
        import os as _os
        _prev_device = _os.environ.get("DEVICE")
        _os.environ["DEVICE"] = "cpu"
        model, df_state, _ = init_df()
        model = model.cpu()
        if _prev_device is None:
            _os.environ.pop("DEVICE", None)
        else:
            _os.environ["DEVICE"] = _prev_device
        _deepfilter_cache["model"] = model
        _deepfilter_cache["df_state"] = df_state

    model = _deepfilter_cache["model"]
    df_state = _deepfilter_cache["df_state"]

    audio, _ = load_audio(str(wav_path), sr=df_state.sr())
    enhanced = enhance(model, df_state, audio)
    return enhanced.numpy().flatten()


def _apply_denoise_ffmpeg(wav_path: Path, output_path: Path) -> Path:
    """Denoise using ffmpeg's afftdn filter, optimized for speech."""
    if not wav_path.exists():
        raise FileNotFoundError(f"Input WAV file not found: {wav_path}")

    ffmpeg = _require_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg, "-y", "-i", str(wav_path),
        "-af", "afftdn=nf=-25:tn=1",
        "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le",
        str(output_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        stderr_text = result.stderr.decode(errors="replace")
        logger.error("ffmpeg afftdn failed (rc=%d):\n%s", result.returncode, stderr_text)
        raise RuntimeError(f"ffmpeg denoise failed: {stderr_text}")
    logger.info("Denoise (ffmpeg): %s → %s", wav_path.name, output_path.name)
    return output_path.resolve()


def _apply_denoise_deepfilter(wav_path: Path, output_path: Path) -> Path:
    """Denoise using DeepFilterNet (state-of-the-art neural noise reduction)."""
    enhanced = _run_deepfilter(wav_path)
    sf.write(str(output_path), enhanced, 48000, subtype="PCM_16")
    logger.info("Denoise (deepfilter): %s → %s", wav_path.name, output_path.name)
    return output_path.resolve()


def apply_denoise(wav_path: Path, output_path: Path, engine: str = "ffmpeg") -> Path:
    """Denoise audio using the specified engine."""
    wav_path = Path(wav_path)

    if not wav_path.exists():
        raise FileNotFoundError(f"Input WAV file not found: {wav_path}")

    if engine == "deepfilter":
        if not _DEEPFILTER_AVAILABLE:
            raise RuntimeError(
                "DeepFilterNet is not installed. "
                "Install it with: pip install deepfilternet"
            )
        return _apply_denoise_deepfilter(wav_path, output_path)
    else:
        return _apply_denoise_ffmpeg(wav_path, output_path)


# ---------------------------------------------------------------------------
# Audio polish (high-pass, de-esser, EQ, compressor, limiter)
# ---------------------------------------------------------------------------


def _apply_polish(wav_path: Path, output_path: Path) -> Path:
    """Apply audio polish: high-pass, de-esser, EQ, compressor, limiter.

    All filters run in a single ffmpeg pass — no intermediate files.
    """
    wav_path = Path(wav_path)
    if not wav_path.exists():
        raise FileNotFoundError(f"Input WAV file not found: {wav_path}")

    ffmpeg = _require_ffmpeg()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    af = ",".join([
        # High-pass: remove rumble below 80Hz
        "highpass=f=80:poles=2",
        # De-esser: boost sibilant band → compress → cut back
        "equalizer=f=6000:width_type=s:width=2.0:g=3",
        "acompressor=threshold=0.1:ratio=4:attack=5:release=50",
        "equalizer=f=6000:width_type=s:width=2.0:g=-3",
        # EQ: body (shelf) + presence (peak) + air (shelf)
        "lowshelf=f=250:width_type=s:width=0.8:g=2",
        "equalizer=f=3000:width_type=s:width=1.0:g=1.5",
        "highshelf=f=10000:width_type=s:width=0.8:g=1",
        # Gentle compressor
        "acompressor=threshold=0.1:ratio=2:attack=20:release=200",
        # Limiter -1dBTP (0.891 linear)
        "alimiter=limit=0.891:level=disabled",
    ])
    cmd = [
        ffmpeg, "-y", "-i", str(wav_path),
        "-af", af,
        "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le",
        str(output_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        stderr_text = result.stderr.decode(errors="replace")
        logger.error("ffmpeg polish failed (rc=%d):\n%s", result.returncode, stderr_text)
        raise RuntimeError(f"ffmpeg polish failed: {stderr_text}")
    logger.info("Polish: %s → %s", wav_path.name, output_path.name)
    return output_path.resolve()


# ---------------------------------------------------------------------------
# Task 5 — pipeline orchestrator
# ---------------------------------------------------------------------------


def _resample_to_16k(wav_path: Path, output_path: Path) -> Path:
    """Downsample WAV to 16kHz mono 16-bit for transcription."""
    ffmpeg = _require_ffmpeg()
    cmd = [
        ffmpeg, "-y", "-i", str(wav_path),
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        str(output_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        stderr_text = result.stderr.decode(errors="replace")
        raise RuntimeError(f"FFmpeg resample failed: {stderr_text}")
    return output_path.resolve()


@dataclass
class PreprocessConfig:
    """Configuration for the preprocessing pipeline."""
    loudnorm: bool = True
    loudnorm_target: float = -16.0
    voice_isolation: bool = True
    denoise: bool = True
    denoise_engine: str = "ffmpeg"
    polish: bool = False


@dataclass
class PipelineResult:
    """Result of a preprocessing pipeline run."""
    original_path: Path
    processed_path: Path
    processed_48k_path: Path
    stats: dict | None = None
    cancelled: bool = False
    steps_completed: list[str] = field(default_factory=list)


def _early_return(
    original_wav: Path, current: Path, original_lufs: float,
    steps_completed: list[str], work_dir: Path,
) -> PipelineResult:
    """Build a cancelled PipelineResult, resampling current to 16kHz."""
    processed_16k = work_dir / "processed.wav"
    _resample_to_16k(current, processed_16k)
    return PipelineResult(
        original_path=original_wav, processed_path=processed_16k,
        processed_48k_path=current,
        stats={"original_lufs": original_lufs},
        cancelled=True, steps_completed=steps_completed,
    )


def run_pipeline(
    input_path: Path,
    work_dir: Path,
    config: PreprocessConfig,
    progress_callback: Callable[[float, str, str], None] | None = None,
    cancel_flag: dict | None = None,
) -> PipelineResult:
    """Run the full preprocessing pipeline.

    Args:
        input_path: Path to the input audio file.
        work_dir: Directory for intermediate and output files.
        config: Which steps to run and parameters.
        progress_callback: Called with (fraction, step_name, message).
        cancel_flag: Dict with key 'cancelled' checked between steps.

    Returns:
        PipelineResult with paths and stats.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    def _cancelled() -> bool:
        return bool(cancel_flag and cancel_flag.get("cancelled"))

    def _progress(frac: float, step: str, msg: str):
        if progress_callback:
            progress_callback(frac, step, msg)

    steps_completed: list[str] = []

    # Step 0: Decode to 48kHz WAV
    _progress(0.0, "decode", "Decoding audio...")
    original_wav = work_dir / "original.wav"
    decode_to_wav(input_path, original_wav, sample_rate=48000)
    steps_completed.append("decode")

    # LUFS analysis (always, even if loudnorm is off)
    _progress(0.1, "analyze", "Measuring loudness...")
    lufs_stats = analyze_lufs(original_wav)
    original_lufs = lufs_stats["input_i"]

    if _cancelled():
        return _early_return(original_wav, original_wav, original_lufs, steps_completed, work_dir)

    # Track the current working file (48kHz)
    current = original_wav

    # Step 1: Denoise (before voice isolation — preserves original harmonics)
    if config.denoise:
        if _cancelled():
            return _early_return(original_wav, current, original_lufs, steps_completed, work_dir)
        engine_label = "deepfilter" if config.denoise_engine == "deepfilter" else "ffmpeg_denoise"
        _progress(0.15, engine_label, "Removing noise...")
        denoised = work_dir / "step1_denoised.wav"
        apply_denoise(current, denoised, engine=config.denoise_engine)
        current = denoised
        steps_completed.append(engine_label)

    # Step 2: Voice isolation (chunked Demucs)
    if config.voice_isolation:
        if _cancelled():
            return _early_return(original_wav, current, original_lufs, steps_completed, work_dir)
        _progress(0.3, "demucs", "Isolating voice (this may take a while)...")
        vocals = work_dir / "step2_vocals.wav"
        apply_voice_isolation(current, vocals)
        current = vocals
        steps_completed.append("demucs")

    # Step 3: Polish (high-pass, de-esser, EQ, compressor, limiter)
    if config.polish:
        if _cancelled():
            return _early_return(original_wav, current, original_lufs, steps_completed, work_dir)
        _progress(0.7, "polish", "Applying audio polish...")
        polished = work_dir / "step3_polished.wav"
        _apply_polish(current, polished)
        current = polished
        steps_completed.append("polish")

    # Step 4: Loudnorm (at the end — normalizes the final result)
    if config.loudnorm:
        if _cancelled():
            return _early_return(original_wav, current, original_lufs, steps_completed, work_dir)
        _progress(0.8, "loudnorm", f"Normalizing to {config.loudnorm_target} LUFS...")
        normalized = work_dir / "step4_loudnorm.wav"
        # Re-analyze current file since it's been modified by previous steps.
        # Pass measured= to skip the redundant pass-1 inside apply_loudnorm.
        current_lufs = analyze_lufs(current) if current != original_wav else lufs_stats
        apply_loudnorm(current, normalized, target_lufs=config.loudnorm_target, measured=current_lufs)
        current = normalized
        steps_completed.append("loudnorm")

    # Final: save 48kHz copy for player + 16kHz for transcription
    _progress(0.9, "resample", "Preparing final output...")
    processed_48k = work_dir / "processed_48k.wav"
    processed_16k = work_dir / "processed.wav"

    # Copy current result as the 48k version (for A/B player)
    if current.resolve() != processed_48k.resolve():
        shutil.copy2(current, processed_48k)

    _resample_to_16k(current, processed_16k)
    steps_completed.append("resample")

    # Measure processed LUFS
    processed_lufs_stats = analyze_lufs(processed_48k)
    original_size = Path(input_path).stat().st_size
    processed_size = processed_16k.stat().st_size

    from pydub import AudioSegment
    duration_sec = len(AudioSegment.from_wav(str(processed_16k))) / 1000.0

    stats = {
        "original_lufs": original_lufs,
        "processed_lufs": processed_lufs_stats["input_i"],
        "duration_sec": duration_sec,
        "original_size": original_size,
        "processed_size": processed_size,
    }

    _progress(1.0, "done", "Processing complete")
    logger.info("Pipeline complete: steps=%s, stats=%s", steps_completed, stats)
    return PipelineResult(
        original_path=original_wav,
        processed_path=processed_16k,
        processed_48k_path=processed_48k,
        stats=stats,
        steps_completed=steps_completed,
    )
