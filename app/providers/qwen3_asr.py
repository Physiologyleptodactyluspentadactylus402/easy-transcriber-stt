from __future__ import annotations
import asyncio
import logging
from pathlib import Path
from typing import Callable
import subprocess
import sys

logger = logging.getLogger("transcriber.qwen3_asr")


def _fmt_eta(seconds: float) -> str:
    """Format seconds into a human-readable ETA string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


def _detect_intel_gpu() -> bool:
    """Check if an Intel Arc/discrete GPU is present (Windows + Linux)."""
    import platform
    try:
        if platform.system() == "Windows":
            import subprocess as _sp
            out = _sp.check_output(
                ["powershell", "-Command",
                 "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
                text=True, timeout=10,
            )
            return "intel" in out.lower() and "arc" in out.lower()
        else:
            # Linux: check for Intel render nodes
            from pathlib import Path as _P
            for dev in _P("/dev/dri").glob("renderD*"):
                return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Try importing the official qwen-asr package (pip install qwen-asr)
# This is the ONLY supported way to run Qwen3-ASR locally.
# It wraps transformers + torch internally.
# ---------------------------------------------------------------------------
try:
    from qwen_asr import Qwen3ASRModel  # type: ignore[import-untyped]
    import torch
    _AVAILABLE = True
except ImportError:
    Qwen3ASRModel = None  # type: ignore[assignment,misc]
    torch = None  # type: ignore[assignment,misc]
    _AVAILABLE = False

from app.providers.base import (
    BaseProvider, ModelInfo, HardwareHint, TranscribeOptions,
    TranscriptResult, Segment,
)

_HF_MODEL_IDS = {
    "0.6b": "Qwen/Qwen3-ASR-0.6B",
    "1.7b": "Qwen/Qwen3-ASR-1.7B",
}


class Qwen3ASRProvider(BaseProvider):
    name = "qwen3_asr"

    models = [
        ModelInfo(
            id="0.6b",
            name="Qwen3-ASR 0.6B",
            description="Compact model, runs on CPU. ~600M parameters.",
            hardware_hint=HardwareHint.CPU_RECOMMENDED,
            supports_timestamps=False,
            supports_live=True,
        ),
        ModelInfo(
            id="1.7b",
            name="Qwen3-ASR 1.7B",
            description="Best accuracy, GPU recommended. ~1.7B parameters.",
            hardware_hint=HardwareHint.GPU_OPTIONAL,
            supports_timestamps=False,
            supports_live=True,
        ),
    ]

    def __init__(self) -> None:
        self._model_cache: dict[str, object] = {}

    def is_available(self) -> bool:
        return _AVAILABLE

    def install_deps(
        self,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> None:
        global Qwen3ASRModel, torch, _AVAILABLE
        if progress_callback:
            progress_callback(0.0, "Installing qwen-asr + torch…")

        # Detect Intel GPU to install XPU-enabled PyTorch instead of CPU-only
        xpu_available = _detect_intel_gpu()
        if xpu_available:
            if progress_callback:
                progress_callback(0.1, "Intel GPU detected — installing PyTorch with XPU…")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--pre",
                "torch", "torchaudio",
                "--index-url", "https://download.pytorch.org/whl/nightly/xpu",
            ])
            if progress_callback:
                progress_callback(0.5, "Installing qwen-asr…")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "qwen-asr",
            ])
        else:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "qwen-asr", "torch", "torchaudio",
            ])
        # pip may have downgraded/upgraded transformers — purge stale
        # cached modules so Python loads the new version from disk.
        import importlib
        for mod_name in list(sys.modules):
            if mod_name == "transformers" or mod_name.startswith("transformers."):
                del sys.modules[mod_name]
        # Also purge any prior failed qwen_asr import
        for mod_name in list(sys.modules):
            if mod_name == "qwen_asr" or mod_name.startswith("qwen_asr."):
                del sys.modules[mod_name]

        # Re-import after install so is_available() flips to True
        # without requiring a server restart
        from qwen_asr import Qwen3ASRModel as _Model  # type: ignore[import-untyped]
        import torch as _torch
        Qwen3ASRModel = _Model
        torch = _torch
        _AVAILABLE = True
        logger.info("qwen-asr installed successfully")
        if progress_callback:
            progress_callback(1.0, "Dependencies installed.")

    @staticmethod
    def _best_device() -> tuple[str, "torch.dtype"]:
        """Pick the best available accelerator: CUDA > XPU (Intel Arc) > CPU."""
        if torch is None:
            return "cpu", None                          # type: ignore[return-value]
        if torch.cuda.is_available():
            return "cuda:0", torch.bfloat16
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            # Intel Arc GPUs support float16; bfloat16 may work on newer
            # drivers but float16 is the safer default.
            return "xpu:0", torch.float16
        return "cpu", torch.float32

    def _load_model(self, model_id: str) -> object:
        """Load a Qwen3ASRModel, caching by model_id."""
        if model_id not in self._model_cache:
            hf_id = _HF_MODEL_IDS.get(model_id, model_id)
            device_map, dtype = self._best_device()

            kwargs: dict = {
                "device_map": device_map,
                "max_new_tokens": 512,
            }
            if dtype is not None:
                kwargs["dtype"] = dtype

            logger.info("Loading Qwen3-ASR model %s on %s (dtype=%s) …", hf_id, device_map, dtype)
            self._model_cache[model_id] = Qwen3ASRModel.from_pretrained(
                hf_id, **kwargs,
            )
            logger.info("Qwen3-ASR model %s loaded", hf_id)
        return self._model_cache[model_id]

    async def transcribe_batch(
        self,
        chunks: list[Path],
        opts: TranscribeOptions,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> TranscriptResult:
        if not _AVAILABLE:
            raise RuntimeError(
                "qwen-asr package not installed. Open Settings to install."
            )
        loop = asyncio.get_running_loop()
        model = await loop.run_in_executor(
            None, lambda: self._load_model(opts.model_id)
        )

        segments: list[Segment] = []
        offset = 0.0

        import time as _time
        t_start = _time.monotonic()
        chunk_times: list[float] = []

        for i, chunk_path in enumerate(chunks, 1):
            logger.info(
                "Transcribing chunk %d/%d: %s", i, len(chunks), chunk_path.name
            )
            if progress_callback:
                frac = (i - 1) / len(chunks)
                if chunk_times:
                    avg = sum(chunk_times) / len(chunk_times)
                    remaining = avg * (len(chunks) - i + 1)
                    eta = _fmt_eta(remaining)
                    msg = f"Chunk {i}/{len(chunks)} — ETA {eta}"
                else:
                    msg = f"Chunk {i}/{len(chunks)}…"
                progress_callback(frac, msg)

            def _run(path=chunk_path):
                t0 = _time.monotonic()
                results = model.transcribe(audio=str(path))
                elapsed = _time.monotonic() - t0
                logger.info("Chunk %s done in %.1fs", path.name, elapsed)
                return results, elapsed

            results, elapsed = await loop.run_in_executor(None, _run)
            chunk_times.append(elapsed)

            # results is a list of TranscriptionResult objects
            # Each has .text and .language attributes
            if results:
                text = results[0].text.strip()
            else:
                text = ""

            segments.append(Segment(
                start=offset,
                end=offset + float(opts.chunk_size_sec),
                text=text,
            ))
            offset += float(opts.chunk_size_sec)

        total_elapsed = _time.monotonic() - t_start
        logger.info("All %d chunks done in %.1fs", len(chunks), total_elapsed)
        if progress_callback:
            progress_callback(1.0, "Transcription complete")

        return TranscriptResult(
            segments=segments,
            provider_name=self.name,
            model_id=opts.model_id,
        )
