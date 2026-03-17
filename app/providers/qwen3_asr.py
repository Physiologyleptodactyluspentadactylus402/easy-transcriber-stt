from __future__ import annotations
import asyncio
import logging
from pathlib import Path
from typing import Callable
import subprocess
import sys

logger = logging.getLogger("transcriber.qwen3_asr")

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
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "qwen-asr", "torch", "torchaudio",
        ])
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

    def _load_model(self, model_id: str) -> object:
        """Load a Qwen3ASRModel, caching by model_id."""
        if model_id not in self._model_cache:
            hf_id = _HF_MODEL_IDS.get(model_id, model_id)
            has_cuda = torch is not None and torch.cuda.is_available()
            device_map = "cuda:0" if has_cuda else "cpu"

            kwargs: dict = {
                "device_map": device_map,
                "max_new_tokens": 512,
            }
            if torch is not None:
                kwargs["dtype"] = torch.bfloat16 if has_cuda else torch.float32

            logger.info("Loading Qwen3-ASR model %s on %s …", hf_id, device_map)
            self._model_cache[model_id] = Qwen3ASRModel.from_pretrained(
                hf_id, **kwargs,
            )
            logger.info("Qwen3-ASR model %s loaded", hf_id)
        return self._model_cache[model_id]

    async def transcribe_batch(
        self, chunks: list[Path], opts: TranscribeOptions
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

        for chunk_path in chunks:
            def _run(path=chunk_path):
                # qwen-asr accepts local paths, URLs, base64, or numpy arrays
                results = model.transcribe(audio=str(path))
                return results

            results = await loop.run_in_executor(None, _run)

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

        return TranscriptResult(
            segments=segments,
            provider_name=self.name,
            model_id=opts.model_id,
        )
