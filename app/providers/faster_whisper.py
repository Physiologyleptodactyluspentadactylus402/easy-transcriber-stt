from __future__ import annotations
import asyncio
from pathlib import Path
from typing import Callable
import subprocess
import sys

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None  # type: ignore[assignment,misc]

from app.providers.base import (
    BaseProvider, ModelInfo, HardwareHint, TranscribeOptions,
    TranscriptResult, Segment,
)

class FasterWhisperProvider(BaseProvider):
    name = "faster_whisper"

    def __init__(self) -> None:
        self._model_cache: dict[str, object] = {}

    models = [
        ModelInfo(
            id="tiny",
            name="Whisper Tiny",
            description="Very fast, lower accuracy. Good for quick drafts.",
            hardware_hint=HardwareHint.CPU,
            supports_timestamps=True,
            supports_live=True,
        ),
        ModelInfo(
            id="base",
            name="Whisper Base",
            description="Fast with decent accuracy.",
            hardware_hint=HardwareHint.CPU,
            supports_timestamps=True,
            supports_live=True,
        ),
        ModelInfo(
            id="small",
            name="Whisper Small",
            description="Good balance of speed and accuracy.",
            hardware_hint=HardwareHint.CPU_RECOMMENDED,
            supports_timestamps=True,
            supports_live=True,
        ),
        ModelInfo(
            id="medium",
            name="Whisper Medium",
            description="High accuracy, slower on CPU.",
            hardware_hint=HardwareHint.GPU_OPTIONAL,
            supports_timestamps=True,
            supports_live=True,
        ),
        ModelInfo(
            id="large-v3",
            name="Whisper Large v3",
            description="Best accuracy. Recommended GPU or fast CPU.",
            hardware_hint=HardwareHint.GPU_RECOMMENDED,
            supports_timestamps=True,
            supports_live=True,
        ),
    ]

    def is_available(self) -> bool:
        return WhisperModel is not None

    def install_deps(
        self,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> None:
        if progress_callback:
            progress_callback(0.0, "Installing faster-whisper…")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "faster-whisper"])
        if progress_callback:
            progress_callback(1.0, "faster-whisper installed.")

    def _load_model(self, model_id: str) -> object:
        if model_id not in self._model_cache:
            self._model_cache[model_id] = WhisperModel(
                model_id, device="cpu", compute_type="int8"
            )
        return self._model_cache[model_id]

    async def transcribe_batch(
        self, chunks: list[Path], opts: TranscribeOptions
    ) -> TranscriptResult:
        if WhisperModel is None:
            raise RuntimeError(
                "faster-whisper is not installed. Open Settings to install it."
            )
        loop = asyncio.get_running_loop()
        model = await loop.run_in_executor(
            None, lambda: self._load_model(opts.model_id)
        )

        segments: list[Segment] = []
        offset = 0.0

        for chunk_path in chunks:
            def _transcribe(path=chunk_path):
                kwargs: dict = {"beam_size": 5}
                if opts.language:
                    kwargs["language"] = opts.language
                if opts.prompt:
                    kwargs["initial_prompt"] = opts.prompt
                raw_segs, info = model.transcribe(str(path), **kwargs)
                return list(raw_segs), info.duration

            raw_segs, duration = await loop.run_in_executor(None, _transcribe)
            for seg in raw_segs:
                segments.append(Segment(
                    start=seg.start + offset,
                    end=seg.end + offset,
                    text=seg.text.strip(),
                ))
            offset += duration

        return TranscriptResult(
            segments=segments,
            provider_name=self.name,
            model_id=opts.model_id,
        )
