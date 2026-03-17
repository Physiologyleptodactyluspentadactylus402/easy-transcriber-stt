from __future__ import annotations
from pathlib import Path
from typing import Callable

try:
    from openai import OpenAI
except ImportError:  # openai not installed — provider will report unavailable
    OpenAI = None  # type: ignore[assignment,misc]

from app.providers.base import (
    BaseProvider, ModelInfo, HardwareHint, TranscribeOptions,
    TranscriptResult, Segment,
)


class OpenAIProvider(BaseProvider):
    name = "openai"

    models = [
        ModelInfo(
            id="whisper-1",
            name="Whisper 1",
            description="OpenAI Whisper — reliable, widely supported",
            hardware_hint=HardwareHint.CLOUD,
            supports_timestamps=True,
        ),
        ModelInfo(
            id="gpt-4o-transcribe",
            name="GPT-4o Transcribe",
            description="Higher accuracy, higher cost",
            hardware_hint=HardwareHint.CLOUD,
            supports_timestamps=True,
        ),
        ModelInfo(
            id="gpt-4o-mini-transcribe",
            name="GPT-4o Mini Transcribe",
            description="Fast and affordable",
            hardware_hint=HardwareHint.CLOUD,
            supports_timestamps=True,
        ),
    ]

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key

    def is_available(self) -> bool:
        return bool(self._api_key)

    def install_deps(
        self,
        progress_callback: "Callable[[float, str], None] | None" = None,
    ) -> None:
        pass  # No heavy deps required

    async def transcribe_batch(
        self, chunks: list[Path], opts: TranscribeOptions
    ) -> TranscriptResult:
        if OpenAI is None:
            raise RuntimeError("openai package is not installed")
        client = OpenAI(api_key=self._api_key)

        segments: list[Segment] = []
        offset = 0.0

        for chunk_path in chunks:
            params: dict = {
                "model": opts.model_id,
                "response_format": "verbose_json",
                "timestamp_granularities": ["segment"],
            }
            if opts.prompt:
                params["prompt"] = opts.prompt
            if opts.language:
                params["language"] = opts.language

            # Use context manager to avoid file descriptor leaks
            with open(chunk_path, "rb") as audio_file:
                params["file"] = audio_file
                response = client.audio.transcriptions.create(**params)

            raw_segs = getattr(response, "segments", None)
            if isinstance(raw_segs, list) and raw_segs:
                for seg in raw_segs:
                    segments.append(Segment(
                        start=seg.start + offset,
                        end=seg.end + offset,
                        text=seg.text.strip(),
                    ))
                offset += response.duration if hasattr(response, "duration") else 0.0
            else:
                # Fallback: treat entire chunk as one segment
                # Use chunk_size_sec from opts rather than a hardcoded magic number
                text = response.text if hasattr(response, "text") else str(response)
                chunk_duration = float(opts.chunk_size_sec)
                segments.append(Segment(
                    start=offset,
                    end=offset + chunk_duration,
                    text=text.strip(),
                ))
                offset += chunk_duration

        return TranscriptResult(
            segments=segments,
            provider_name=self.name,
            model_id=opts.model_id,
        )
