from __future__ import annotations
import asyncio
import base64
from pathlib import Path
from typing import Callable

import httpx

from app.providers.base import (
    BaseProvider, ModelInfo, HardwareHint, TranscribeOptions,
    TranscriptResult, Segment,
)

_BASE_URL = "http://localhost:11434"


class OllamaProvider(BaseProvider):
    name = "ollama"

    def __init__(self) -> None:
        self._models: list[ModelInfo] = []
        self._available = False
        self._probe()

    def _probe(self) -> None:
        try:
            resp = httpx.get(f"{_BASE_URL}/api/tags", timeout=2.0)
            resp.raise_for_status()
            self._models = [
                ModelInfo(
                    id=m["name"],
                    name=m["name"],
                    description="Ollama model — audio support depends on the model",
                    hardware_hint=HardwareHint.CPU,
                    supports_timestamps=False,
                )
                for m in resp.json().get("models", [])
            ]
            self._available = True
        except Exception:
            self._available = False

    @property
    def models(self) -> list[ModelInfo]:
        return self._models

    def is_available(self) -> bool:
        return self._available

    def install_deps(
        self,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> None:
        pass  # httpx already in requirements.txt; Ollama daemon installed by user

    async def transcribe_batch(
        self, chunks: list[Path], opts: TranscribeOptions,
        progress_callback=None,
    ) -> TranscriptResult:
        if not self._available:
            raise RuntimeError(
                "Ollama daemon is not running. Start Ollama and reload the app."
            )
        loop = asyncio.get_running_loop()
        segments: list[Segment] = []
        offset = 0.0

        for chunk_path in chunks:
            def _call(path=chunk_path, off=offset):
                audio_b64 = base64.b64encode(path.read_bytes()).decode()
                payload = {
                    "model": opts.model_id,
                    "prompt": "Transcribe the following audio. Return only the transcript text.",
                    "images": [audio_b64],
                    "stream": False,
                }
                resp = httpx.post(
                    f"{_BASE_URL}/api/generate",
                    json=payload,
                    timeout=120.0,
                )
                resp.raise_for_status()
                return resp.json().get("response", "").strip()

            text = await loop.run_in_executor(None, _call)
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
