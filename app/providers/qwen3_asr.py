from __future__ import annotations
import asyncio
from pathlib import Path
from typing import Callable
import subprocess
import sys

try:
    from transformers import pipeline
    import torch
except ImportError:
    pipeline = None  # type: ignore[assignment,misc]
    torch = None  # type: ignore[assignment,misc]

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
            description="Compact model, runs on CPU. No segment timestamps.",
            hardware_hint=HardwareHint.CPU_RECOMMENDED,
            supports_timestamps=False,
            supports_live=True,
        ),
        ModelInfo(
            id="1.7b",
            name="Qwen3-ASR 1.7B",
            description="Better accuracy with segment timestamps. GPU recommended.",
            hardware_hint=HardwareHint.GPU_OPTIONAL,
            supports_timestamps=True,
            supports_live=True,
        ),
    ]

    def __init__(self) -> None:
        self._pipe_cache: dict[str, object] = {}

    def is_available(self) -> bool:
        return pipeline is not None

    def install_deps(
        self,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> None:
        global pipeline, torch
        if progress_callback:
            progress_callback(0.0, "Installing transformers and torch…")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "git+https://github.com/huggingface/transformers.git",
            "torch", "torchaudio",
        ])
        # Re-import after install so is_available() returns True
        # without requiring a server restart
        from transformers import pipeline as _pipeline
        import torch as _torch
        pipeline = _pipeline
        torch = _torch
        if progress_callback:
            progress_callback(1.0, "Dependencies installed.")

    def _load_pipe(self, model_id: str) -> object:
        if model_id not in self._pipe_cache:
            hf_id = _HF_MODEL_IDS.get(model_id, model_id)
            device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
            dtype = (torch.float16 if device == "cuda" else torch.float32) if torch else None
            pipe_kwargs: dict = {
                "model": hf_id,
                "device": device,
            }
            if dtype is not None:
                pipe_kwargs["torch_dtype"] = dtype
            self._pipe_cache[model_id] = pipeline(
                "automatic-speech-recognition",
                trust_remote_code=True,
                **pipe_kwargs,
            )
        return self._pipe_cache[model_id]

    async def transcribe_batch(
        self, chunks: list[Path], opts: TranscribeOptions
    ) -> TranscriptResult:
        if pipeline is None:
            raise RuntimeError(
                "transformers/torch not installed. Open Settings to install."
            )
        loop = asyncio.get_running_loop()
        pipe = await loop.run_in_executor(None, lambda: self._load_pipe(opts.model_id))
        return_timestamps = opts.model_id == "1.7b"

        segments: list[Segment] = []
        offset = 0.0

        for chunk_path in chunks:
            def _run(path=chunk_path):
                return pipe(str(path), return_timestamps=return_timestamps)

            result = await loop.run_in_executor(None, _run)

            if return_timestamps and isinstance(result.get("chunks"), list):
                chunk_segs = result["chunks"]
                for ch in chunk_segs:
                    ts = ch.get("timestamp") or (offset, offset + opts.chunk_size_sec)
                    segments.append(Segment(
                        start=(ts[0] or offset) + offset,
                        end=(ts[1] or offset + opts.chunk_size_sec) + offset,
                        text=ch["text"].strip(),
                    ))
                offset = segments[-1].end if segments else offset + opts.chunk_size_sec
            else:
                text = (result.get("text") or "").strip()
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
