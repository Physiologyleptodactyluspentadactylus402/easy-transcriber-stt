from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import AsyncGenerator, Literal
import time
import uuid


class HardwareHint(str, Enum):
    CPU = "cpu"
    CPU_RECOMMENDED = "cpu_recommended"
    GPU_OPTIONAL = "gpu_optional"
    GPU_RECOMMENDED = "gpu_recommended"
    CLOUD = "cloud"


@dataclass
class ModelInfo:
    id: str
    name: str
    description: str
    hardware_hint: HardwareHint
    supports_live: bool = False
    supports_speaker_labels: bool = False
    supports_timestamps: bool = True


@dataclass
class TranscribeOptions:
    language: str | None = None
    prompt: str = ""
    speaker_labels: bool = False
    output_formats: list[str] = field(default_factory=lambda: ["txt"])
    chunk_size_sec: int = 600
    model_id: str = "whisper-1"


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str | None = None


@dataclass
class TranscriptResult:
    segments: list[Segment]
    language_detected: str | None = None
    provider_name: str = ""
    model_id: str = ""


@dataclass
class Job:
    input_files: list[Path]
    opts: TranscribeOptions
    provider_name: str
    model_id: str
    status: Literal["pending", "running", "done", "error", "cancelled"]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    progress: float = 0.0
    error_message: str | None = None
    created_at: float = field(default_factory=time.time)
    output_files: list[Path] = field(default_factory=list)
    merge_output: bool = False


class BaseProvider(ABC):
    name: str
    models: list[ModelInfo]

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this provider can run right now."""

    @abstractmethod
    def install_deps(self) -> None:
        """Install heavy dependencies. May take minutes. Called from GUI."""

    @abstractmethod
    async def transcribe_batch(
        self, chunks: list[Path], opts: TranscribeOptions
    ) -> TranscriptResult:
        """Transcribe pre-split audio chunks in order."""

    async def transcribe_live(
        self,
        audio_stream: AsyncGenerator[bytes, None],
        opts: TranscribeOptions,
    ) -> AsyncGenerator[Segment, None]:
        raise NotImplementedError(f"{self.name} does not support live transcription")
        # make this a proper async generator
        return
        yield  # noqa: unreachable — makes Python treat this as async generator
