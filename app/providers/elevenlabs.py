from __future__ import annotations
from pathlib import Path
from typing import Callable
from app.providers.base import (
    BaseProvider, ModelInfo, HardwareHint, TranscribeOptions,
    TranscriptResult, Segment,
)

try:
    from elevenlabs import ElevenLabs
except ImportError:  # package not installed — will fail at runtime if used
    ElevenLabs = None  # type: ignore[assignment,misc]


class ElevenLabsProvider(BaseProvider):
    name = "elevenlabs"

    models = [
        ModelInfo(
            id="scribe_v2",
            name="Scribe v2",
            description="Best accuracy, speaker labels, 90+ languages",
            hardware_hint=HardwareHint.CLOUD,
            supports_speaker_labels=True,
            supports_timestamps=True,
        ),
        ModelInfo(
            id="scribe_v2_realtime",
            name="Scribe v2 Realtime",
            description="<150ms latency live transcription",
            hardware_hint=HardwareHint.CLOUD,
            supports_live=True,
            supports_timestamps=True,
        ),
        ModelInfo(
            id="scribe_v1",
            name="Scribe v1 (legacy)",
            description="Previous generation — use v2 unless on legacy plan",
            hardware_hint=HardwareHint.CLOUD,
            supports_speaker_labels=True,
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
        pass

    async def transcribe_batch(
        self, chunks: list[Path], opts: TranscribeOptions,
        progress_callback=None,
    ) -> TranscriptResult:
        if ElevenLabs is None:
            raise ImportError("elevenlabs package is not installed. Run: pip install elevenlabs")

        if not chunks:
            return TranscriptResult(segments=[], provider_name=self.name, model_id=opts.model_id)

        client = ElevenLabs(api_key=self._api_key)
        all_segments: list[Segment] = []
        offset = 0.0
        last_response = None

        for chunk_path in chunks:
            with open(chunk_path, "rb") as f:
                response = client.speech_to_text.convert(
                    file=f,
                    model_id=opts.model_id,
                    language_code=opts.language,
                    diarize=opts.speaker_labels,
                    additional_formats=[],
                    keyterm_prompting_enabled=bool(opts.prompt),
                    keyterms=opts.prompt.split(",") if opts.prompt else [],
                )
            last_response = response

            # Parse utterances (with speaker) or words, or fall back to plain text
            if opts.speaker_labels and response.utterances:
                for utt in response.utterances:
                    all_segments.append(Segment(
                        start=(utt.start or 0.0) + offset,
                        end=(utt.end or 0.0) + offset,
                        text=utt.text.strip(),
                        speaker=f"Speaker {utt.speaker_id}" if utt.speaker_id is not None else None,
                    ))
            elif response.words:
                all_segments.extend(_words_to_segments(response.words, offset))
            else:
                chunk_duration = float(opts.chunk_size_sec)
                all_segments.append(Segment(
                    start=offset,
                    end=offset + chunk_duration,
                    text=(response.text or "").strip(),
                ))

            # Advance offset by last segment end
            if all_segments:
                offset = all_segments[-1].end

        return TranscriptResult(
            segments=all_segments,
            language_detected=getattr(last_response, "language_code", None),
            provider_name=self.name,
            model_id=opts.model_id,
        )


def _words_to_segments(words, offset: float, max_gap_sec: float = 1.5) -> list[Segment]:
    """Group words into segments separated by pauses > max_gap_sec."""
    if not words:
        return []
    segments: list[Segment] = []
    current_words = [words[0]]

    for word in words[1:]:
        gap = (word.start or 0) - (current_words[-1].end or 0)
        if gap > max_gap_sec:
            segments.append(_make_segment(current_words, offset))
            current_words = [word]
        else:
            current_words.append(word)

    segments.append(_make_segment(current_words, offset))
    return segments


def _make_segment(words, offset: float) -> Segment:
    return Segment(
        start=(words[0].start or 0.0) + offset,
        end=(words[-1].end or 0.0) + offset,
        text=" ".join(w.text for w in words).strip(),
    )
