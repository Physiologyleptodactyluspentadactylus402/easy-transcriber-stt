from __future__ import annotations
import asyncio
import time
from pathlib import Path

from app.providers.base import TranscribeOptions, Segment, TranscriptResult
from app.core.output import format_transcript

# Approximate compressed bytes per second for webm/opus at 64kbps
_BYTES_PER_SEC = 8_000
# Run transcription when buffer holds this many seconds of audio
_BUFFER_SECS = 10


class LiveSession:
    def __init__(
        self,
        session_id: str,
        provider: object,
        opts: TranscribeOptions,
        output_dir: Path,
    ) -> None:
        self.session_id = session_id
        self.provider = provider
        self.opts = opts
        self.output_dir = Path(output_dir)
        self.segments: list[Segment] = []
        self._buffer = bytearray()
        self._last_save = time.time()
        self._task: asyncio.Task | None = None
        self._stopped = False

    def add_chunk(self, data: bytes) -> None:
        """Append a raw webm/opus chunk to the accumulation buffer."""
        self._buffer.extend(data)

    async def start(self, ws_manager) -> None:
        """Start the background flush-and-transcribe loop."""
        self._task = asyncio.create_task(self._loop(ws_manager))

    async def stop(self, ws_manager) -> list[Path]:
        """Flush remaining audio, write final outputs, return file paths."""
        self._stopped = True
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        # Final transcription of remaining buffer
        if self._buffer:
            await self._transcribe_buffer(bytes(self._buffer), ws_manager)
            self._buffer.clear()
        return self._write_final_outputs()

    async def _loop(self, ws_manager) -> None:
        """Periodically flush the buffer when it holds >= _BUFFER_SECS of audio."""
        threshold = _BUFFER_SECS * _BYTES_PER_SEC
        while not self._stopped:
            await asyncio.sleep(1.0)
            buf_snapshot = bytes(self._buffer)
            if len(buf_snapshot) >= threshold:
                self._buffer.clear()
                await self._transcribe_buffer(buf_snapshot, ws_manager)
            if time.time() - self._last_save >= 30:
                self._save_incremental_txt()
                self._save_incremental_webm(buf_snapshot)
                self._last_save = time.time()

    async def _transcribe_buffer(self, buf_bytes: bytes, ws_manager) -> None:
        """Decode webm → WAV (temp file), transcribe, emit segment messages."""
        import tempfile, os
        from pydub import AudioSegment as PyAudio

        loop = asyncio.get_running_loop()

        def _decode() -> Path:
            tmp_webm = tempfile.NamedTemporaryFile(suffix=".webm", delete=False)
            tmp_webm.write(buf_bytes)
            tmp_webm.close()
            try:
                audio = PyAudio.from_file(tmp_webm.name, format="webm")
            except Exception as exc:
                os.unlink(tmp_webm.name)
                raise RuntimeError(f"ffmpeg could not decode audio: {exc}") from exc
            wav_path = tmp_webm.name + ".wav"
            audio.export(wav_path, format="wav")
            os.unlink(tmp_webm.name)
            return Path(wav_path)

        try:
            wav_path = await loop.run_in_executor(None, _decode)
        except RuntimeError:
            return  # skip buffer if ffmpeg unavailable

        try:
            offset = self.segments[-1].end if self.segments else 0.0
            result: TranscriptResult = await self.provider.transcribe_batch(
                [wav_path], self.opts
            )
            for seg in result.segments:
                adjusted = Segment(
                    start=seg.start + offset,
                    end=seg.end + offset,
                    text=seg.text,
                    speaker=seg.speaker,
                )
                self.segments.append(adjusted)
                await ws_manager.broadcast_global({
                    "type": "segment",
                    "text": adjusted.text,
                    "start": adjusted.start,
                    "end": adjusted.end,
                    "speaker": adjusted.speaker,
                })
        finally:
            import os as _os
            if wav_path.exists():
                _os.unlink(str(wav_path))

    def _save_incremental_txt(self) -> None:
        """Overwrite <session_id>_live.txt with full transcript so far."""
        if not self.segments:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        txt_path = self.output_dir / f"{self.session_id}_live.txt"
        result = TranscriptResult(
            segments=self.segments,
            provider_name=getattr(self.provider, "name", ""),
            model_id=self.opts.model_id,
        )
        txt_path.write_text(format_transcript(result, "txt"), encoding="utf-8")

    def _save_incremental_webm(self, buf_bytes: bytes) -> None:
        """Overwrite <session_id>_live.webm with all audio received so far."""
        if not buf_bytes:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        webm_path = self.output_dir / f"{self.session_id}_live.webm"
        webm_path.write_bytes(buf_bytes)

    def _write_final_outputs(self) -> list[Path]:
        """Write all requested output formats and return their paths."""
        if not self.segments:
            return []
        self.output_dir.mkdir(parents=True, exist_ok=True)
        result = TranscriptResult(
            segments=self.segments,
            provider_name=getattr(self.provider, "name", ""),
            model_id=self.opts.model_id,
        )
        output_files: list[Path] = []
        for fmt in self.opts.output_formats:
            path = self.output_dir / f"{self.session_id}.{fmt}"
            path.write_text(format_transcript(result, fmt), encoding="utf-8")
            output_files.append(path)
        return output_files


class LiveSessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, LiveSession] = {}

    def add(self, session: LiveSession) -> None:
        self._sessions[session.session_id] = session

    def get(self, session_id: str) -> LiveSession | None:
        return self._sessions.get(session_id)

    def remove(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
