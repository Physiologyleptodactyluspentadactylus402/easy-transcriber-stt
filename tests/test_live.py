# tests/test_live.py
import pytest
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch
from app.core.live import LiveSession, LiveSessionManager
from app.providers.base import TranscribeOptions, TranscriptResult, Segment


def _mock_provider(segments=None):
    provider = MagicMock()
    result = TranscriptResult(
        segments=segments or [Segment(start=0.0, end=5.0, text="Hello world.")],
        provider_name="faster_whisper",
        model_id="tiny",
    )
    provider.transcribe_batch = AsyncMock(return_value=result)
    return provider


def test_live_session_manager_add_and_get():
    mgr = LiveSessionManager()
    session = LiveSession(
        session_id="abc",
        provider=_mock_provider(),
        opts=TranscribeOptions(),
        output_dir=Path("/tmp"),
    )
    mgr.add(session)
    assert mgr.get("abc") is session
    assert mgr.get("nonexistent") is None


def test_live_session_manager_remove():
    mgr = LiveSessionManager()
    session = LiveSession("xyz", _mock_provider(), TranscribeOptions(), Path("/tmp"))
    mgr.add(session)
    mgr.remove("xyz")
    assert mgr.get("xyz") is None


def test_live_session_add_chunk_accumulates(tmp_path):
    s = LiveSession("s1", _mock_provider(), TranscribeOptions(), tmp_path)
    s.add_chunk(b"AAAA")
    s.add_chunk(b"BBBB")
    assert len(s._buffer) == 8


@pytest.mark.asyncio
async def test_live_session_stop_returns_output_files(tmp_path):
    provider = _mock_provider([
        Segment(start=0.0, end=3.0, text="First."),
        Segment(start=3.0, end=6.0, text="Second."),
    ])
    s = LiveSession("s2", provider, TranscribeOptions(output_formats=["txt", "srt"]), tmp_path)

    ws_manager = MagicMock()
    ws_manager.broadcast_global = AsyncMock()

    # Simulate receiving audio and then stopping
    s.add_chunk(b"fake_webm_data")

    # Patch _transcribe_buffer to avoid needing ffmpeg in tests
    async def fake_transcribe(buf_bytes, ws_mgr):
        for seg in provider.transcribe_batch.return_value.segments:
            s.segments.append(seg)
            await ws_mgr.broadcast_global({"type": "segment", "text": seg.text})

    with patch.object(s, "_transcribe_buffer", new=fake_transcribe):
        output_files = await s.stop(ws_manager)

    assert len(output_files) >= 1
    txt_file = next((f for f in output_files if f.suffix == ".txt"), None)
    assert txt_file is not None
    content = txt_file.read_text(encoding="utf-8")
    assert "First." in content


@pytest.mark.asyncio
async def test_live_session_incremental_save(tmp_path):
    provider = _mock_provider([Segment(start=0.0, end=5.0, text="Test.")])
    s = LiveSession("s3", provider, TranscribeOptions(output_formats=["txt"]), tmp_path)
    s.segments.append(Segment(start=0.0, end=5.0, text="Test."))
    s._save_incremental_txt()
    txt = tmp_path / "s3_live.txt"
    assert txt.exists()
    assert "Test." in txt.read_text(encoding="utf-8")
