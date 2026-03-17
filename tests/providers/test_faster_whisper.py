# tests/providers/test_faster_whisper.py
import pytest
from unittest.mock import MagicMock, patch, call
from pathlib import Path
from app.providers.faster_whisper import FasterWhisperProvider
from app.providers.base import TranscribeOptions


@pytest.fixture
def provider():
    return FasterWhisperProvider()


def test_models_list(provider):
    ids = [m.id for m in provider.models]
    assert "tiny" in ids
    assert "base" in ids
    assert "small" in ids
    assert "medium" in ids
    assert "large-v3" in ids


def test_is_available_when_package_missing():
    import app.providers.faster_whisper as mod
    original = mod.WhisperModel
    mod.WhisperModel = None
    p = FasterWhisperProvider()
    assert p.is_available() is False
    mod.WhisperModel = original


def test_is_available_when_package_present():
    import app.providers.faster_whisper as mod
    if mod.WhisperModel is not None:
        p = FasterWhisperProvider()
        assert p.is_available() is True


def test_install_deps_calls_pip(tmp_path):
    import subprocess
    p = FasterWhisperProvider()
    with patch("subprocess.check_call") as mock_call:
        p.install_deps()
        mock_call.assert_called_once()
        args = mock_call.call_args[0][0]
        assert "pip" in args
        assert "faster-whisper" in args


def test_install_deps_fires_callbacks(tmp_path):
    p = FasterWhisperProvider()
    calls = []
    with patch("subprocess.check_call"):
        p.install_deps(progress_callback=lambda prog, msg: calls.append((prog, msg)))
    assert len(calls) >= 2
    assert calls[0][0] == 0.0
    assert calls[-1][0] == 1.0


@pytest.mark.asyncio
async def test_transcribe_batch_single_chunk(tmp_path):
    dummy = tmp_path / "chunk_01.mp3"
    dummy.write_bytes(b"fake audio")

    mock_seg = MagicMock()
    mock_seg.start = 0.0
    mock_seg.end = 5.0
    mock_seg.text = " Hello world."

    mock_info = MagicMock()
    mock_info.duration = 10.0

    with patch("app.providers.faster_whisper.WhisperModel") as MockModel:
        instance = MockModel.return_value
        instance.transcribe.return_value = ([mock_seg], mock_info)

        p = FasterWhisperProvider()
        opts = TranscribeOptions(model_id="tiny")
        result = await p.transcribe_batch([dummy], opts)

    assert len(result.segments) == 1
    assert result.segments[0].text == "Hello world."
    assert result.segments[0].start == 0.0
    assert result.provider_name == "faster_whisper"


@pytest.mark.asyncio
async def test_transcribe_batch_offsets_second_chunk(tmp_path):
    chunk1 = tmp_path / "chunk_01.mp3"
    chunk2 = tmp_path / "chunk_02.mp3"
    chunk1.write_bytes(b"x")
    chunk2.write_bytes(b"x")

    def make_seg(start, end, text):
        s = MagicMock()
        s.start, s.end, s.text = start, end, text
        return s

    info1 = MagicMock(); info1.duration = 10.0
    info2 = MagicMock(); info2.duration = 8.0

    with patch("app.providers.faster_whisper.WhisperModel") as MockModel:
        instance = MockModel.return_value
        instance.transcribe.side_effect = [
            ([make_seg(0.0, 5.0, " First.")], info1),
            ([make_seg(0.0, 4.0, " Second.")], info2),
        ]
        p = FasterWhisperProvider()
        opts = TranscribeOptions(model_id="tiny")
        result = await p.transcribe_batch([chunk1, chunk2], opts)

    assert len(result.segments) == 2
    assert result.segments[1].start == pytest.approx(10.0)
    assert result.segments[1].end == pytest.approx(14.0)


@pytest.mark.asyncio
async def test_transcribe_raises_when_unavailable(tmp_path):
    import app.providers.faster_whisper as mod
    original = mod.WhisperModel
    mod.WhisperModel = None
    dummy = tmp_path / "chunk_01.mp3"
    dummy.write_bytes(b"x")
    p = FasterWhisperProvider()
    with pytest.raises(RuntimeError, match="faster-whisper"):
        await p.transcribe_batch([dummy], TranscribeOptions())
    mod.WhisperModel = original
