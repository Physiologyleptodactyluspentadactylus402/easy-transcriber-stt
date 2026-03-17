import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from app.providers.qwen3_asr import Qwen3ASRProvider
from app.providers.base import TranscribeOptions


@pytest.fixture
def provider():
    return Qwen3ASRProvider()


def test_models_list(provider):
    ids = [m.id for m in provider.models]
    assert "0.6b" in ids
    assert "1.7b" in ids


def test_0_6b_has_no_timestamps(provider):
    m = next(m for m in provider.models if m.id == "0.6b")
    assert m.supports_timestamps is False


def test_1_7b_has_timestamps(provider):
    m = next(m for m in provider.models if m.id == "1.7b")
    assert m.supports_timestamps is True


def test_is_available_when_package_missing():
    import app.providers.qwen3_asr as mod
    original = mod.pipeline
    mod.pipeline = None
    p = Qwen3ASRProvider()
    assert p.is_available() is False
    mod.pipeline = original


def test_install_deps_calls_pip():
    p = Qwen3ASRProvider()
    calls = []
    with patch("subprocess.check_call") as mock_cc:
        p.install_deps(progress_callback=lambda prog, msg: calls.append(prog))
    mock_cc.assert_called_once()
    args = mock_cc.call_args[0][0]
    assert "transformers" in args or "torch" in args
    assert calls[0] == 0.0
    assert calls[-1] == 1.0


@pytest.mark.asyncio
async def test_transcribe_batch_no_timestamps(tmp_path):
    dummy = tmp_path / "chunk_01.mp3"
    dummy.write_bytes(b"fake")

    mock_result = {"text": " Ciao mondo."}

    with patch("app.providers.qwen3_asr.pipeline") as mock_pipeline:
        mock_pipe = MagicMock(return_value=mock_result)
        mock_pipeline.return_value = mock_pipe

        p = Qwen3ASRProvider()
        opts = TranscribeOptions(model_id="0.6b")
        result = await p.transcribe_batch([dummy], opts)

    assert len(result.segments) == 1
    assert result.segments[0].text == "Ciao mondo."
    assert result.provider_name == "qwen3_asr"


@pytest.mark.asyncio
async def test_transcribe_batch_with_timestamps(tmp_path):
    dummy = tmp_path / "chunk_01.mp3"
    dummy.write_bytes(b"fake")

    mock_result = {
        "chunks": [
            {"text": " First segment.", "timestamp": (0.0, 4.0)},
            {"text": " Second segment.", "timestamp": (4.0, 8.5)},
        ]
    }

    with patch("app.providers.qwen3_asr.pipeline") as mock_pipeline:
        mock_pipe = MagicMock(return_value=mock_result)
        mock_pipeline.return_value = mock_pipe

        p = Qwen3ASRProvider()
        opts = TranscribeOptions(model_id="1.7b")
        result = await p.transcribe_batch([dummy], opts)

    assert len(result.segments) == 2
    assert result.segments[0].start == 0.0
    assert result.segments[0].end == 4.0
    assert result.segments[1].start == 4.0
    assert result.segments[1].end == 8.5


@pytest.mark.asyncio
async def test_transcribe_raises_when_unavailable(tmp_path):
    import app.providers.qwen3_asr as mod
    original = mod.pipeline
    mod.pipeline = None
    dummy = tmp_path / "chunk_01.mp3"
    dummy.write_bytes(b"x")
    p = Qwen3ASRProvider()
    with pytest.raises(RuntimeError, match="transformers"):
        await p.transcribe_batch([dummy], TranscribeOptions())
    mod.pipeline = original
