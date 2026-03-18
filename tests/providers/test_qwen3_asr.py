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


def test_no_timestamps(provider):
    """Both models report supports_timestamps=False (qwen-asr API doesn't return them)."""
    for m in provider.models:
        assert m.supports_timestamps is False


def test_is_available_when_package_missing():
    import app.providers.qwen3_asr as mod
    original = mod._AVAILABLE
    mod._AVAILABLE = False
    p = Qwen3ASRProvider()
    assert p.is_available() is False
    mod._AVAILABLE = original


def test_install_deps_calls_pip():
    p = Qwen3ASRProvider()
    calls = []
    # Mock both subprocess.check_call AND the re-import that follows
    with patch("subprocess.check_call") as mock_cc, \
         patch.dict("sys.modules", {"qwen_asr": MagicMock()}):
        p.install_deps(progress_callback=lambda prog, msg: calls.append(prog))
    # May call pip once (CPU) or twice (XPU torch + qwen-asr)
    assert mock_cc.call_count >= 1
    all_args = [c[0][0] for c in mock_cc.call_args_list]
    # qwen-asr must appear in at least one pip call
    assert any("qwen-asr" in args for args in all_args)
    assert calls[0] == 0.0
    assert calls[-1] == 1.0


@pytest.mark.asyncio
async def test_transcribe_batch(tmp_path):
    dummy = tmp_path / "chunk_01.mp3"
    dummy.write_bytes(b"fake")

    # Mock a TranscriptionResult-like object
    mock_result = MagicMock()
    mock_result.text = " Ciao mondo."
    mock_result.language = "Italian"

    mock_model = MagicMock()
    mock_model.transcribe.return_value = [mock_result]

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False

    with patch("app.providers.qwen3_asr.Qwen3ASRModel") as mock_cls, \
         patch("app.providers.qwen3_asr._AVAILABLE", True), \
         patch("app.providers.qwen3_asr.torch", mock_torch):
        mock_cls.from_pretrained.return_value = mock_model

        p = Qwen3ASRProvider()
        opts = TranscribeOptions(model_id="0.6b")
        result = await p.transcribe_batch([dummy], opts)

    assert len(result.segments) == 1
    assert result.segments[0].text == "Ciao mondo."
    assert result.provider_name == "qwen3_asr"


@pytest.mark.asyncio
async def test_transcribe_batch_empty_result(tmp_path):
    dummy = tmp_path / "chunk_01.mp3"
    dummy.write_bytes(b"fake")

    mock_model = MagicMock()
    mock_model.transcribe.return_value = []

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False

    with patch("app.providers.qwen3_asr.Qwen3ASRModel") as mock_cls, \
         patch("app.providers.qwen3_asr._AVAILABLE", True), \
         patch("app.providers.qwen3_asr.torch", mock_torch):
        mock_cls.from_pretrained.return_value = mock_model

        p = Qwen3ASRProvider()
        opts = TranscribeOptions(model_id="1.7b")
        result = await p.transcribe_batch([dummy], opts)

    assert len(result.segments) == 1
    assert result.segments[0].text == ""


@pytest.mark.asyncio
async def test_transcribe_raises_when_unavailable(tmp_path):
    import app.providers.qwen3_asr as mod
    original = mod._AVAILABLE
    mod._AVAILABLE = False
    dummy = tmp_path / "chunk_01.mp3"
    dummy.write_bytes(b"x")
    p = Qwen3ASRProvider()
    with pytest.raises(RuntimeError, match="qwen-asr"):
        await p.transcribe_batch([dummy], TranscribeOptions())
    mod._AVAILABLE = original
