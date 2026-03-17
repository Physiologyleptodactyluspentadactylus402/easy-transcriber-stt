import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path
from app.providers.ollama import OllamaProvider
from app.providers.base import TranscribeOptions


def test_install_deps_is_noop():
    p = OllamaProvider()
    p.install_deps()  # must not raise


def test_is_available_false_when_daemon_unreachable():
    with patch("httpx.get", side_effect=Exception("connection refused")):
        p = OllamaProvider()
        assert p.is_available() is False


def test_is_available_true_when_daemon_running():
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"models": [{"name": "whisper-cli"}]}
    with patch("httpx.get", return_value=mock_resp):
        p = OllamaProvider()
        assert p.is_available() is True
        assert any(m.id == "whisper-cli" for m in p.models)


def test_models_empty_when_daemon_unreachable():
    with patch("httpx.get", side_effect=Exception("refused")):
        p = OllamaProvider()
        assert p.models == []


@pytest.mark.asyncio
async def test_transcribe_batch_sends_audio(tmp_path):
    dummy = tmp_path / "chunk_01.mp3"
    dummy.write_bytes(b"fakeaudio")

    mock_get = MagicMock()
    mock_get.raise_for_status.return_value = None
    mock_get.json.return_value = {"models": [{"name": "whisper-cli"}]}

    mock_post = MagicMock()
    mock_post.raise_for_status.return_value = None
    mock_post.json.return_value = {"response": "Ciao mondo."}

    with patch("httpx.get", return_value=mock_get), \
         patch("httpx.post", return_value=mock_post):
        p = OllamaProvider()
        opts = TranscribeOptions(model_id="whisper-cli")
        result = await p.transcribe_batch([dummy], opts)

    assert len(result.segments) == 1
    assert result.segments[0].text == "Ciao mondo."
    assert result.provider_name == "ollama"


@pytest.mark.asyncio
async def test_transcribe_raises_when_unavailable(tmp_path):
    dummy = tmp_path / "chunk_01.mp3"
    dummy.write_bytes(b"x")
    with patch("httpx.get", side_effect=Exception("refused")):
        p = OllamaProvider()
        with pytest.raises(RuntimeError, match="Ollama"):
            await p.transcribe_batch([dummy], TranscribeOptions())
