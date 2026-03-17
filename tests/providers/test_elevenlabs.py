import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from app.providers.elevenlabs import ElevenLabsProvider
from app.providers.base import TranscribeOptions


@pytest.fixture
def provider():
    return ElevenLabsProvider(api_key="el-test-key")


def test_models_include_scribe_v2(provider):
    ids = [m.id for m in provider.models]
    assert "scribe_v2" in ids


def test_scribe_v2_supports_speaker_labels(provider):
    m = next(m for m in provider.models if m.id == "scribe_v2")
    assert m.supports_speaker_labels is True


def test_scribe_v2_realtime_supports_live(provider):
    m = next(m for m in provider.models if m.id == "scribe_v2_realtime")
    assert m.supports_live is True


def test_is_available_with_key(provider):
    assert provider.is_available() is True


def test_is_available_without_key():
    assert ElevenLabsProvider(api_key=None).is_available() is False


def test_install_deps_is_noop(provider):
    provider.install_deps()


@pytest.mark.asyncio
async def test_transcribe_batch_returns_segments(provider, tmp_path):
    dummy = tmp_path / "chunk_01.mp3"
    dummy.write_bytes(b"x")

    mock_result = MagicMock()
    mock_result.words = None
    mock_result.utterances = None
    mock_result.text = "Hello from ElevenLabs."
    mock_result.language_code = "en"

    with patch("app.providers.elevenlabs.ElevenLabs") as MockClient:
        instance = MockClient.return_value
        instance.speech_to_text.convert.return_value = mock_result

        opts = TranscribeOptions(model_id="scribe_v2")
        result = await provider.transcribe_batch([dummy], opts)

    assert len(result.segments) == 1
    assert result.segments[0].text == "Hello from ElevenLabs."
    assert result.provider_name == "elevenlabs"
