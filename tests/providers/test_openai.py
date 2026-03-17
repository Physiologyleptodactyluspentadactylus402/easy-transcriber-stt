import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path
from app.providers.openai import OpenAIProvider
from app.providers.base import TranscribeOptions


@pytest.fixture
def provider():
    return OpenAIProvider(api_key="sk-test-key")


def test_models_list_not_empty(provider):
    assert len(provider.models) >= 3


def test_model_ids(provider):
    ids = [m.id for m in provider.models]
    assert "whisper-1" in ids
    assert "gpt-4o-transcribe" in ids


def test_is_available_with_key(provider):
    # has key set, so should report available (key validity checked at transcription time)
    assert provider.is_available() is True


def test_is_available_without_key():
    p = OpenAIProvider(api_key=None)
    assert p.is_available() is False


def test_install_deps_is_noop(provider):
    # OpenAI provider has no heavy deps — install_deps does nothing
    provider.install_deps()  # should not raise


@pytest.mark.asyncio
async def test_transcribe_batch_returns_result(provider, tmp_path):
    # Create a dummy audio file
    dummy_audio = tmp_path / "chunk_01.mp3"
    dummy_audio.write_bytes(b"fake audio data")

    mock_response = MagicMock()
    mock_response.text = "Hello world."

    with patch("app.providers.openai.OpenAI") as MockClient:
        instance = MockClient.return_value
        instance.audio.transcriptions.create.return_value = mock_response

        from app.providers.base import TranscribeOptions
        opts = TranscribeOptions(output_formats=["txt"])
        result = await provider.transcribe_batch([dummy_audio], opts)

    assert len(result.segments) == 1
    assert "Hello world." in result.segments[0].text
    assert result.provider_name == "openai"


@pytest.mark.asyncio
async def test_transcribe_batch_uses_prompt(provider, tmp_path):
    dummy = tmp_path / "chunk_01.mp3"
    dummy.write_bytes(b"x")
    mock_response = MagicMock()
    mock_response.text = "test"

    with patch("app.providers.openai.OpenAI") as MockClient:
        instance = MockClient.return_value
        instance.audio.transcriptions.create.return_value = mock_response
        opts = TranscribeOptions(prompt="DNA polymerase")
        await provider.transcribe_batch([dummy], opts)
        call_kwargs = instance.audio.transcriptions.create.call_args.kwargs
        assert call_kwargs.get("prompt") == "DNA polymerase"
