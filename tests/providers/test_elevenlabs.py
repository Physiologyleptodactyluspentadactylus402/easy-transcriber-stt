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


def _make_word(text, start, end, speaker_id=None, word_type="word"):
    w = MagicMock()
    w.text = text
    w.start = start
    w.end = end
    w.speaker_id = speaker_id
    w.type = word_type
    return w


@pytest.mark.asyncio
async def test_transcribe_batch_returns_segments(provider, tmp_path):
    """Fallback path: no word-level data, use full text."""
    dummy = tmp_path / "chunk_01.mp3"
    dummy.write_bytes(b"x")

    mock_result = MagicMock()
    mock_result.words = None
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


@pytest.mark.asyncio
async def test_transcribe_batch_with_words(provider, tmp_path):
    """Word-level timing is grouped into segments by pauses."""
    dummy = tmp_path / "chunk_01.mp3"
    dummy.write_bytes(b"x")

    mock_result = MagicMock()
    mock_result.words = [
        _make_word("Hello", 0.0, 0.5),
        _make_word("world", 0.6, 1.0),
        # big pause → new segment
        _make_word("Second", 5.0, 5.5),
        _make_word("sentence", 5.6, 6.0),
    ]
    mock_result.text = "Hello world Second sentence"
    mock_result.language_code = "en"

    with patch("app.providers.elevenlabs.ElevenLabs") as MockClient:
        instance = MockClient.return_value
        instance.speech_to_text.convert.return_value = mock_result

        opts = TranscribeOptions(model_id="scribe_v2")
        result = await provider.transcribe_batch([dummy], opts)

    assert len(result.segments) == 2
    assert result.segments[0].text == "Hello world"
    assert result.segments[1].text == "Second sentence"


@pytest.mark.asyncio
async def test_transcribe_batch_with_diarization(provider, tmp_path):
    """Speaker labels split segments by speaker_id."""
    dummy = tmp_path / "chunk_01.mp3"
    dummy.write_bytes(b"x")

    mock_result = MagicMock()
    mock_result.words = [
        _make_word("Hi", 0.0, 0.3, speaker_id="A"),
        _make_word("there", 0.4, 0.7, speaker_id="A"),
        _make_word("Hello", 0.8, 1.1, speaker_id="B"),  # speaker change
        _make_word("back", 1.2, 1.5, speaker_id="B"),
    ]
    mock_result.text = "Hi there Hello back"
    mock_result.language_code = "en"

    with patch("app.providers.elevenlabs.ElevenLabs") as MockClient:
        instance = MockClient.return_value
        instance.speech_to_text.convert.return_value = mock_result

        opts = TranscribeOptions(model_id="scribe_v2", speaker_labels=True)
        result = await provider.transcribe_batch([dummy], opts)

    assert len(result.segments) == 2
    assert result.segments[0].text == "Hi there"
    assert result.segments[0].speaker == "Speaker A"
    assert result.segments[1].text == "Hello back"
    assert result.segments[1].speaker == "Speaker B"


@pytest.mark.asyncio
async def test_transcribe_batch_api_call_params(provider, tmp_path):
    """Verify the correct parameters are passed to the API."""
    dummy = tmp_path / "chunk_01.mp3"
    dummy.write_bytes(b"x")

    mock_result = MagicMock()
    mock_result.words = []
    mock_result.text = "test"
    mock_result.language_code = "it"

    with patch("app.providers.elevenlabs.ElevenLabs") as MockClient:
        instance = MockClient.return_value
        instance.speech_to_text.convert.return_value = mock_result

        opts = TranscribeOptions(
            model_id="scribe_v2",
            language="it",
            speaker_labels=True,
            prompt="machine learning,neural network",
        )
        await provider.transcribe_batch([dummy], opts)

        call_kwargs = instance.speech_to_text.convert.call_args.kwargs
        assert call_kwargs["model_id"] == "scribe_v2"
        assert call_kwargs["language_code"] == "it"
        assert call_kwargs["diarize"] is True
        assert call_kwargs["timestamps_granularity"] == "word"
        assert call_kwargs["keyterms"] == ["machine learning", "neural network"]
        assert "keyterm_prompting_enabled" not in call_kwargs
        assert "additional_formats" not in call_kwargs
