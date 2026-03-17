import pytest
from app.providers.base import (
    HardwareHint, ModelInfo, TranscribeOptions, Segment,
    TranscriptResult, BaseProvider
)


def test_hardware_hint_values():
    assert HardwareHint.CPU == "cpu"
    assert HardwareHint.CLOUD == "cloud"


def test_model_info_defaults():
    m = ModelInfo(
        id="test-model",
        name="Test Model",
        description="A test model",
        hardware_hint=HardwareHint.CPU,
    )
    assert m.supports_live is False
    assert m.supports_speaker_labels is False
    assert m.supports_timestamps is True


def test_transcribe_options_defaults():
    opts = TranscribeOptions()
    assert opts.language is None
    assert opts.prompt == ""
    assert opts.speaker_labels is False
    assert opts.output_formats == ["txt"]
    assert opts.chunk_size_sec == 600


def test_segment_no_speaker():
    s = Segment(start=0.0, end=5.0, text="Hello world")
    assert s.speaker is None


def test_transcript_result_empty():
    r = TranscriptResult(segments=[])
    assert r.language_detected is None
    assert r.provider_name == ""


def test_transcribe_options_model_id_default():
    opts = TranscribeOptions()
    assert opts.model_id == "whisper-1"


def test_base_provider_is_abstract():
    with pytest.raises(TypeError):
        BaseProvider()  # cannot instantiate abstract class
