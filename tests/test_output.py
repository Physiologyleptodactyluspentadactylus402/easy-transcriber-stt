import pytest
from app.providers.base import Segment, TranscriptResult
from app.core.output import format_transcript, SUPPORTED_FORMATS


SEGMENTS = [
    Segment(start=0.0, end=5.5, text="Hello world."),
    Segment(start=5.5, end=11.0, text="This is a test."),
]

SEGMENTS_WITH_SPEAKERS = [
    Segment(start=0.0, end=5.5, text="Hello world.", speaker="Speaker 1"),
    Segment(start=5.5, end=11.0, text="This is a test.", speaker="Speaker 2"),
]

RESULT = TranscriptResult(segments=SEGMENTS, provider_name="test", model_id="m1")
RESULT_SPEAKERS = TranscriptResult(segments=SEGMENTS_WITH_SPEAKERS)


def test_supported_formats():
    assert set(SUPPORTED_FORMATS) == {"txt", "srt", "vtt", "md"}


def test_txt_basic():
    out = format_transcript(RESULT, "txt")
    assert "Hello world." in out
    assert "This is a test." in out


def test_txt_speaker_labels():
    out = format_transcript(RESULT_SPEAKERS, "txt")
    assert "[Speaker 1] Hello world." in out
    assert "[Speaker 2] This is a test." in out


def test_srt_contains_sequence_numbers():
    out = format_transcript(RESULT, "srt")
    assert "1\n" in out
    assert "2\n" in out


def test_srt_timestamp_format():
    out = format_transcript(RESULT, "srt")
    assert "00:00:00,000 --> 00:00:05,500" in out


def test_vtt_header():
    out = format_transcript(RESULT, "vtt")
    assert out.startswith("WEBVTT")


def test_vtt_timestamp_format():
    out = format_transcript(RESULT, "vtt")
    assert "00:00:00.000 --> 00:00:05.500" in out


def test_md_has_timestamp_headers():
    out = format_transcript(RESULT, "md")
    assert "## 00:00:00" in out


def test_md_speaker_bold():
    out = format_transcript(RESULT_SPEAKERS, "md")
    assert "**Speaker 1:**" in out


def test_unsupported_format_raises():
    with pytest.raises(ValueError, match="Unsupported format"):
        format_transcript(RESULT, "pdf")


def test_merge_two_results():
    from app.core.output import merge_transcripts
    merged = merge_transcripts(
        [(RESULT, "file_a.mp3"), (RESULT, "file_b.mp3")], "txt"
    )
    assert "file_a.mp3" in merged
    assert "file_b.mp3" in merged
    assert merged.count("Hello world.") == 2
