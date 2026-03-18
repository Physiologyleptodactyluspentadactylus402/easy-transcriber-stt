import pytest
from pathlib import Path
from pydub import AudioSegment
from pydub.generators import Sine


@pytest.fixture(scope="session")
def audio_tone_10s(tmp_path_factory) -> Path:
    """A 10-second 440 Hz sine wave at -20 dBFS, exported as WAV."""
    tmp = tmp_path_factory.mktemp("audio")
    path = tmp / "tone_10s.wav"
    # Sine generator produces a full-scale tone; reduce by 20 dB to reach -20 dBFS.
    tone = Sine(440).to_audio_segment(duration=10_000).apply_gain(-20)
    tone.export(str(path), format="wav")
    return path


@pytest.fixture(scope="session")
def audio_120s(tmp_path_factory) -> Path:
    """A 120-second silent MP3 audio file for testing."""
    tmp = tmp_path_factory.mktemp("audio")
    path = tmp / "test_120s.mp3"
    audio = AudioSegment.silent(duration=120_000)  # 120s in ms
    audio.export(str(path), format="mp3")
    return path


@pytest.fixture(scope="session")
def audio_30s(tmp_path_factory) -> Path:
    """A 30-second silent MP3 for testing single-chunk scenarios."""
    tmp = tmp_path_factory.mktemp("audio")
    path = tmp / "test_30s.mp3"
    audio = AudioSegment.silent(duration=30_000)
    audio.export(str(path), format="mp3")
    return path
