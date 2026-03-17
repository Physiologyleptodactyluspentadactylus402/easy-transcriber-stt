import pytest
from pathlib import Path
from app.core.audio import split_audio


def test_split_produces_two_chunks_for_120s_file(audio_120s, tmp_path):
    """120s audio with 60s chunks + 1s overlap should produce 2 chunks."""
    chunks = split_audio(audio_120s, tmp_path, chunk_size_sec=60, overlap_sec=1)
    assert len(chunks) == 2


def test_split_single_chunk_for_short_file(audio_30s, tmp_path):
    """30s audio with 60s chunk size produces exactly 1 chunk."""
    chunks = split_audio(audio_30s, tmp_path, chunk_size_sec=60)
    assert len(chunks) == 1


def test_chunks_are_mp3(audio_120s, tmp_path):
    chunks = split_audio(audio_120s, tmp_path, chunk_size_sec=60)
    for chunk in chunks:
        assert chunk.suffix == ".mp3"


def test_chunks_exist_on_disk(audio_120s, tmp_path):
    chunks = split_audio(audio_120s, tmp_path, chunk_size_sec=60)
    for chunk in chunks:
        assert chunk.exists()
        assert chunk.stat().st_size > 100  # not empty


def test_chunk_numbering_is_sequential(audio_120s, tmp_path):
    chunks = split_audio(audio_120s, tmp_path, chunk_size_sec=60)
    names = [c.stem for c in chunks]
    assert names == ["chunk_01", "chunk_02"]


def test_output_dir_is_cleaned_before_split(audio_120s, tmp_path):
    """Old chunk files are removed before a new split."""
    old_file = tmp_path / "chunk_99.mp3"
    old_file.write_bytes(b"old")
    split_audio(audio_120s, tmp_path, chunk_size_sec=60)
    assert not old_file.exists()


def test_split_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        split_audio(Path("/nonexistent/file.mp3"), tmp_path)
