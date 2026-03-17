from __future__ import annotations
from pathlib import Path
from pydub import AudioSegment


def split_audio(
    input_file: Path,
    output_dir: Path,
    chunk_size_sec: int = 600,
    overlap_sec: int = 1,
) -> list[Path]:
    """
    Split an audio file into overlapping MP3 chunks.

    Each chunk overlaps with the next by overlap_sec seconds to avoid
    cutting words at boundaries. Returns sorted list of chunk paths.
    """
    input_file = Path(input_file)
    if not input_file.exists():
        raise FileNotFoundError(f"Audio file not found: {input_file}")

    output_dir = Path(output_dir)
    # Clean existing chunks
    if output_dir.exists():
        for f in output_dir.iterdir():
            if f.is_file():
                f.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    audio = AudioSegment.from_file(str(input_file))
    total_ms = len(audio)
    chunk_ms = chunk_size_sec * 1000
    overlap_ms = overlap_sec * 1000
    step_ms = chunk_ms - overlap_ms

    chunks: list[Path] = []
    chunk_num = 1
    start_ms = 0

    while start_ms < total_ms:
        end_ms = min(start_ms + chunk_ms, total_ms)
        chunk = audio[start_ms:end_ms]

        if len(chunk) < 100:  # skip truly tiny tail fragments (< 100ms)
            break

        filename = output_dir / f"chunk_{chunk_num:02d}.mp3"
        chunk.export(str(filename), format="mp3", bitrate="192k")
        chunks.append(filename)

        chunk_num += 1
        start_ms += step_ms
        # Don't continue if the remaining audio is smaller than a full chunk
        if total_ms - end_ms < chunk_ms:
            break

    return chunks
