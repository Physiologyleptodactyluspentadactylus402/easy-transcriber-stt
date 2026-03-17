from __future__ import annotations
from app.providers.base import TranscriptResult

SUPPORTED_FORMATS = ["txt", "srt", "vtt", "md"]


def _ts_srt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds % 1) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _ts_vtt(seconds: float) -> str:
    return _ts_srt(seconds).replace(",", ".")


def _ts_md(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_transcript(result: TranscriptResult, fmt: str) -> str:
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format: {fmt}. Choose from {SUPPORTED_FORMATS}")
    if fmt == "txt":
        return _to_txt(result)
    if fmt == "srt":
        return _to_srt(result)
    if fmt == "vtt":
        return _to_vtt(result)
    if fmt == "md":
        return _to_md(result)


def _to_txt(result: TranscriptResult) -> str:
    lines = []
    for seg in result.segments:
        prefix = f"[{seg.speaker}] " if seg.speaker else ""
        lines.append(f"{prefix}{seg.text}")
    return "\n\n".join(lines)


def _to_srt(result: TranscriptResult) -> str:
    blocks = []
    for i, seg in enumerate(result.segments, 1):
        prefix = f"[{seg.speaker}] " if seg.speaker else ""
        blocks.append(
            f"{i}\n{_ts_srt(seg.start)} --> {_ts_srt(seg.end)}\n{prefix}{seg.text}"
        )
    return "\n\n".join(blocks)


def _to_vtt(result: TranscriptResult) -> str:
    blocks = ["WEBVTT\n"]
    for i, seg in enumerate(result.segments, 1):
        prefix = f"[{seg.speaker}] " if seg.speaker else ""
        blocks.append(
            f"{i}\n{_ts_vtt(seg.start)} --> {_ts_vtt(seg.end)}\n{prefix}{seg.text}"
        )
    return "\n\n".join(blocks)


def _to_md(result: TranscriptResult) -> str:
    lines = []
    for seg in result.segments:
        lines.append(f"## {_ts_md(seg.start)}")
        if seg.speaker:
            lines.append(f"**{seg.speaker}:** {seg.text}")
        else:
            lines.append(seg.text)
        lines.append("")
    return "\n".join(lines)


def merge_transcripts(
    results: list[tuple[TranscriptResult, str]], fmt: str
) -> str:
    """Concatenate multiple TranscriptResults into one output string."""
    parts = []
    for result, filename in results:
        parts.append(f"---\n# {filename}\n\n{format_transcript(result, fmt)}")
    return "\n\n".join(parts)
