# Transcriber MVP Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fully working local web app for batch audio transcription using OpenAI and ElevenLabs, with job queue, history, and complete Alpine.js + Tailwind frontend.

**Architecture:** FastAPI backend with WebSocket progress streaming. Frontend served as Jinja2 templates with vendored Alpine.js + Tailwind CSS (committed to repo). Audio split into chunks by pydub, transcribed by provider, assembled into txt/srt/vtt/md output. SQLite history. Settings in settings.json.

**Tech Stack:** Python 3.10+, FastAPI, uvicorn, pydub, openai SDK, elevenlabs SDK, Alpine.js 3.x, Tailwind CSS 3.x, SQLite (stdlib), pytest, httpx

**Sub-plan note:** This is Plan 1 of 3. Plan 2 adds local providers (faster-whisper, Qwen3-ASR, Ollama). Plan 3 adds live transcription mode.

---

## Chunk 1: Foundation

### Task 1: Repo Cleanup

**Files:**
- Delete: `main_ui.py`, `transcribe2.py`, `split_audio.py`, `AudioTranscriber.spec`, `AppIcon.icns`, `start.bh.txt`
- Delete dirs: `build/`, `dist/`, `__pycache__/`
- Modify: `.gitignore` (create/replace)

- [ ] **Step 1: Delete obsolete files**

```bash
cd "C:/Users/agius/Desktop/Trascrizioni"
rm -f main_ui.py transcribe2.py split_audio.py AudioTranscriber.spec AppIcon.icns start.bh.txt
rm -rf build dist __pycache__
```

- [ ] **Step 2: Write `.gitignore`**

Create `.gitignore`:
```
# Python
__pycache__/
*.pyc
*.pyo
.venv/
venv/
*.egg-info/

# App runtime
.env
API KEY.txt
settings.json
audio_chunks/
app/transcriber.db

# OS
.DS_Store
Thumbs.db

# IDE
.idea/
.vscode/
```

- [ ] **Step 3: Commit cleanup**

```bash
git init   # if not already a git repo
git add .gitignore
git rm --cached -r --ignore-unmatch main_ui.py transcribe2.py split_audio.py AudioTranscriber.spec AppIcon.icns start.bh.txt build dist __pycache__ 2>/dev/null || true
git add -A
git commit -m "chore: clean up legacy PyInstaller/tkinter files"
```

---

### Task 2: Project Skeleton

**Files:**
- Create: `app/__init__.py`, `app/core/__init__.py`, `app/providers/__init__.py`
- Create: `app/static/.gitkeep`, `app/templates/.gitkeep`, `app/locales/.gitkeep`
- Create: `tests/__init__.py`, `tests/providers/__init__.py`
- Create: `requirements.txt`, `requirements-local.txt`, `.env.example`

- [ ] **Step 1: Create directory tree**

```bash
mkdir -p app/core app/providers app/static app/templates app/locales
mkdir -p tests/providers
touch app/__init__.py app/core/__init__.py app/providers/__init__.py
touch tests/__init__.py tests/providers/__init__.py
```

- [ ] **Step 2: Write `requirements.txt`**

```
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
python-dotenv>=1.0.1
pydub>=0.25.1
openai>=1.50.0
elevenlabs>=1.0.0
jinja2>=3.1.4
python-multipart>=0.0.9
httpx>=0.27.0
pytest>=8.0.0
pytest-asyncio>=0.23.0
```

- [ ] **Step 3: Write `requirements-local.txt`**

```
# Installed on-demand when user selects a local model
faster-whisper>=1.0.0
# torch and transformers installed separately per platform — see SETUP.md
```

- [ ] **Step 4: Write `.env.example`**

```
# OpenAI API key — get one at https://platform.openai.com/api-keys
OPENAI_API_KEY=sk-...

# ElevenLabs API key — get one at https://elevenlabs.io/app/settings/api-keys
ELEVENLABS_API_KEY=...
```

- [ ] **Step 5: Commit skeleton**

```bash
git add .
git commit -m "chore: scaffold project directory structure"
```

---

### Task 3: Core Data Types

**Files:**
- Create: `app/providers/base.py`
- Create: `tests/test_base.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_base.py`:
```python
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


def test_base_provider_is_abstract():
    with pytest.raises(TypeError):
        BaseProvider()  # cannot instantiate abstract class
```

- [ ] **Step 2: Run test — expect failure**

```bash
cd "C:/Users/agius/Desktop/Trascrizioni"
python -m pytest tests/test_base.py -v
```
Expected: `ModuleNotFoundError: No module named 'app'`

- [ ] **Step 3: Write `app/providers/base.py`**

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import AsyncGenerator, Literal
import time
import uuid


class HardwareHint(str, Enum):
    CPU = "cpu"
    CPU_RECOMMENDED = "cpu_recommended"
    GPU_OPTIONAL = "gpu_optional"
    GPU_RECOMMENDED = "gpu_recommended"
    CLOUD = "cloud"


@dataclass
class ModelInfo:
    id: str
    name: str
    description: str
    hardware_hint: HardwareHint
    supports_live: bool = False
    supports_speaker_labels: bool = False
    supports_timestamps: bool = True


@dataclass
class TranscribeOptions:
    language: str | None = None
    prompt: str = ""
    speaker_labels: bool = False
    output_formats: list[str] = field(default_factory=lambda: ["txt"])
    chunk_size_sec: int = 600


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str | None = None


@dataclass
class TranscriptResult:
    segments: list[Segment]
    language_detected: str | None = None
    provider_name: str = ""
    model_id: str = ""


@dataclass
class Job:
    input_files: list[Path]
    opts: TranscribeOptions
    provider_name: str
    model_id: str
    status: Literal["pending", "running", "done", "error", "cancelled"]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    progress: float = 0.0
    error_message: str | None = None
    created_at: float = field(default_factory=time.time)
    output_files: list[Path] = field(default_factory=list)
    merge_output: bool = False


class BaseProvider(ABC):
    name: str
    models: list[ModelInfo]

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this provider can run right now."""

    @abstractmethod
    def install_deps(self) -> None:
        """Install heavy dependencies. May take minutes. Called from GUI."""

    @abstractmethod
    async def transcribe_batch(
        self, chunks: list[Path], opts: TranscribeOptions
    ) -> TranscriptResult:
        """Transcribe pre-split audio chunks in order."""

    async def transcribe_live(
        self,
        audio_stream: AsyncGenerator[bytes, None],
        opts: TranscribeOptions,
    ) -> AsyncGenerator[Segment, None]:
        raise NotImplementedError(f"{self.name} does not support live transcription")
        # make this a proper async generator
        return
        yield  # noqa: unreachable — makes Python treat this as async generator
```

- [ ] **Step 4: Install dependencies and run tests**

```bash
pip install -r requirements.txt
python -m pytest tests/test_base.py -v
```
Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/providers/base.py tests/test_base.py
git commit -m "feat: core data types and BaseProvider interface"
```

---

## Chunk 2: Audio Pipeline + Output Formatters

### Task 4: Audio Splitter

**Files:**
- Create: `app/core/audio.py`
- Create: `tests/test_audio.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Write `tests/conftest.py`**

```python
import pytest
from pathlib import Path
from pydub import AudioSegment
from pydub.generators import Sine


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
```

- [ ] **Step 2: Write `tests/test_audio.py`**

```python
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
```

- [ ] **Step 3: Run tests — expect failure**

```bash
python -m pytest tests/test_audio.py -v
```
Expected: `ImportError: cannot import name 'split_audio'`

- [ ] **Step 4: Write `app/core/audio.py`**

```python
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

        if len(chunk) < 100:  # skip tiny tail fragments
            break

        filename = output_dir / f"chunk_{chunk_num:02d}.mp3"
        chunk.export(str(filename), format="mp3", bitrate="192k")
        chunks.append(filename)

        chunk_num += 1
        start_ms += step_ms
        if end_ms == total_ms:
            break

    return chunks
```

- [ ] **Step 5: Run tests — expect pass**

```bash
python -m pytest tests/test_audio.py -v
```
Expected: all 7 tests PASS

- [ ] **Step 6: Commit**

```bash
git add app/core/audio.py tests/test_audio.py tests/conftest.py
git commit -m "feat: audio splitter with overlap"
```

---

### Task 5: Output Formatters

**Files:**
- Create: `app/core/output.py`
- Create: `tests/test_output.py`

- [ ] **Step 1: Write `tests/test_output.py`**

```python
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
```

- [ ] **Step 2: Run tests — expect failure**

```bash
python -m pytest tests/test_output.py -v
```
Expected: `ImportError: cannot import name 'format_transcript'`

- [ ] **Step 3: Write `app/core/output.py`**

```python
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
```

- [ ] **Step 4: Run tests — expect pass**

```bash
python -m pytest tests/test_output.py -v
```
Expected: all 11 tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/core/output.py tests/test_output.py
git commit -m "feat: output formatters (txt, srt, vtt, md)"
```

---

## Chunk 3: Data Layer

### Task 6: Job Queue

**Files:**
- Create: `app/core/queue.py`
- Create: `tests/test_queue.py`

- [ ] **Step 1: Write `tests/test_queue.py`**

```python
import pytest
from pathlib import Path
from app.core.queue import JobQueue
from app.providers.base import Job, TranscribeOptions


def make_job(**kwargs) -> Job:
    defaults = dict(
        input_files=[Path("test.mp3")],
        opts=TranscribeOptions(),
        provider_name="openai",
        model_id="whisper-1",
        status="pending",
    )
    defaults.update(kwargs)
    return Job(**defaults)


def test_job_has_uuid_id():
    job = make_job()
    assert len(job.id) == 36  # UUID4 format


def test_two_jobs_have_different_ids():
    assert make_job().id != make_job().id


def test_queue_add_and_get():
    q = JobQueue()
    job = make_job()
    q.add(job)
    assert q.get(job.id) is job


def test_queue_get_missing_returns_none():
    q = JobQueue()
    assert q.get("nonexistent") is None


def test_queue_list_returns_all():
    q = JobQueue()
    j1, j2 = make_job(), make_job()
    q.add(j1)
    q.add(j2)
    assert len(q.list()) == 2


def test_queue_update_status():
    q = JobQueue()
    job = make_job()
    q.add(job)
    q.update(job.id, status="running", progress=0.5)
    updated = q.get(job.id)
    assert updated.status == "running"
    assert updated.progress == 0.5


def test_queue_cancel():
    q = JobQueue()
    job = make_job()
    q.add(job)
    q.cancel(job.id)
    assert q.get(job.id).status == "cancelled"
```

- [ ] **Step 2: Run — expect failure**

```bash
python -m pytest tests/test_queue.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Write `app/core/queue.py`**

```python
from __future__ import annotations
from app.providers.base import Job
import threading


class JobQueue:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def add(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        return list(self._jobs.values())

    def update(self, job_id: str, **kwargs) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for key, value in kwargs.items():
                setattr(job, key, value)

    def cancel(self, job_id: str) -> None:
        self.update(job_id, status="cancelled")
```

- [ ] **Step 4: Run — expect pass**

```bash
python -m pytest tests/test_queue.py -v
```
Expected: all 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/core/queue.py tests/test_queue.py
git commit -m "feat: in-memory job queue"
```

---

### Task 7: History (SQLite)

**Files:**
- Create: `app/core/history.py`
- Create: `tests/test_history.py`

- [ ] **Step 1: Write `tests/test_history.py`**

```python
import json
import pytest
from pathlib import Path
from app.core.history import History
from app.providers.base import Job, TranscribeOptions


@pytest.fixture
def db(tmp_path):
    return History(tmp_path / "test.db")


def make_done_job(input_files=None, output_files=None):
    return Job(
        input_files=input_files or [Path("lecture.mp3")],
        opts=TranscribeOptions(),
        provider_name="openai",
        model_id="whisper-1",
        status="done",
        output_files=output_files or [Path("/out/lecture.srt")],
    )


def test_save_and_retrieve(db):
    job = make_done_job()
    db.save(job, duration_sec=120.0)
    sessions = db.list()
    assert len(sessions) == 1
    assert sessions[0]["id"] == job.id


def test_list_ordered_by_date_descending(db):
    j1 = make_done_job()
    j2 = make_done_job()
    db.save(j1, duration_sec=60.0)
    db.save(j2, duration_sec=90.0)
    sessions = db.list()
    assert sessions[0]["id"] == j2.id  # most recent first


def test_input_filenames_stored_as_json(db):
    job = make_done_job(input_files=[Path("a.mp3"), Path("b.mp3")])
    db.save(job, duration_sec=0.0)
    session = db.list()[0]
    assert session["input_filenames"] == ["a.mp3", "b.mp3"]


def test_output_paths_stored_as_json(db):
    job = make_done_job(output_files=[Path("/out/a.srt"), Path("/out/a.txt")])
    db.save(job, duration_sec=0.0)
    session = db.list()[0]
    assert session["output_paths"] == ["/out/a.srt", "/out/a.txt"]


def test_delete_removes_entry(db):
    job = make_done_job()
    db.save(job, duration_sec=0.0)
    db.delete(job.id)
    assert db.list() == []
```

- [ ] **Step 2: Run — expect failure**

```bash
python -m pytest tests/test_history.py -v
```

- [ ] **Step 3: Write `app/core/history.py`**

```python
from __future__ import annotations
import json
import sqlite3
from pathlib import Path
from app.providers.base import Job


class History:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    input_filenames TEXT,
                    provider_name TEXT,
                    model_id TEXT,
                    status TEXT,
                    created_at REAL,
                    duration_sec REAL,
                    output_paths TEXT,
                    error_message TEXT
                )
            """)

    def save(self, job: Job, duration_sec: float) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO sessions
                    (id, input_filenames, provider_name, model_id, status,
                     created_at, duration_sec, output_paths, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.id,
                    json.dumps([p.name for p in job.input_files]),
                    job.provider_name,
                    job.model_id,
                    job.status,
                    job.created_at,
                    duration_sec,
                    json.dumps([str(p) for p in job.output_files]),
                    job.error_message,
                ),
            )

    def list(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY created_at DESC"
            ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["input_filenames"] = json.loads(d["input_filenames"])
            d["output_paths"] = json.loads(d["output_paths"])
            result.append(d)
        return result

    def delete(self, session_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
```

- [ ] **Step 4: Run — expect pass**

```bash
python -m pytest tests/test_history.py -v
```
Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/core/history.py tests/test_history.py
git commit -m "feat: SQLite history store"
```

---

### Task 8: Settings

**Files:**
- Create: `app/settings.py`
- Create: `tests/test_settings.py`

- [ ] **Step 1: Write `tests/test_settings.py`**

```python
import json
import pytest
from pathlib import Path
from app.settings import Settings


@pytest.fixture
def settings_file(tmp_path):
    return tmp_path / "settings.json"


def test_defaults_when_no_file(settings_file):
    s = Settings(settings_file)
    assert s.language == "en"
    assert s.output_dir is None
    assert s.chunk_size_sec == 600
    assert s.default_output_formats == ["txt", "srt"]


def test_save_and_reload(settings_file):
    s = Settings(settings_file)
    s.language = "it"
    s.save()
    s2 = Settings(settings_file)
    assert s2.language == "it"


def test_output_dir_stored_as_string(settings_file):
    s = Settings(settings_file)
    s.output_dir = Path("/some/dir")
    s.save()
    raw = json.loads(settings_file.read_text())
    assert raw["output_dir"] == "/some/dir"


def test_output_dir_loaded_as_path(settings_file):
    settings_file.write_text(json.dumps({"output_dir": "/some/dir"}))
    s = Settings(settings_file)
    assert isinstance(s.output_dir, Path)


def test_api_key_not_stored_in_settings(settings_file):
    s = Settings(settings_file)
    s.save()
    raw = json.loads(settings_file.read_text())
    assert "api_key" not in raw
    assert "OPENAI_API_KEY" not in raw
```

- [ ] **Step 2: Run — expect failure**

```bash
python -m pytest tests/test_settings.py -v
```

- [ ] **Step 3: Write `app/settings.py`**

```python
from __future__ import annotations
import json
import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
_SETTINGS_PATH = _PROJECT_ROOT / "settings.json"
_DB_PATH = _PROJECT_ROOT / "app" / "transcriber.db"
_LIVE_OUTPUT_DEFAULT = Path.home() / "Documents" / "Transcriber"


class Settings:
    def __init__(self, path: Path | None = None) -> None:
        self._path = path or _SETTINGS_PATH
        data: dict = {}
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        self.language: str = data.get("language", "en")
        raw_dir = data.get("output_dir")
        self.output_dir: Path | None = Path(raw_dir) if raw_dir else None
        self.chunk_size_sec: int = data.get("chunk_size_sec", 600)
        self.default_provider: str = data.get("default_provider", "openai")
        self.default_model: str = data.get("default_model", "whisper-1")
        self.default_output_formats: list[str] = data.get(
            "default_output_formats", ["txt", "srt"]
        )
        self.wizard_complete: bool = data.get("wizard_complete", False)

    def save(self) -> None:
        data = {
            "language": self.language,
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "chunk_size_sec": self.chunk_size_sec,
            "default_provider": self.default_provider,
            "default_model": self.default_model,
            "default_output_formats": self.default_output_formats,
            "wizard_complete": self.wizard_complete,
        }
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @property
    def openai_api_key(self) -> str | None:
        return os.getenv("OPENAI_API_KEY")

    @property
    def elevenlabs_api_key(self) -> str | None:
        return os.getenv("ELEVENLABS_API_KEY")

    @property
    def db_path(self) -> Path:
        return _DB_PATH

    def resolve_output_dir(self, input_file: Path | None = None) -> Path:
        """Return the effective output directory for a job."""
        if self.output_dir:
            return self.output_dir
        if input_file:
            return input_file.parent
        path = _LIVE_OUTPUT_DEFAULT
        path.mkdir(parents=True, exist_ok=True)
        return path
```

- [ ] **Step 4: Run — expect pass**

```bash
python -m pytest tests/test_settings.py -v
```
Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/settings.py tests/test_settings.py
git commit -m "feat: settings management"
```

---

## Chunk 4: Providers

### Task 9: OpenAI Provider

**Files:**
- Create: `app/providers/openai.py`
- Create: `tests/providers/test_openai.py`

- [ ] **Step 1: Write `tests/providers/test_openai.py`**

```python
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
```

- [ ] **Step 2: Add `model_id` to `TranscribeOptions` in `app/providers/base.py` first**

This must happen before the provider is written so the field exists when the provider uses it.

Edit `app/providers/base.py` — add `model_id` field to `TranscribeOptions`:

```python
@dataclass
class TranscribeOptions:
    language: str | None = None
    prompt: str = ""
    speaker_labels: bool = False
    output_formats: list[str] = field(default_factory=lambda: ["txt"])
    chunk_size_sec: int = 600
    model_id: str = "whisper-1"   # ← add this line
```

- [ ] **Step 3: Update `test_base.py` for new field**

Add to `tests/test_base.py`:
```python
def test_transcribe_options_model_id_default():
    opts = TranscribeOptions()
    assert opts.model_id == "whisper-1"
```

Run to verify:
```bash
python -m pytest tests/test_base.py -v
```
Expected: all tests PASS including the new one.

- [ ] **Step 4: Run provider test — expect failure**

```bash
python -m pytest tests/providers/test_openai.py -v
```
Expected: `ImportError` (provider not yet written)

- [ ] **Step 5: Write `app/providers/openai.py`**

```python
from __future__ import annotations
from pathlib import Path
from app.providers.base import (
    BaseProvider, ModelInfo, HardwareHint, TranscribeOptions,
    TranscriptResult, Segment,
)


class OpenAIProvider(BaseProvider):
    name = "openai"

    models = [
        ModelInfo(
            id="whisper-1",
            name="Whisper 1",
            description="OpenAI Whisper — reliable, widely supported",
            hardware_hint=HardwareHint.CLOUD,
            supports_timestamps=True,
        ),
        ModelInfo(
            id="gpt-4o-transcribe",
            name="GPT-4o Transcribe",
            description="Higher accuracy, higher cost",
            hardware_hint=HardwareHint.CLOUD,
            supports_timestamps=True,
        ),
        ModelInfo(
            id="gpt-4o-mini-transcribe",
            name="GPT-4o Mini Transcribe",
            description="Fast and affordable",
            hardware_hint=HardwareHint.CLOUD,
            supports_timestamps=True,
        ),
    ]

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key

    def is_available(self) -> bool:
        return bool(self._api_key)

    def install_deps(self) -> None:
        pass  # No heavy deps required

    async def transcribe_batch(
        self, chunks: list[Path], opts: TranscribeOptions
    ) -> TranscriptResult:
        from openai import OpenAI
        client = OpenAI(api_key=self._api_key)

        segments: list[Segment] = []
        offset = 0.0

        for chunk_path in chunks:
            params: dict = {
                "model": opts.model_id,
                "response_format": "verbose_json",
                "timestamp_granularities": ["segment"],
            }
            if opts.prompt:
                params["prompt"] = opts.prompt
            if opts.language:
                params["language"] = opts.language

            # Use context manager to avoid file descriptor leaks
            with open(chunk_path, "rb") as audio_file:
                params["file"] = audio_file
                response = client.audio.transcriptions.create(**params)

            if hasattr(response, "segments") and response.segments:
                for seg in response.segments:
                    segments.append(Segment(
                        start=seg.start + offset,
                        end=seg.end + offset,
                        text=seg.text.strip(),
                    ))
                offset += response.duration if hasattr(response, "duration") else 0.0
            else:
                # Fallback: treat entire chunk as one segment
                # Use chunk_size_sec from opts rather than a hardcoded magic number
                text = response.text if hasattr(response, "text") else str(response)
                chunk_duration = float(opts.chunk_size_sec)
                segments.append(Segment(
                    start=offset,
                    end=offset + chunk_duration,
                    text=text.strip(),
                ))
                offset += chunk_duration

        return TranscriptResult(
            segments=segments,
            provider_name=self.name,
            model_id=opts.model_id,
        )
```

- [ ] **Step 6: Run all tests**

```bash
python -m pytest tests/ -v
```
Expected: all tests PASS

- [ ] **Step 7: Commit**

```bash
git add app/providers/openai.py tests/providers/test_openai.py app/providers/base.py tests/test_base.py
git commit -m "feat: OpenAI provider (whisper-1, gpt-4o-transcribe)"
```

---

### Task 10: ElevenLabs Provider

**Files:**
- Create: `app/providers/elevenlabs.py`
- Create: `tests/providers/test_elevenlabs.py`

- [ ] **Step 1: Write `tests/providers/test_elevenlabs.py`**

```python
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
```

- [ ] **Step 2: Run — expect failure**

```bash
python -m pytest tests/providers/test_elevenlabs.py -v
```

- [ ] **Step 3: Write `app/providers/elevenlabs.py`**

```python
from __future__ import annotations
from pathlib import Path
from app.providers.base import (
    BaseProvider, ModelInfo, HardwareHint, TranscribeOptions,
    TranscriptResult, Segment,
)


class ElevenLabsProvider(BaseProvider):
    name = "elevenlabs"

    models = [
        ModelInfo(
            id="scribe_v2",
            name="Scribe v2",
            description="Best accuracy, speaker labels, 90+ languages",
            hardware_hint=HardwareHint.CLOUD,
            supports_speaker_labels=True,
            supports_timestamps=True,
        ),
        ModelInfo(
            id="scribe_v2_realtime",
            name="Scribe v2 Realtime",
            description="<150ms latency live transcription",
            hardware_hint=HardwareHint.CLOUD,
            supports_live=True,
            supports_timestamps=True,
        ),
        ModelInfo(
            id="scribe_v1",
            name="Scribe v1 (legacy)",
            description="Previous generation — use v2 unless on legacy plan",
            hardware_hint=HardwareHint.CLOUD,
            supports_speaker_labels=True,
            supports_timestamps=True,
        ),
    ]

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key

    def is_available(self) -> bool:
        return bool(self._api_key)

    def install_deps(self) -> None:
        pass

    async def transcribe_batch(
        self, chunks: list[Path], opts: TranscribeOptions
    ) -> TranscriptResult:
        from elevenlabs import ElevenLabs

        if not chunks:
            return TranscriptResult(segments=[], provider_name=self.name, model_id=opts.model_id)

        client = ElevenLabs(api_key=self._api_key)
        all_segments: list[Segment] = []
        offset = 0.0
        last_response = None

        for chunk_path in chunks:
            with open(chunk_path, "rb") as f:
                response = client.speech_to_text.convert(
                    file=f,
                    model_id=opts.model_id,
                    language_code=opts.language,
                    diarize=opts.speaker_labels,
                    additional_formats=[],
                    keyterm_prompting_enabled=bool(opts.prompt),
                    keyterms=opts.prompt.split(",") if opts.prompt else [],
                )
            last_response = response

            # Parse utterances (with speaker) or words, or fall back to plain text
            if opts.speaker_labels and response.utterances:
                for utt in response.utterances:
                    all_segments.append(Segment(
                        start=(utt.start or 0.0) + offset,
                        end=(utt.end or 0.0) + offset,
                        text=utt.text.strip(),
                        speaker=f"Speaker {utt.speaker_id}" if utt.speaker_id is not None else None,
                    ))
            elif response.words:
                all_segments.extend(_words_to_segments(response.words, offset))
            else:
                chunk_duration = float(opts.chunk_size_sec)
                all_segments.append(Segment(
                    start=offset,
                    end=offset + chunk_duration,
                    text=(response.text or "").strip(),
                ))

            # Advance offset by last segment end
            if all_segments:
                offset = all_segments[-1].end

        return TranscriptResult(
            segments=all_segments,
            language_detected=getattr(last_response, "language_code", None),
            provider_name=self.name,
            model_id=opts.model_id,
        )


def _words_to_segments(words, offset: float, max_gap_sec: float = 1.5) -> list[Segment]:
    """Group words into segments separated by pauses > max_gap_sec."""
    if not words:
        return []
    segments: list[Segment] = []
    current_words = [words[0]]

    for word in words[1:]:
        gap = (word.start or 0) - (current_words[-1].end or 0)
        if gap > max_gap_sec:
            segments.append(_make_segment(current_words, offset))
            current_words = [word]
        else:
            current_words.append(word)

    segments.append(_make_segment(current_words, offset))
    return segments


def _make_segment(words, offset: float) -> Segment:
    return Segment(
        start=(words[0].start or 0.0) + offset,
        end=(words[-1].end or 0.0) + offset,
        text=" ".join(w.text for w in words).strip(),
    )
```

- [ ] **Step 4: Run all tests**

```bash
python -m pytest tests/ -v
```
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/providers/elevenlabs.py tests/providers/test_elevenlabs.py
git commit -m "feat: ElevenLabs provider (Scribe v1, v2, v2-realtime)"
```

---

## Chunk 5: FastAPI App + i18n

### Task 11: i18n + Locale Files

**Files:**
- Create: `app/core/i18n.py`
- Create: `app/locales/en.json`
- Create: `app/locales/it.json`

- [ ] **Step 1: Write `app/locales/en.json`**

```json
{
  "nav_transcribe": "Transcribe",
  "nav_live": "Live",
  "nav_history": "History",
  "nav_settings": "Settings",
  "transcribe_drop_prompt": "Drop audio files here or",
  "transcribe_drop_browse": "browse",
  "transcribe_drop_formats": "mp3 · m4a · wav · ogg · flac · webm",
  "transcribe_provider": "Provider",
  "transcribe_model": "Model",
  "transcribe_output": "Output formats",
  "transcribe_merge": "Merge all outputs",
  "transcribe_start": "Start Queue",
  "transcribe_stop": "Stop",
  "transcribe_status_waiting": "waiting",
  "transcribe_status_processing": "processing",
  "transcribe_status_done": "done",
  "transcribe_status_error": "error",
  "settings_title": "Settings",
  "settings_api_keys": "API Keys",
  "settings_openai_key": "OpenAI API Key",
  "settings_elevenlabs_key": "ElevenLabs API Key",
  "settings_validate": "Validate",
  "settings_language": "Language",
  "settings_output_dir": "Output directory",
  "settings_same_as_input": "Same as input file",
  "settings_chunk_size": "Chunk size (seconds)",
  "history_title": "History",
  "history_empty": "No transcriptions yet.",
  "history_rerun": "Re-run",
  "history_delete": "Delete",
  "wizard_title": "Welcome to Transcriber",
  "wizard_language": "Choose your language",
  "wizard_setup_type": "How do you want to transcribe?",
  "wizard_openai": "I have an OpenAI API key",
  "wizard_elevenlabs": "I have an ElevenLabs API key",
  "wizard_local": "I want to use local models only",
  "wizard_decide_later": "I'll decide later",
  "wizard_done": "Get started",
  "error_api_key_missing": "API key required",
  "error_no_input_file": "Please select an audio file",
  "error_ffmpeg_missing": "ffmpeg is not installed",
  "hardware_cpu": "CPU",
  "hardware_cpu_recommended": "CPU+",
  "hardware_gpu_optional": "GPU optional",
  "hardware_gpu_recommended": "GPU",
  "hardware_cloud": "Cloud"
}
```

- [ ] **Step 2: Write `app/locales/it.json`**

```json
{
  "nav_transcribe": "Trascrizione",
  "nav_live": "Live",
  "nav_history": "Cronologia",
  "nav_settings": "Impostazioni",
  "transcribe_drop_prompt": "Trascina file audio qui oppure",
  "transcribe_drop_browse": "sfoglia",
  "transcribe_drop_formats": "mp3 · m4a · wav · ogg · flac · webm",
  "transcribe_provider": "Provider",
  "transcribe_model": "Modello",
  "transcribe_output": "Formati di output",
  "transcribe_merge": "Unisci tutti gli output",
  "transcribe_start": "Avvia coda",
  "transcribe_stop": "Interrompi",
  "transcribe_status_waiting": "in attesa",
  "transcribe_status_processing": "elaborazione",
  "transcribe_status_done": "completato",
  "transcribe_status_error": "errore",
  "settings_title": "Impostazioni",
  "settings_api_keys": "Chiavi API",
  "settings_openai_key": "Chiave API OpenAI",
  "settings_elevenlabs_key": "Chiave API ElevenLabs",
  "settings_validate": "Verifica",
  "settings_language": "Lingua",
  "settings_output_dir": "Cartella di output",
  "settings_same_as_input": "Stessa cartella del file di input",
  "settings_chunk_size": "Dimensione chunk (secondi)",
  "history_title": "Cronologia",
  "history_empty": "Nessuna trascrizione ancora.",
  "history_rerun": "Riprocessa",
  "history_delete": "Elimina",
  "wizard_title": "Benvenuto in Transcriber",
  "wizard_language": "Scegli la tua lingua",
  "wizard_setup_type": "Come vuoi trascrivere?",
  "wizard_openai": "Ho una chiave API OpenAI",
  "wizard_elevenlabs": "Ho una chiave API ElevenLabs",
  "wizard_local": "Voglio usare modelli locali",
  "wizard_decide_later": "Lo decido dopo",
  "wizard_done": "Inizia",
  "error_api_key_missing": "Chiave API richiesta",
  "error_no_input_file": "Seleziona un file audio",
  "error_ffmpeg_missing": "ffmpeg non è installato",
  "hardware_cpu": "CPU",
  "hardware_cpu_recommended": "CPU+",
  "hardware_gpu_optional": "GPU opzionale",
  "hardware_gpu_recommended": "GPU",
  "hardware_cloud": "Cloud"
}
```

- [ ] **Step 3: Write `app/core/i18n.py`**

```python
from __future__ import annotations
import json
from pathlib import Path

_LOCALES_DIR = Path(__file__).parent.parent / "locales"
_SUPPORTED = ["en", "it"]
_DEFAULT = "en"
_cache: dict[str, dict] = {}


def load_locale(language: str) -> dict:
    if language not in _SUPPORTED:
        language = _DEFAULT
    if language not in _cache:
        path = _LOCALES_DIR / f"{language}.json"
        _cache[language] = json.loads(path.read_text(encoding="utf-8"))
    return _cache[language]
```

- [ ] **Step 4: Commit**

```bash
git add app/core/i18n.py app/locales/
git commit -m "feat: i18n with English and Italian locales"
```

---

### Task 12: FastAPI App

**Files:**
- Create: `app/main.py`
- Create: `tests/test_main.py`

- [ ] **Step 1: Write `tests/test_main.py`**

```python
import pytest
from fastapi.testclient import TestClient
from app.main import create_app


@pytest.fixture
def client(tmp_path):
    app = create_app(settings_path=tmp_path / "settings.json", db_path=tmp_path / "test.db")
    return TestClient(app)


def test_root_returns_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_locale_endpoint_english(client):
    r = client.get("/api/locale?lang=en")
    assert r.status_code == 200
    data = r.json()
    assert data["nav_transcribe"] == "Transcribe"


def test_locale_endpoint_italian(client):
    r = client.get("/api/locale?lang=it")
    assert r.status_code == 200
    assert r.json()["nav_transcribe"] == "Trascrizione"


def test_providers_endpoint(client):
    r = client.get("/api/providers")
    assert r.status_code == 200
    names = [p["name"] for p in r.json()]
    assert "openai" in names
    assert "elevenlabs" in names


def test_settings_get(client):
    r = client.get("/api/settings")
    assert r.status_code == 200
    data = r.json()
    assert "language" in data
    assert "default_output_formats" in data


def test_settings_update(client):
    r = client.patch("/api/settings", json={"language": "it"})
    assert r.status_code == 200
    r2 = client.get("/api/settings")
    assert r2.json()["language"] == "it"


def test_history_empty_on_fresh_db(client):
    r = client.get("/api/history")
    assert r.status_code == 200
    assert r.json() == []


def test_upload_and_queue_job(client, tmp_path):
    dummy_audio = tmp_path / "test.mp3"
    dummy_audio.write_bytes(b"fake")
    with open(dummy_audio, "rb") as f:
        r = client.post(
            "/api/jobs",
            files={"files": ("test.mp3", f, "audio/mpeg")},
            data={
                "provider_name": "openai",
                "model_id": "whisper-1",
                "output_formats": '["txt"]',
                "merge_output": "false",
            },
        )
    assert r.status_code == 200
    assert "job_id" in r.json()
```

- [ ] **Step 2: Run — expect failure**

```bash
python -m pytest tests/test_main.py -v
```

- [ ] **Step 3: Create `app/templates/index.html` minimal shell**

```html
<!DOCTYPE html>
<html lang="{{ language }}">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Transcriber</title>
  <link rel="stylesheet" href="/static/tailwind.css">
  <script>
    window.APP_PORT = {{ port }};
    window.APP_LANGUAGE = "{{ language }}";
    window.APP_WIZARD_COMPLETE = {{ wizard_complete | lower }};
  </script>
</head>
<body class="bg-gray-950 text-gray-100 min-h-screen" x-data="app()" x-init="init()">
  <div id="app-root">
    <!-- Full UI injected by Alpine.js — see static/app.js -->
    <div class="flex items-center justify-center min-h-screen">
      <p class="text-gray-500">Loading...</p>
    </div>
  </div>
  <script src="/static/alpine.min.js" defer></script>
  <script src="/static/app.js" defer></script>
</body>
</html>
```

- [ ] **Step 4: Write `app/main.py`**

```python
from __future__ import annotations
import asyncio
import json
import shutil
import tempfile
from pathlib import Path
from fastapi import FastAPI, UploadFile, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from app.settings import Settings
from app.core.history import History
from app.core.i18n import load_locale
from app.core.queue import JobQueue
from app.providers.base import Job, TranscribeOptions
from app.providers.openai import OpenAIProvider
from app.providers.elevenlabs import ElevenLabsProvider

_STATIC_DIR = Path(__file__).parent / "static"
_TEMPLATES_DIR = Path(__file__).parent / "templates"
_AUDIO_CHUNKS_DIR = Path(__file__).parent.parent / "audio_chunks"


def create_app(
    settings_path: Path | None = None,
    db_path: Path | None = None,
    port: int = 8000,
) -> FastAPI:
    settings = Settings(settings_path)
    history = History(db_path or settings.db_path)
    queue = JobQueue()
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

    def _get_providers() -> dict[str, object]:
        return {
            "openai": OpenAIProvider(api_key=settings.openai_api_key),
            "elevenlabs": ElevenLabsProvider(api_key=settings.elevenlabs_api_key),
        }

    app = FastAPI(title="Transcriber")
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        return templates.TemplateResponse("index.html", {
            "request": request,
            "port": port,
            "language": settings.language,
            "wizard_complete": settings.wizard_complete,
        })

    @app.get("/api/locale")
    async def get_locale(lang: str = "en"):
        return load_locale(lang)

    @app.get("/api/providers")
    async def get_providers():
        providers = _get_providers()
        result = []
        for name, p in providers.items():
            result.append({
                "name": name,
                "available": p.is_available(),
                "models": [
                    {
                        "id": m.id,
                        "name": m.name,
                        "description": m.description,
                        "hardware_hint": m.hardware_hint.value,
                        "supports_live": m.supports_live,
                        "supports_speaker_labels": m.supports_speaker_labels,
                        "supports_timestamps": m.supports_timestamps,
                    }
                    for m in p.models
                ],
            })
        return result

    @app.get("/api/settings")
    async def get_settings():
        return {
            "language": settings.language,
            "output_dir": str(settings.output_dir) if settings.output_dir else None,
            "chunk_size_sec": settings.chunk_size_sec,
            "default_provider": settings.default_provider,
            "default_model": settings.default_model,
            "default_output_formats": settings.default_output_formats,
            "wizard_complete": settings.wizard_complete,
            "openai_key_set": bool(settings.openai_api_key),
            "elevenlabs_key_set": bool(settings.elevenlabs_api_key),
        }

    @app.patch("/api/settings")
    async def update_settings(body: dict):
        allowed = {
            "language", "output_dir", "chunk_size_sec",
            "default_provider", "default_model",
            "default_output_formats", "wizard_complete",
        }
        for key, value in body.items():
            if key in allowed:
                setattr(settings, key, value)
        settings.save()
        return {"ok": True}

    @app.get("/api/history")
    async def get_history():
        return history.list()

    @app.delete("/api/history/{session_id}")
    async def delete_history(session_id: str):
        history.delete(session_id)
        return {"ok": True}

    @app.post("/api/jobs")
    async def create_job(
        files: list[UploadFile],
        provider_name: str = Form(...),
        model_id: str = Form(...),
        output_formats: str = Form('["txt"]'),
        merge_output: str = Form("false"),
        prompt: str = Form(""),
        language: str = Form(""),
        speaker_labels: str = Form("false"),
    ):
        # Save uploaded files to temp dir
        tmp_dir = Path(tempfile.mkdtemp())
        saved_files: list[Path] = []
        for upload in files:
            dest = tmp_dir / upload.filename
            dest.write_bytes(await upload.read())
            saved_files.append(dest)

        opts = TranscribeOptions(
            language=language or None,
            prompt=prompt,
            speaker_labels=speaker_labels.lower() == "true",
            output_formats=json.loads(output_formats),
            chunk_size_sec=settings.chunk_size_sec,
            model_id=model_id,
        )
        job = Job(
            input_files=saved_files,
            opts=opts,
            provider_name=provider_name,
            model_id=model_id,
            status="pending",
            merge_output=merge_output.lower() == "true",
        )
        queue.add(job)

        # Run job in background
        asyncio.create_task(_run_job(job, settings, _get_providers(), history, _ws_manager))
        return {"job_id": job.id}

    # WebSocket connection manager
    class ConnectionManager:
        def __init__(self):
            self.connections: dict[str, set[WebSocket]] = {}

        async def connect(self, ws: WebSocket, job_id: str):
            await ws.accept()
            self.connections.setdefault(job_id, set()).add(ws)

        def disconnect(self, ws: WebSocket, job_id: str):
            if job_id in self.connections:
                self.connections[job_id].discard(ws)  # set.discard is safe

        async def broadcast(self, job_id: str, msg: dict):
            for ws in list(self.connections.get(job_id, [])):
                try:
                    await ws.send_json(msg)
                except Exception:
                    pass

    _ws_manager = ConnectionManager()

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        await ws.accept()
        job_id = None
        try:
            while True:
                data = await ws.receive_json()
                if data.get("type") == "subscribe":
                    job_id = data["job_id"]
                    await _ws_manager.connect(ws, job_id)
                elif data.get("type") == "cancel" and job_id:
                    queue.cancel(job_id)
        except WebSocketDisconnect:
            if job_id:
                _ws_manager.disconnect(ws, job_id)

    return app


async def _run_job(job: Job, settings: Settings, providers: dict, history: History, ws_manager) -> None:
    """Execute a transcription job in the background."""
    import time
    from app.core.audio import split_audio
    from app.core.output import format_transcript, merge_transcripts

    async def emit(msg: dict):
        await ws_manager.broadcast(job.id, msg)

    job.status = "running"
    await emit({"type": "progress", "job_id": job.id, "progress": 0, "message": "Starting..."})

    try:
        provider = providers.get(job.provider_name)
        if not provider or not provider.is_available():
            raise RuntimeError(f"Provider '{job.provider_name}' is not available.")

        all_results = []
        chunk_dir = Path("audio_chunks") / job.id
        chunk_dir.mkdir(parents=True, exist_ok=True)

        for file_idx, input_file in enumerate(job.input_files):
            await emit({"type": "progress", "job_id": job.id,
                        "progress": file_idx / len(job.input_files),
                        "message": f"Splitting {input_file.name}..."})

            file_chunk_dir = chunk_dir / f"file_{file_idx}"
            chunks = split_audio(input_file, file_chunk_dir, job.opts.chunk_size_sec)

            await emit({"type": "progress", "job_id": job.id,
                        "progress": (file_idx + 0.3) / len(job.input_files),
                        "message": f"Transcribing {input_file.name} ({len(chunks)} chunks)..."})

            result = await provider.transcribe_batch(chunks, job.opts)
            all_results.append((result, input_file.name))

        # Write output files
        output_files: list[Path] = []
        output_dir = settings.resolve_output_dir(job.input_files[0] if job.input_files else None)
        output_dir.mkdir(parents=True, exist_ok=True)

        base_name = job.input_files[0].stem if len(job.input_files) == 1 else "merged"

        if job.merge_output and len(all_results) > 1:
            for fmt in job.opts.output_formats:
                content = merge_transcripts(all_results, fmt)
                out_path = output_dir / f"{base_name}.{fmt}"
                out_path.write_text(content, encoding="utf-8")
                output_files.append(out_path)
        else:
            for result, filename in all_results:
                stem = Path(filename).stem
                for fmt in job.opts.output_formats:
                    content = format_transcript(result, fmt)
                    out_path = output_dir / f"{stem}.{fmt}"
                    out_path.write_text(content, encoding="utf-8")
                    output_files.append(out_path)

        # Cleanup chunks
        shutil.rmtree(str(chunk_dir), ignore_errors=True)

        job.status = "done"
        job.output_files = output_files

        # Compute total audio duration from the last segment end time across all results
        total_duration = 0.0
        for result, _ in all_results:
            if result.segments:
                total_duration += result.segments[-1].end

        history.save(job, duration_sec=total_duration)

        await emit({"type": "done", "job_id": job.id,
                    "output_files": [str(p) for p in output_files]})

    except Exception as e:
        job.status = "error"
        job.error_message = str(e)
        history.save(job, duration_sec=0.0)
        await emit({"type": "error", "job_id": job.id, "message": str(e), "retryable": False})
```

- [ ] **Step 5: Run tests**

```bash
python -m pytest tests/test_main.py -v
```
Expected: all 8 tests PASS

- [ ] **Step 6: Commit**

```bash
git add app/main.py app/templates/index.html tests/test_main.py
git commit -m "feat: FastAPI app with routes, WebSocket, and job runner"
```

---

## Chunk 6: Frontend + Startup

### Task 13: Static Assets

**Files:**
- Create: `app/static/alpine.min.js` (downloaded)
- Create: `app/static/tailwind.css` (downloaded)

- [ ] **Step 1: Download Alpine.js**

```bash
curl -L "https://cdn.jsdelivr.net/npm/alpinejs@3.14.1/dist/cdn.min.js" -o app/static/alpine.min.js
```
Expected: file ~44KB saved to `app/static/alpine.min.js`

- [ ] **Step 2: Download Tailwind CSS (full pre-built CSS, not the CDN play script)**

`cdn.tailwindcss.com` serves a JavaScript runtime, not a CSS file. The correct pre-built CSS is from the npm package dist:

```bash
curl -L "https://cdn.jsdelivr.net/npm/tailwindcss@3.4.1/dist/tailwind.css" -o app/static/tailwind.css
```
Expected: file ~3.7MB saved to `app/static/tailwind.css`

Note: This is the full un-purged build. If the file size seems wrong (e.g. it downloads a tiny JS-like file), verify the URL resolves to actual CSS by checking: `head -c 50 app/static/tailwind.css` — it should start with `/* ! tailwindcss v3`.

- [ ] **Step 3: Commit static assets to repo**

These files must be committed so the repo is self-contained and works offline:

```bash
git add app/static/alpine.min.js app/static/tailwind.css
git commit -m "chore: vendor Alpine.js 3.14.1 and Tailwind CSS 3.4.1"
```

---

### Task 14: Frontend App (Alpine.js)

**Files:**
- Create: `app/static/app.js`

- [ ] **Step 1: Write `app/static/app.js`**

```javascript
// Global locale (injected by Jinja2 as window.APP_LANGUAGE)
let _locale = {};

async function loadLocale() {
  const lang = window.APP_LANGUAGE || 'en';
  const r = await fetch(`/api/locale?lang=${lang}`);
  _locale = await r.json();
}

function t(key) {
  return _locale[key] || key;
}

function app() {
  return {
    // Navigation
    currentSection: 'transcribe',

    // Providers
    providers: [],
    selectedProvider: null,
    selectedModel: null,

    // Transcribe section
    files: [],          // { name, size, status, progress, jobId, outputFiles, error }
    outputFormats: ['txt', 'srt'],
    mergeOutput: false,
    prompt: '',
    speakerLabels: false,
    isRunning: false,

    // Settings
    settings: {},

    // History
    historyItems: [],

    // Wizard
    showWizard: !window.APP_WIZARD_COMPLETE,

    // WebSocket
    _ws: null,

    async init() {
      await loadLocale();
      await this.loadProviders();
      await this.loadSettings();
      await this.loadHistory();
      this._connectWs();
    },

    async loadProviders() {
      const r = await fetch('/api/providers');
      this.providers = await r.json();
      if (this.providers.length) {
        this.selectedProvider = this.providers[0];
        if (this.selectedProvider.models.length)
          this.selectedModel = this.selectedProvider.models[0];
      }
    },

    async loadSettings() {
      const r = await fetch('/api/settings');
      this.settings = await r.json();
      this.outputFormats = this.settings.default_output_formats || ['txt'];
      // Set provider/model from settings
      const sp = this.providers.find(p => p.name === this.settings.default_provider);
      if (sp) {
        this.selectedProvider = sp;
        const sm = sp.models.find(m => m.id === this.settings.default_model);
        if (sm) this.selectedModel = sm;
      }
    },

    async loadHistory() {
      const r = await fetch('/api/history');
      this.historyItems = await r.json();
    },

    async saveSettings(patch) {
      await fetch('/api/settings', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(patch),
      });
      await this.loadSettings();
    },

    handleDrop(event) {
      event.preventDefault();
      const dropped = Array.from(event.dataTransfer.files);
      this._addFiles(dropped);
    },

    handleFileInput(event) {
      this._addFiles(Array.from(event.target.files));
    },

    _addFiles(fileList) {
      const supported = ['mp3','m4a','wav','ogg','flac','webm'];
      for (const f of fileList) {
        const ext = f.name.split('.').pop().toLowerCase();
        if (supported.includes(ext)) {
          this.files.push({ name: f.name, size: f.size, status: 'pending',
                            progress: 0, jobId: null, outputFiles: [], error: null, _file: f });
        }
      }
    },

    removeFile(idx) {
      this.files.splice(idx, 1);
    },

    toggleFormat(fmt) {
      const idx = this.outputFormats.indexOf(fmt);
      if (idx >= 0) this.outputFormats.splice(idx, 1);
      else this.outputFormats.push(fmt);
    },

    async startQueue() {
      if (!this.files.length) return;
      this.isRunning = true;

      const formData = new FormData();
      for (const item of this.files) formData.append('files', item._file);
      formData.append('provider_name', this.selectedProvider?.name || 'openai');
      formData.append('model_id', this.selectedModel?.id || 'whisper-1');
      formData.append('output_formats', JSON.stringify(this.outputFormats));
      formData.append('merge_output', String(this.mergeOutput));
      formData.append('prompt', this.prompt);
      formData.append('speaker_labels', String(this.speakerLabels));

      const r = await fetch('/api/jobs', { method: 'POST', body: formData });
      const { job_id } = await r.json();

      // Mark all files as running (single job for whole queue)
      for (const f of this.files) { f.status = 'running'; f.jobId = job_id; }

      this._subscribeJob(job_id);
    },

    _subscribeJob(jobId) {
      if (!this._ws || this._ws.readyState !== WebSocket.OPEN) return;
      this._ws.send(JSON.stringify({ type: 'subscribe', job_id: jobId }));
    },

    _connectWs() {
      const port = window.APP_PORT || 8000;
      this._ws = new WebSocket(`ws://localhost:${port}/ws`);
      this._ws.onmessage = (e) => {
        const msg = JSON.parse(e.data);
        this._handleWsMessage(msg);
      };
      this._ws.onclose = () => {
        setTimeout(() => this._connectWs(), 2000);
      };
    },

    _handleWsMessage(msg) {
      if (msg.type === 'progress') {
        const item = this.files.find(f => f.jobId === msg.job_id);
        if (item) { item.progress = msg.progress * 100; item.status = 'running'; }
      } else if (msg.type === 'done') {
        for (const f of this.files) {
          if (f.jobId === msg.job_id) {
            f.status = 'done'; f.progress = 100;
            f.outputFiles = msg.output_files || [];
          }
        }
        this.isRunning = false;
        this.loadHistory();
      } else if (msg.type === 'error') {
        const item = this.files.find(f => f.jobId === msg.job_id);
        if (item) { item.status = 'error'; item.error = msg.message; }
        this.isRunning = false;
      }
    },

    stopQueue() {
      const running = this.files.find(f => f.status === 'running');
      if (running?.jobId && this._ws) {
        this._ws.send(JSON.stringify({ type: 'cancel', job_id: running.jobId }));
      }
      this.isRunning = false;
    },

    async deleteHistory(id) {
      await fetch(`/api/history/${id}`, { method: 'DELETE' });
      this.historyItems = this.historyItems.filter(h => h.id !== id);
    },

    async completeWizard(setupType) {
      await this.saveSettings({ wizard_complete: true });
      this.showWizard = false;
    },

    hardwareBadgeClass(hint) {
      const map = {
        cpu: 'bg-green-900 text-green-300',
        cpu_recommended: 'bg-green-900 text-green-300',
        gpu_optional: 'bg-yellow-900 text-yellow-300',
        gpu_recommended: 'bg-orange-900 text-orange-300',
        cloud: 'bg-blue-900 text-blue-300',
      };
      return map[hint] || 'bg-gray-800 text-gray-400';
    },

    formatBytes(bytes) {
      if (bytes < 1024) return bytes + ' B';
      if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
      return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    },

    t,
  };
}
```

- [ ] **Step 2: Update `app/templates/index.html` with full UI**

Replace the file with the full UI shell. This is long — write it completely:

```html
<!DOCTYPE html>
<html lang="{{ language }}">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Transcriber</title>
  <link rel="stylesheet" href="/static/tailwind.css">
  <script>
    window.APP_PORT = {{ port }};
    window.APP_LANGUAGE = "{{ language }}";
    window.APP_WIZARD_COMPLETE = {{ wizard_complete | lower }};
  </script>
  <style>
    [x-cloak] { display: none !important; }
    .sidebar-icon { @apply w-10 h-9 rounded-lg flex items-center justify-center text-lg transition-colors; }
    .sidebar-icon.active { @apply bg-gray-800; }
    .sidebar-icon:not(.active) { @apply text-gray-500 hover:text-gray-300 hover:bg-gray-900 cursor-pointer; }
  </style>
</head>
<body class="bg-gray-950 text-gray-100 min-h-screen font-sans" x-data="app()" x-init="init()" x-cloak>

  <!-- First-run Wizard Modal -->
  <div x-show="showWizard" class="fixed inset-0 bg-black/70 z-50 flex items-center justify-center p-4">
    <div class="bg-gray-900 border border-gray-700 rounded-xl p-8 max-w-md w-full shadow-2xl">
      <h2 class="text-xl font-bold mb-2" x-text="t('wizard_title')"></h2>
      <p class="text-gray-400 text-sm mb-6" x-text="t('wizard_setup_type')"></p>
      <div class="space-y-3">
        <button @click="completeWizard('openai')"
                class="w-full text-left px-4 py-3 bg-gray-800 hover:bg-gray-700 rounded-lg text-sm transition-colors">
          🟢 <span x-text="t('wizard_openai')"></span>
        </button>
        <button @click="completeWizard('elevenlabs')"
                class="w-full text-left px-4 py-3 bg-gray-800 hover:bg-gray-700 rounded-lg text-sm transition-colors">
          💜 <span x-text="t('wizard_elevenlabs')"></span>
        </button>
        <button @click="completeWizard('local')"
                class="w-full text-left px-4 py-3 bg-gray-800 hover:bg-gray-700 rounded-lg text-sm transition-colors">
          ⚡ <span x-text="t('wizard_local')"></span>
        </button>
        <button @click="completeWizard('later')"
                class="w-full text-left px-4 py-3 text-gray-500 hover:text-gray-300 text-sm transition-colors">
          <span x-text="t('wizard_decide_later')"></span>
        </button>
      </div>
    </div>
  </div>

  <!-- Main Layout -->
  <div class="flex h-screen overflow-hidden">

    <!-- Sidebar -->
    <div class="w-14 bg-gray-1000 border-r border-gray-800 flex flex-col items-center py-4 gap-1 flex-shrink-0" style="background:#010409">
      <!-- Logo -->
      <div class="w-9 h-9 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-lg flex items-center justify-center text-lg mb-4">🎙</div>

      <button @click="currentSection='transcribe'"
              :class="currentSection==='transcribe' ? 'active' : ''"
              class="sidebar-icon" title="Transcribe">📂</button>
      <button @click="currentSection='history'"
              :class="currentSection==='history' ? 'active' : ''"
              class="sidebar-icon" title="History">📋</button>

      <div class="flex-1"></div>

      <button @click="currentSection='settings'"
              :class="currentSection==='settings' ? 'active' : ''"
              class="sidebar-icon" title="Settings">⚙️</button>
      <button @click="saveSettings({language: settings.language==='en'?'it':'en'})"
              class="w-8 h-8 rounded-full bg-gray-800 flex items-center justify-center text-xs text-gray-400 hover:text-white hover:bg-gray-700 transition-colors cursor-pointer"
              x-text="settings.language?.toUpperCase() || 'EN'"></button>
    </div>

    <!-- Main Panel -->
    <div class="flex-1 flex flex-col overflow-hidden">

      <!-- TRANSCRIBE SECTION -->
      <div x-show="currentSection==='transcribe'" class="flex flex-col h-full">
        <!-- Header -->
        <div class="px-4 py-3 border-b border-gray-800 flex items-center justify-between flex-shrink-0">
          <div>
            <div class="font-semibold text-sm" x-text="t('nav_transcribe')"></div>
            <div class="text-xs text-gray-500" x-text="`${files.length} file(s) in queue`"></div>
          </div>
          <div class="flex gap-2">
            <!-- Provider selector -->
            <select x-model="selectedProvider"
                    @change="selectedModel = selectedProvider?.models[0] || null"
                    class="bg-gray-800 border border-gray-700 text-xs rounded-md px-2 py-1 text-gray-300">
              <template x-for="p in providers" :key="p.name">
                <option :value="p" :disabled="!p.available" x-text="p.name"></option>
              </template>
            </select>
            <!-- Model selector -->
            <select x-model="selectedModel"
                    class="bg-gray-800 border border-gray-700 text-xs rounded-md px-2 py-1 text-gray-300">
              <template x-for="m in (selectedProvider?.models || [])" :key="m.id">
                <option :value="m" x-text="m.name"></option>
              </template>
            </select>
          </div>
        </div>

        <!-- Drop zone -->
        <div class="mx-4 mt-3 border-2 border-dashed border-gray-700 rounded-xl p-6 text-center flex-shrink-0
                    hover:border-indigo-500 transition-colors cursor-pointer"
             @dragover.prevent
             @drop="handleDrop($event)"
             @click="$refs.fileInput.click()">
          <div class="text-2xl mb-1">📁</div>
          <div class="text-sm text-gray-400">
            <span x-text="t('transcribe_drop_prompt')"></span>
            <span class="text-indigo-400 underline cursor-pointer" x-text="t('transcribe_drop_browse')"></span>
          </div>
          <div class="text-xs text-gray-600 mt-1" x-text="t('transcribe_drop_formats')"></div>
          <input type="file" class="hidden" x-ref="fileInput"
                 accept=".mp3,.m4a,.wav,.ogg,.flac,.webm" multiple
                 @change="handleFileInput($event)">
        </div>

        <!-- Output format toggles -->
        <div class="mx-4 mt-2 flex gap-2 flex-wrap flex-shrink-0">
          <template x-for="fmt in ['txt','srt','vtt','md']" :key="fmt">
            <button @click="toggleFormat(fmt)"
                    :class="outputFormats.includes(fmt)
                      ? 'bg-indigo-600 text-white'
                      : 'bg-gray-800 text-gray-400 hover:text-gray-200'"
                    class="px-3 py-1 rounded-md text-xs font-mono transition-colors"
                    x-text="fmt.toUpperCase()"></button>
          </template>
          <!-- Speaker labels (if supported) -->
          <button x-show="selectedModel?.supports_speaker_labels"
                  @click="speakerLabels = !speakerLabels"
                  :class="speakerLabels ? 'bg-purple-700 text-white' : 'bg-gray-800 text-gray-400'"
                  class="px-3 py-1 rounded-md text-xs transition-colors">
            👥 Speakers
          </button>
        </div>

        <!-- Prompt input (ElevenLabs Scribe v2 only) -->
        <div x-show="selectedModel?.id === 'scribe_v2'" class="mx-4 mt-2 flex-shrink-0">
          <input x-model="prompt" type="text"
                 placeholder="Keyterms (comma-separated, e.g. ATP, mitosis)"
                 class="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-1.5 text-xs text-gray-300 placeholder-gray-600">
        </div>

        <!-- Queue list -->
        <div class="flex-1 overflow-y-auto px-4 mt-3 space-y-2">
          <template x-for="(item, idx) in files" :key="item.name + idx">
            <div class="bg-gray-900 border rounded-lg px-3 py-2"
                 :class="item.status==='error' ? 'border-red-800' : 'border-gray-800'">
              <div class="flex items-center justify-between mb-1">
                <div class="text-sm font-medium truncate max-w-xs" x-text="item.name"></div>
                <div class="flex items-center gap-2">
                  <span class="text-xs"
                        :class="{
                          'text-indigo-400': item.status==='running',
                          'text-green-400': item.status==='done',
                          'text-red-400': item.status==='error',
                          'text-gray-500': item.status==='pending'
                        }"
                        x-text="item.status==='running' ? Math.round(item.progress)+'%' : item.status"></span>
                  <button x-show="item.status==='pending'" @click="removeFile(idx)"
                          class="text-gray-600 hover:text-red-400 text-xs">✕</button>
                </div>
              </div>
              <div x-show="item.status==='running'" class="bg-gray-800 rounded h-1">
                <div class="bg-indigo-500 h-1 rounded transition-all"
                     :style="`width: ${item.progress}%`"></div>
              </div>
              <div x-show="item.status==='error'" class="text-xs text-red-400 mt-1" x-text="item.error"></div>
              <div x-show="item.status==='done'" class="mt-1 flex gap-2 flex-wrap">
                <template x-for="path in item.outputFiles" :key="path">
                  <a :href="`/api/download?path=${encodeURIComponent(path)}`"
                     class="text-xs text-indigo-400 hover:underline"
                     x-text="path.split(/[\\/]/).pop()"></a>
                </template>
              </div>
            </div>
          </template>
        </div>

        <!-- Bottom action bar -->
        <div class="px-4 py-3 border-t border-gray-800 flex items-center gap-3 flex-shrink-0">
          <label class="flex items-center gap-2 text-xs text-gray-400 cursor-pointer flex-1">
            <div @click="mergeOutput = !mergeOutput"
                 :class="mergeOutput ? 'bg-indigo-600' : 'bg-gray-700'"
                 class="w-8 h-4 rounded-full relative transition-colors cursor-pointer">
              <div :class="mergeOutput ? 'translate-x-4' : 'translate-x-0.5'"
                   class="absolute top-0.5 w-3 h-3 bg-white rounded-full transition-transform"></div>
            </div>
            <span x-text="t('transcribe_merge')"></span>
          </label>
          <button x-show="isRunning" @click="stopQueue()"
                  class="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded-lg text-xs transition-colors">
            ⏹ <span x-text="t('transcribe_stop')"></span>
          </button>
          <button @click="startQueue()" :disabled="!files.length || isRunning"
                  class="px-4 py-1.5 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40
                         disabled:cursor-not-allowed rounded-lg text-xs font-semibold transition-colors">
            ▶ <span x-text="t('transcribe_start')"></span>
          </button>
        </div>
      </div>

      <!-- HISTORY SECTION -->
      <div x-show="currentSection==='history'" class="flex flex-col h-full">
        <div class="px-4 py-3 border-b border-gray-800 flex-shrink-0">
          <div class="font-semibold text-sm" x-text="t('history_title')"></div>
        </div>
        <div class="flex-1 overflow-y-auto px-4 py-3 space-y-2">
          <p x-show="!historyItems.length" class="text-gray-600 text-sm text-center py-8"
             x-text="t('history_empty')"></p>
          <template x-for="item in historyItems" :key="item.id">
            <div class="bg-gray-900 border border-gray-800 rounded-lg px-4 py-3">
              <div class="flex items-start justify-between gap-2">
                <div class="min-w-0">
                  <div class="text-sm font-medium truncate"
                       x-text="item.input_filenames.join(', ')"></div>
                  <div class="text-xs text-gray-500 mt-0.5"
                       x-text="`${item.provider_name} · ${item.model_id} · ${new Date(item.created_at*1000).toLocaleString()}`"></div>
                  <div class="mt-1.5 flex gap-2 flex-wrap">
                    <template x-for="path in item.output_paths" :key="path">
                      <a :href="`/api/download?path=${encodeURIComponent(path)}`"
                         class="text-xs text-indigo-400 hover:underline"
                         x-text="path.split(/[\\/]/).pop()"></a>
                    </template>
                  </div>
                </div>
                <button @click="deleteHistory(item.id)"
                        class="text-gray-600 hover:text-red-400 text-sm flex-shrink-0">🗑</button>
              </div>
            </div>
          </template>
        </div>
      </div>

      <!-- SETTINGS SECTION -->
      <div x-show="currentSection==='settings'" class="flex flex-col h-full">
        <div class="px-4 py-3 border-b border-gray-800 flex-shrink-0">
          <div class="font-semibold text-sm" x-text="t('settings_title')"></div>
        </div>
        <div class="flex-1 overflow-y-auto px-4 py-4 space-y-6">

          <!-- API Keys -->
          <div>
            <div class="text-xs uppercase text-gray-500 font-semibold mb-3"
                 x-text="t('settings_api_keys')"></div>
            <div class="space-y-3">
              <div>
                <label class="text-xs text-gray-400" x-text="t('settings_openai_key')"></label>
                <div class="flex gap-2 mt-1">
                  <div class="flex-1 bg-gray-800 border border-gray-700 rounded-md px-3 py-1.5 text-xs text-gray-400"
                       x-text="settings.openai_key_set ? 'sk-...set' : 'Not configured'"></div>
                </div>
              </div>
              <div>
                <label class="text-xs text-gray-400" x-text="t('settings_elevenlabs_key')"></label>
                <div class="flex gap-2 mt-1">
                  <div class="flex-1 bg-gray-800 border border-gray-700 rounded-md px-3 py-1.5 text-xs text-gray-400"
                       x-text="settings.elevenlabs_key_set ? '...set' : 'Not configured'"></div>
                </div>
              </div>
              <p class="text-xs text-gray-600">Keys are stored in <code>.env</code> in the project folder.</p>
            </div>
          </div>

          <!-- Language -->
          <div>
            <label class="text-xs text-gray-400" x-text="t('settings_language')"></label>
            <select @change="saveSettings({language: $event.target.value})"
                    class="mt-1 block bg-gray-800 border border-gray-700 rounded-md px-3 py-1.5 text-sm text-gray-300">
              <option value="en" :selected="settings.language==='en'">English</option>
              <option value="it" :selected="settings.language==='it'">Italiano</option>
            </select>
          </div>

          <!-- Chunk size -->
          <div>
            <label class="text-xs text-gray-400" x-text="t('settings_chunk_size')"></label>
            <input type="number" :value="settings.chunk_size_sec" min="30" max="1800"
                   @change="saveSettings({chunk_size_sec: parseInt($event.target.value)})"
                   class="mt-1 block w-32 bg-gray-800 border border-gray-700 rounded-md px-3 py-1.5 text-sm text-gray-300">
          </div>

          <!-- Re-run wizard -->
          <div class="pt-2 border-t border-gray-800">
            <button @click="showWizard = true"
                    class="text-xs text-indigo-400 hover:underline">
              Run setup wizard again
            </button>
          </div>

        </div>
      </div>

    </div><!-- end main panel -->
  </div><!-- end layout -->

  <script src="/static/alpine.min.js" defer></script>
  <script src="/static/app.js" defer></script>
</body>
</html>
```

- [ ] **Step 3: Add download endpoint to `app/main.py`**

Add inside `create_app`, after history endpoints:

```python
@app.get("/api/download")
async def download_file(path: str):
    file_path = Path(path)
    if not file_path.exists():
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(file_path), filename=file_path.name)
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/ -v
```
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/static/app.js app/templates/index.html app/main.py
git commit -m "feat: Alpine.js frontend with transcribe, history, and settings UI"
```

---

### Task 15: start.py + Launchers

**Files:**
- Create/Replace: `start.py`
- Create: `start.bat`
- Create: `start.sh`

- [ ] **Step 1: Write `start.py`**

```python
#!/usr/bin/env python3
"""
Transcriber bootstrapper.
Run this script to install dependencies and open the app in your browser.
"""
import subprocess
import sys
import socket
import webbrowser
import time
from pathlib import Path

MIN_PYTHON = (3, 10)
THIS_DIR = Path(__file__).parent.resolve()
STATIC_DIR = THIS_DIR / "app" / "static"

REQUIRED_ASSETS = [
    STATIC_DIR / "alpine.min.js",
    STATIC_DIR / "tailwind.css",
]


def check_python():
    if sys.version_info < MIN_PYTHON:
        print(f"[error] Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required. "
              f"You have {sys.version_info.major}.{sys.version_info.minor}.")
        print("  Download: https://www.python.org/downloads/")
        sys.exit(1)
    print(f"[ok] Python {sys.version_info.major}.{sys.version_info.minor}")


def install_requirements():
    req_file = THIS_DIR / "requirements.txt"
    print("[setup] Installing/verifying Python packages...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(req_file), "--quiet"],
        capture_output=False,
    )
    if result.returncode != 0:
        print("[error] Package installation failed. Check the output above.")
        sys.exit(1)
    print("[ok] Packages installed")


def check_static_assets():
    missing = [a for a in REQUIRED_ASSETS if not a.exists()]
    if missing:
        print(f"[error] Missing static assets: {[str(a.name) for a in missing]}")
        print("  Run: git checkout app/static/ OR re-clone the repository.")
        sys.exit(1)
    print("[ok] Static assets present")


def check_ffmpeg():
    import shutil
    if shutil.which("ffmpeg"):
        print("[ok] ffmpeg found")
    else:
        print("[warning] ffmpeg not found — audio conversion will not work.")
        print("  macOS:   brew install ffmpeg")
        print("  Ubuntu:  sudo apt-get install ffmpeg")
        print("  Windows: choco install ffmpeg   (or download from ffmpeg.org)")


def find_free_port(start: int = 8000, end: int = 8010) -> int:
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("localhost", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No free port found in range 8000-8010")


def load_dotenv():
    env_file = THIS_DIR / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv as _load
            _load(env_file)
            print("[ok] .env loaded")
        except ImportError:
            pass


def main():
    check_python()
    install_requirements()
    check_static_assets()
    check_ffmpeg()
    load_dotenv()

    port = find_free_port()
    url = f"http://localhost:{port}"
    print(f"\n[run] Starting Transcriber on {url}")
    print("      Press Ctrl+C to stop.\n")

    # Small delay then open browser
    def open_browser():
        time.sleep(1.2)
        webbrowser.open(url)

    import threading
    threading.Thread(target=open_browser, daemon=True).start()

    # Import here so packages are already installed above
    import uvicorn
    from app.main import create_app
    app = create_app(port=port)
    uvicorn.run(app, host="localhost", port=port, log_level="warning")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `start.bat` (Windows)**

```batch
@echo off
echo Starting Transcriber...
python start.py
if errorlevel 1 (
    echo.
    echo [error] Something went wrong. See message above.
    echo If Python is not installed: https://www.python.org/downloads/
    pause
)
```

- [ ] **Step 3: Write `start.sh` (macOS/Linux)**

```bash
#!/bin/bash
echo "Starting Transcriber..."
python3 start.py
if [ $? -ne 0 ]; then
    echo ""
    echo "[error] Something went wrong. See message above."
    echo "If Python is not installed: https://www.python.org/downloads/"
    read -p "Press Enter to close..."
fi
```

```bash
chmod +x start.sh
```

- [ ] **Step 4: Run full test suite one last time**

```bash
python -m pytest tests/ -v --tb=short
```
Expected: all tests PASS, no failures

- [ ] **Step 5: Final commit**

```bash
git add start.py start.bat start.sh
git commit -m "feat: start.py bootstrapper and double-click launchers"
```

---

## End-to-End Manual Smoke Test

Before declaring Plan 1 complete, verify the app works end-to-end:

- [ ] Run `python start.py` — should install deps, open browser at localhost
- [ ] Browser shows the main UI with sidebar (📂, 📋, ⚙️)
- [ ] First-run wizard appears if `settings.json` doesn't exist
- [ ] Drop a short audio file onto the drop zone — it appears in queue
- [ ] Select provider "openai", set API key in `.env`, click "Start Queue"
- [ ] Progress bar moves, status updates in real-time via WebSocket
- [ ] On completion, output file links appear — click to download
- [ ] History section shows the completed session
- [ ] Language toggle (EN/IT) switches UI strings
- [ ] `python -m pytest tests/ -v` passes with no failures

```bash
git commit --allow-empty -m "chore: Plan 1 (MVP) complete — batch transcription with OpenAI + ElevenLabs"
git tag v0.1.0-mvp
```

---

*Plan 2: Local Providers (faster-whisper, Qwen3-ASR, Ollama) — `docs/superpowers/plans/2026-03-17-transcriber-local-providers.md`*
*Plan 3: Live Transcription Mode — `docs/superpowers/plans/2026-03-17-transcriber-live-mode.md`*
