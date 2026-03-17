# Transcriber — Full Redesign Spec
**Date:** 2026-03-17
**Status:** Draft v2 (post-review)
**Author:** brainstorming session

---

## Overview

Complete redesign of the current Python/CustomTkinter transcription tool into a local web application targeting university students and professionals. The app runs entirely on the user's machine, requires no cloud infrastructure beyond optional API providers, and is distributed as open source.

The core transcription logic (audio splitting via pydub, chunked processing) is preserved and refactored. The UI, architecture, and provider system are rebuilt from scratch.

---

## Goals

- **Disarmingly easy setup** — double-click or single command, guided on-boarding, no terminal knowledge required
- **Beautiful, intuitive GUI** — local web app in the browser, modern design
- **Multi-provider** — OpenAI, ElevenLabs, and local models (faster-whisper, Qwen3-ASR, Ollama)
- **Offline-first** — all local providers work with no internet; API providers clearly marked as cloud
- **Open source** — MIT license, structured for community contributions
- **i18n** — English and Italian on day one, extensible to other languages

---

## Tech Stack

### Backend
- **Python 3.10+** with **FastAPI**
- **WebSockets** for real-time progress updates and live transcription streaming
- **pydub** + **ffmpeg** for audio splitting, format conversion, and webm→wav decoding
- **python-dotenv** for API key management

### Frontend
- **Alpine.js** (vendored, ~44KB minified) — reactive UI without a build step
- **Tailwind CSS** (vendored, full CDN build ~3.7MB, not purged) — committed to `app/static/`
- Both files committed directly to the repo — zero CDN dependency, works fully offline
- Dev note: to generate a purged/smaller Tailwind build, run `npx tailwindcss` against templates (optional, requires Node.js, documented in `CONTRIBUTING.md`)
- **Jinja2 templates** served by FastAPI

### Distribution
- `start.py` bootstrapper handles everything for end users
- Static frontend assets (Alpine.js, Tailwind CSS) are **committed to the repo** — `start.py` does not need to download them
- Heavy ML dependencies (torch, transformers, ctranslate2) are installed on-demand via GUI when the user first selects a local model that requires them

---

## Project Structure

```
transcriber/
├── start.py                  # Bootstrapper: installs deps, opens browser
├── requirements.txt          # Core deps (fastapi, uvicorn, pydub, openai, elevenlabs, python-dotenv, jinja2, websockets)
├── requirements-local.txt    # Heavy deps (faster-whisper, torch, transformers, ctranslate2) — installed on demand
├── .env                      # API keys — gitignored
├── .env.example              # Template for users
├── settings.json             # User preferences (language, output dir, etc.) — gitignored
├── app/
│   ├── main.py               # FastAPI app, routes, WebSocket endpoints
│   ├── settings.py           # Settings management: reads/writes settings.json and .env
│   ├── core/
│   │   ├── audio.py          # Audio splitting with overlap (refactored from split_audio.py)
│   │   ├── output.py         # Output formatters: txt, srt, vtt, md
│   │   ├── queue.py          # Job queue with UUID4 job IDs and status tracking
│   │   ├── history.py        # SQLite-backed session history (app/transcriber.db)
│   │   └── i18n.py           # Internationalization loader
│   ├── providers/
│   │   ├── base.py           # Abstract BaseProvider (abc.ABC + @abstractmethod)
│   │   ├── openai.py         # OpenAI (whisper-1, gpt-4o-transcribe, gpt-4o-mini-transcribe)
│   │   ├── elevenlabs.py     # ElevenLabs (scribe_v2, scribe_v2_realtime; scribe_v1 kept for legacy)
│   │   ├── faster_whisper.py # faster-whisper (tiny → large-v3)
│   │   ├── qwen3_asr.py      # Qwen3-ASR (0.6B, 1.7B) via HuggingFace transformers
│   │   └── ollama.py         # Ollama daemon integration
│   ├── static/
│   │   ├── alpine.min.js     # Vendored Alpine.js (committed)
│   │   └── tailwind.css      # Vendored Tailwind CSS full build (committed)
│   ├── templates/
│   │   └── index.html        # SPA shell — Jinja2 injects port + locale at serve time
│   ├── locales/
│   │   ├── en.json           # English strings
│   │   └── it.json           # Italian strings
│   └── transcriber.db        # SQLite history database — gitignored
├── docs/
│   └── superpowers/specs/
│       └── 2026-03-17-transcriber-redesign.md
└── audio_chunks/             # Temp directory — gitignored
```

---

## Core Data Types

Defined in `app/providers/base.py` and shared across the codebase:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import AsyncGenerator
import uuid

class HardwareHint(str, Enum):
    CPU = "cpu"                       # runs on any machine
    CPU_RECOMMENDED = "cpu_recommended"  # runs on CPU, better with decent hardware
    GPU_OPTIONAL = "gpu_optional"     # noticeably faster with GPU
    GPU_RECOMMENDED = "gpu_recommended"  # CPU possible but slow
    CLOUD = "cloud"                   # requires internet + API key

@dataclass
class ModelInfo:
    id: str                           # e.g. "large-v3", "scribe_v2"
    name: str                         # display name
    description: str                  # short human-readable description
    hardware_hint: HardwareHint
    supports_live: bool = False
    supports_speaker_labels: bool = False
    supports_timestamps: bool = True

@dataclass
class TranscribeOptions:
    language: str | None = None       # ISO 639-1 code, None = auto-detect
    prompt: str = ""                  # keyterm/context prompt (provider-dependent)
    speaker_labels: bool = False      # request diarization if supported
    output_formats: list[str] = field(default_factory=lambda: ["txt"])  # txt, srt, vtt, md
    chunk_size_sec: int = 600         # audio chunk length for splitting

@dataclass
class Segment:
    start: float                      # seconds from audio start
    end: float
    text: str
    speaker: str | None = None        # "Speaker 1", "Speaker 2", etc. — None if not available

@dataclass
class TranscriptResult:
    segments: list[Segment]
    language_detected: str | None = None
    provider_name: str = ""
    model_id: str = ""
```

---

## Provider System

### Abstract Interface (`app/providers/base.py`)

```python
class BaseProvider(ABC):
    name: str
    models: list[ModelInfo]

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider can run: deps installed, API key present, daemon running."""

    @abstractmethod
    def install_deps(self) -> None:
        """Install heavy dependencies. Called from the GUI install modal. May take minutes."""

    @abstractmethod
    async def transcribe_batch(
        self, chunks: list[Path], opts: TranscribeOptions
    ) -> TranscriptResult:
        """Transcribe pre-split audio chunks in order. Returns merged result."""

    async def transcribe_live(
        self, audio_stream: AsyncGenerator[bytes, None], opts: TranscribeOptions
    ) -> AsyncGenerator[Segment, None]:
        """Stream transcription from live audio. Default: raise NotImplementedError."""
        raise NotImplementedError(f"{self.name} does not support live transcription")
```

### Provider Matrix

| Provider | Models | Live | Speakers | Timestamps | Requires |
|---|---|---|---|---|---|
| OpenAI | whisper-1, gpt-4o-transcribe, gpt-4o-mini-transcribe | ✗ | ✗ | ✓ segment | API key |
| ElevenLabs | scribe_v2 (default), scribe_v2_realtime, scribe_v1 (legacy) | scribe_v2_realtime only | ✓ | ✓ word+segment | API key |
| faster-whisper | tiny, base, small, medium, large-v3 | ✓ (buffered) | ✗ | ✓ segment | ctranslate2 (auto-install) |
| Qwen3-ASR | 0.6B, 1.7B | ✓ (buffered) | ✗ | ✓ segment† | torch + transformers (on-demand) |
| Ollama | enumerated via `/api/tags` at runtime | ✓ | ✗ | varies | Ollama daemon running |

†Qwen3-ASR segment timestamps: the 1.7B model produces segment-level timestamps; the 0.6B model does not — timestamps for 0.6B fall back to chunk-boundary estimates. Both `ModelInfo` instances set `supports_timestamps` accordingly. Subject to change with model updates.

### ElevenLabs model note
`scribe_v2` is the recommended default. `scribe_v1` is included for users on legacy plans but not shown prominently in the UI. `scribe_v2_realtime` is shown only in the Live Mode provider selector.

### Ollama integration note
Model list is fetched at runtime from `http://localhost:11434/api/tags`. If the Ollama daemon is not running, the provider is shown as unavailable with a "Start Ollama" instruction. Audio is sent as base64-encoded wav to Ollama's multimodal inference endpoint. Only models with audio capability are surfaced. Timestamp support depends on the specific model's output — the provider attempts to parse timestamps from the response and falls back to chunk-boundary estimates.

---

## Audio Splitting (`app/core/audio.py`)

Refactored from `split_audio.py` with the following changes:

- **Chunk overlap:** Each chunk includes a 1-second overlap with the next chunk to avoid cutting words at boundaries. The overlapping portion is trimmed from the final concatenated transcript (detected via simple text deduplication of the last sentence).
- **Chunk size:** Configurable via `TranscribeOptions.chunk_size_sec`, default 600s.
- **Output format:** Chunks exported as MP3 at 192kbps (same as current implementation).
- **Temp directory:** `audio_chunks/` relative to the project root, cleaned up after each job completes successfully. On error, chunks are retained for debugging.

---

## Job Queue & History

### Job Queue (`app/core/queue.py`)

Each transcription job has:

```python
@dataclass
class Job:
    # Fields without defaults first (Python dataclass requirement)
    input_files: list[Path]           # one or more audio files
    opts: TranscribeOptions
    provider_name: str
    model_id: str
    status: Literal["pending", "running", "done", "error", "cancelled"]
    # Fields with defaults
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    progress: float = 0.0             # 0.0 to 1.0
    error_message: str | None = None
    created_at: float = field(default_factory=time.time)
    output_files: list[Path] = field(default_factory=list)
    merge_output: bool = False
```

**`merge_output`:** when `True` and `input_files` contains more than one file, the output formatter concatenates all files' transcripts into a single output file per format (separated by `---\n# <original_filename>\n`). When `False`, each input file produces its own output file(s).

Jobs are processed sequentially (one at a time). The queue is in-memory and does not persist across app restarts — history is persisted separately.

**Error handling per chunk:** If a single chunk fails with a retryable error (network timeout, rate limit), it is retried up to 3 times with exponential backoff. Non-retryable errors (invalid API key, model not found, corrupt audio) abort the job immediately and surface a clear error message in the UI.

### History (`app/core/history.py`)

Completed (and failed) jobs are written to `app/transcriber.db` (SQLite) on job completion. Schema:

```sql
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,              -- Job UUID
    input_filenames TEXT,             -- JSON array of original filenames, e.g. '["a.mp3","b.mp3"]'
    provider_name TEXT,
    model_id TEXT,
    status TEXT,
    created_at REAL,
    duration_sec REAL,               -- total audio duration processed
    output_paths TEXT,               -- JSON array of absolute output file paths
    error_message TEXT
);
```

`input_filenames` and `output_paths` are serialized as JSON arrays (`json.dumps` / `json.loads`). Filenames only (not full paths) are stored in `input_filenames` for display purposes. Full absolute paths are stored in `output_paths` for download links.

The History UI reads from this table. Output file download links resolve against the actual filesystem path — if a file has been moved or deleted, the link shows "file not found" with an option to remove the entry.

---

## WebSocket Message Protocol

All WebSocket messages are JSON objects with a `type` field. Server → client:

```jsonc
// Progress update (batch transcription)
{ "type": "progress", "job_id": "uuid", "progress": 0.57, "message": "chunk 4/7" }

// New live transcript segment
{ "type": "segment", "text": "...", "start": 12.4, "end": 15.1, "speaker": null }

// Job completed
{ "type": "done", "job_id": "uuid", "output_files": ["path/to/out.srt", ...] }

// Error
{ "type": "error", "job_id": "uuid", "message": "Invalid API key", "retryable": false }

// Install progress (dependency download)
{ "type": "install_progress", "package": "faster-whisper", "progress": 0.34, "message": "Downloading..." }

// Install done
{ "type": "install_done", "package": "faster-whisper", "success": true, "error": null }
```

Client → server:

```jsonc
// Subscribe to a batch job's updates
{ "type": "subscribe", "job_id": "uuid" }

// Cancel a running batch job
{ "type": "cancel", "job_id": "uuid" }

// Start a live transcription session (must be sent before audio_chunk)
// Server responds with { "type": "live_session_started", "session_id": "uuid" }
{ "type": "start_live", "provider_name": "faster_whisper", "model_id": "large-v3",
  "opts": { "language": null, "speaker_labels": false, "output_formats": ["txt", "srt"] } }

// Live audio chunk — must include session_id from start_live response
{ "type": "audio_chunk", "session_id": "uuid", "data": "<base64-encoded webm/opus>" }

// Stop live session and finalize output
{ "type": "stop_live", "session_id": "uuid" }
```

Server → client (live session lifecycle):
```jsonc
{ "type": "live_session_started", "session_id": "uuid" }
{ "type": "live_session_stopped", "session_id": "uuid", "output_files": [...] }
```

Each WebSocket connection is scoped to one session (batch job or live session). Multiple browser tabs may connect simultaneously. WebSocket scheme is always `ws://` — TLS/proxy support is out of scope for v1.

---

## Live Transcription Architecture

```
Browser (MediaRecorder API)
  └─ webm/opus chunks, ~3s each → base64 encoded
       └─ WebSocket "audio_chunk" message → FastAPI
            └─ server accumulates chunks into a 10s rolling buffer
                 └─ provider.transcribe_live(buffer)
                      ├─ faster-whisper: decode webm→pcm (ffmpeg), run ctranslate2 inference on 10s segment
                      ├─ Qwen3-ASR: decode webm→wav (ffmpeg), run transformers inference on 10s segment
                      └─ ElevenLabs Scribe v2 Realtime: forward raw webm chunks to ElevenLabs WebSocket
                 └─ Segment result → WebSocket "segment" message → Alpine.js UI
                      └─ incremental append to .txt every 30s (other formats generated at stop)
```

**Latency note:** Local model live transcription (faster-whisper, Qwen3-ASR) uses a 10-second buffer, producing one transcript update every ~10 seconds. This is a deliberate trade-off: smaller buffers produce worse accuracy on CPU. On GPU, the buffer can be reduced to 5 seconds. The UI displays a "buffered live" indicator for local models vs. "realtime" for ElevenLabs Scribe v2 Realtime.

**webm decoding:** `MediaRecorder` produces webm/opus. Before sending to local models, the server decodes each buffer chunk to PCM using `pydub` (which calls `ffmpeg`). If ffmpeg is not installed, live transcription with local models is disabled. ElevenLabs Scribe v2 Realtime accepts webm/opus natively and does not require this decode step.

**Audio recording:** Raw webm chunks are accumulated in memory and flushed to `<output_dir>/<session_id>.webm` every 30 seconds. On Stop, the full webm file is decoded to `.wav` via ffmpeg and offered for download alongside the transcript. If ffmpeg is missing, only the `.webm` is saved.

**Incremental save:** Every 30 seconds, the current accumulated segments are written to `<output_dir>/<session_id>_live.txt` (overwrite, not append, so the file always contains the full session so far). Other formats (srt, vtt, md) are generated only at Stop from the full segment list.

---

## Output Formats (`app/core/output.py`)

All formats receive a `TranscriptResult` and return a string.

| Format | Description |
|---|---|
| `.txt` | Plain text. Each segment on its own paragraph, blank line separator. Speaker prefix if available: `[Speaker 1] text` |
| `.srt` | SubRip. Sequence number, `HH:MM:SS,mmm --> HH:MM:SS,mmm`, text. Speaker prefix if available. |
| `.vtt` | WebVTT. Same as SRT but `HH:MM:SS.mmm` separators and `WEBVTT` header. |
| `.md` | Markdown. `## HH:MM:SS` timestamp headers, `**Speaker 1:**` bold labels if available. |

When batch merge is enabled across multiple input files, all output files are concatenated in queue order with a `---` separator and the source filename as a comment.

---

## Startup & Distribution

### `start.py` responsibilities
1. Check Python version (≥ 3.10), print clear message and exit if not met
2. Install/upgrade core pip packages from `requirements.txt`
3. Verify `app/static/alpine.min.js` and `app/static/tailwind.css` are present (they are committed — this is a sanity check, not a download)
4. Check ffmpeg — if missing, print OS-specific install instructions (non-blocking warning, not a hard failure)
5. Load `.env` if present
6. Try `localhost:8000`; if port is busy, try sequential ports 8001–8010; bind to the first available. The chosen port is stored in a module-level variable and injected into the Jinja2 template context as `{{ port }}` so the frontend can construct correct WebSocket URLs
7. Start uvicorn server
8. Open browser automatically (`webbrowser.open(f"http://localhost:{port}")`)

### Double-click launchers
- **Windows:** `start.bat` — `@echo off`, calls `python start.py`, `pause` on exit so errors are visible
- **macOS/Linux:** `start.sh` — `#!/bin/bash`, calls `python3 start.py`
- Both check for Python first with a human-readable error if not found

### First-time setup for students without Python
- `README.md` includes a one-paragraph "Install Python" section (python.org link, add to PATH checkbox reminder for Windows)
- `SETUP.md` with OS-specific step-by-step instructions and screenshots

### On-demand heavy dependencies
When a user selects a local model requiring heavy deps for the first time:
1. UI shows a modal: package name, disk space, estimated download time, "Install" / "Cancel" buttons
2. On confirm, backend calls `provider.install_deps()` in a background thread
3. `install_deps()` sends `install_progress` WebSocket messages as packages download
4. On completion, sends `install_done` — UI dismisses modal and makes the model selectable
5. No app restart required

---

## Settings (`app/settings.py`)

User preferences are stored in `settings.json` (gitignored) in the project root. API keys are stored in `.env` (gitignored). The two files are kept separate: `.env` is standard for secrets, `settings.json` for non-secret preferences.

`settings.json` schema:
```json
{
  "language": "en",
  "output_dir": null,
  "chunk_size_sec": 600,
  "default_provider": "openai",
  "default_model": "whisper-1",
  "default_output_formats": ["txt", "srt"]
}
```

`output_dir`: if `null`, output files are saved in the same directory as the input file. For **live transcription sessions** (no input file), output is saved to `~/Documents/Transcriber/` (created automatically if it does not exist). Users can override this in Settings.

---

## API Key Security

- Keys stored in `.env` only — never in source code
- `.env` listed in `.gitignore`
- `.env.example` with placeholder values committed instead
- `API KEY.txt` (legacy file) — excluded from all commits via `.gitignore`. Users who have their key there must copy it to `.env` manually; `settings.py` does **not** read `API KEY.txt` automatically (migration guide in `SETUP.md`)
- Keys are never logged, never sent to the frontend (masked preview `sk-...abc` only)
- Validation happens server-side via a test API call on the Settings page and before any transcription starts

---

## UI Design

### Layout: Sidebar + Main Panel

A persistent narrow sidebar (64px) with icon navigation. Main panel changes per section. Dark theme default, respects system preference via CSS `prefers-color-scheme`.

### Sections

**1. Transcribe Files** (📂)
- Drag & drop zone accepting mp3, m4a, wav, ogg, flac, webm
- File queue with per-file progress bar, status, estimated time remaining
- Provider selector (dropdown) + Model selector (contextual to provider, showing hardware hint badge)
- Output format selector: txt / srt / vtt / md (multi-select checkboxes)
- "Merge all outputs" toggle for batch sessions
- "Start Queue" / "Stop" buttons
- Keyterm prompt field (shown when ElevenLabs Scribe v2 selected)
- Speaker labels toggle (shown when provider supports it)
- Error states: per-file error badge with message; retryable errors show "Retry" button

**2. Live Mode** (🔴)
- Microphone selector (browser `getUserMedia` device enumeration)
- Provider + model selector (only live-capable providers shown; local models show "buffered" badge)
- Waveform visualizer (Web Audio API `AnalyserNode`)
- Real-time transcript scrolling (last segment highlighted, cursor blinking)
- Recording timer
- "Save audio + transcript" toggle (default: on)
- "Pause" / "Stop & Save" buttons
- Post-processing modal: after stopping, offer to re-transcribe saved audio with any batch provider
- If ffmpeg missing: live mode for local models shows warning "ffmpeg required for local live transcription"

**3. History** (📋)
- List of past sessions from SQLite, ordered by date descending
- Each entry: filename(s), date, provider/model, audio duration, output file links with download
- "Re-run" button (pre-fills settings from that session)
- "Delete" button (removes DB entry; does not delete output files)

**4. Settings** (⚙️)
- API keys: OpenAI, ElevenLabs — masked input, "Validate" button per key
- Local models: per-model download status, install/remove button, disk usage
- ffmpeg status with OS-specific install command
- Language selector (EN / IT)
- Default output directory (browse button, shows current value, "Same as input file" default)
- Default output formats
- Chunk size (advanced, default 600s)

### First-Run Wizard

Shown on first launch if `settings.json` does not exist or has never been completed:
1. Language selection
2. "How do you want to transcribe?" — API (OpenAI / ElevenLabs) / Local models / I'll decide later
3. If API selected: key input with validation + "Get a key" link
4. ffmpeg check: green check or OS-specific install instructions
5. Done — wizard completion flag written to `settings.json`

Re-accessible from Settings → "Run setup wizard again".

---

## Internationalization

- All UI strings in `app/locales/en.json` and `app/locales/it.json`
- Language stored in `settings.json` under `"language"` key
- Locale JSON is loaded server-side and injected into the Jinja2 template as a JS variable `window.LOCALE`
- Alpine.js components access translations via a global `$t('key')` magic function
- Adding a new language: add `app/locales/<code>.json`, add the option to the Settings language selector

---

## Files to Remove from Current Repo

Before building the new structure:

**Delete:**
- `main_ui.py` — replaced by FastAPI + templates
- `transcribe2.py` — logic refactored into `app/providers/`
- `split_audio.py` — refactored into `app/core/audio.py`
- `AudioTranscriber.spec` — PyInstaller artifact
- `AppIcon.icns` — PyInstaller artifact
- `.DS_Store` — add to `.gitignore`
- `start.bh.txt` — unknown temp file
- `build/`, `dist/` — PyInstaller build artifacts
- `__pycache__/` — add to `.gitignore`

**Add to `.gitignore` (do not delete, just ignore):**
- `audio_chunks/`
- `.env`
- `API KEY.txt`
- `settings.json`
- `app/transcriber.db`
- `*.pyc`, `__pycache__/`, `.venv/`, `.DS_Store`

**Keep:**
- `start.py` — will be rewritten
- `requirements.txt` — will be updated
- `README.md` — will be rewritten
- `.env` / `API KEY.txt` — kept locally, never committed

---

## Open Source

- License: **MIT** — `LICENSE` file in repo root
- `CONTRIBUTING.md`: dev setup (Python only for core; Node.js optional for Tailwind CSS rebuild)
- `.env.example` committed with all supported key names and comments
- GitHub Actions CI: `ruff` lint + basic import smoke test on Python 3.10, 3.11, 3.12

---

## Out of Scope (v1)

- Speaker diarization for local models (requires pyannote.audio — future)
- Cloud deployment / SaaS version
- Mobile support
- Plugin system
- Real-time collaboration
- Qwen3-ASR live mode (pending stable streaming release)
- Word-level timestamps (segment-level only in v1)
