# Audio Lab — Audio Preprocessing Pipeline

**Date:** 2026-03-18
**Status:** Approved

## Overview

A dedicated "Audio Lab" tab in the Transcriber web app that lets users preprocess audio files before transcription. The pipeline normalizes volume, isolates voice from background noise, and applies neural denoising. A real-time A/B player allows comparing the original and processed audio at the same playback position.

**Target user:** University students transcribing lecture recordings with poor audio quality (echo, background chatter, low volume, HVAC noise).

## Decisions

| Question | Decision |
|----------|----------|
| Where in the UI? | Dedicated "Audio Lab" tab (between Live and History) |
| Preset vs manual? | Presets with custom override (Lecture / Clean / Custom) |
| Before/after preview? | Single player with A/B switch at same timestamp |
| Dependencies bundled? | Yes — installed with the app (demucs, deepfilternet in requirements) |

## Architecture

### Processing Pipeline (`app/core/preprocess.py`)

Three sequential steps, each independently toggleable. The pipeline works at **48kHz** internally (the native rate for DeepFilterNet) and downsamples to **16kHz mono** only at the very end for optimal transcription input.

```
input.m4a
  │
  ▼ FFmpeg decode → WAV 48kHz mono (pipeline working rate)
  │
  ▼ Step 0 (always): LUFS analysis
  │   FFmpeg loudnorm Pass 1 only — measure original integrated LUFS,
  │   true peak, LRA. Stored in stats for the UI even if loudnorm is off.
  │
  ▼ Step 1: Loudnorm (FFmpeg loudnorm filter, EBU R128)
  │   Pass 2: apply normalization to target LUFS (default -16)
  │   Input: 48kHz WAV → Output: 48kHz WAV
  │
  ▼ Step 2: Voice Isolation (Demucs htdemucs)
  │   Demucs resamples internally to 44.1kHz, outputs stems at 44.1kHz.
  │   We resample the vocals stem back to 48kHz after extraction.
  │   Input: 48kHz WAV → [internal 44.1kHz] → Output: 48kHz WAV
  │
  ▼ Step 3: Denoise (DeepFilterNet)
  │   Trained on 48kHz audio — fed at native rate.
  │   Input: 48kHz WAV → Output: 48kHz WAV
  │
  ▼ Final: FFmpeg resample 48kHz → 16kHz mono WAV (16-bit)
  │   Optimal format for all speech recognition models.
  │
  ▼ output.wav (16-bit, 16kHz mono)
```

**Key design choices:**
- Internal pipeline runs at 48kHz to match DeepFilterNet's native rate.
- Demucs works at 44.1kHz internally; we resample its output back to 48kHz before DeepFilterNet.
- Final output is downsampled to 16kHz mono — optimal for speech recognition.
- Each step reads WAV and writes WAV. No lossy re-encoding between steps.
- The original file is never modified. All processing creates new files.
- Output is WAV (not MP3) to avoid quality loss before transcription.
- A 48kHz WAV copy is also kept for the A/B player (better listening quality than 16kHz).

### Presets

| Preset | Loudnorm | Voice Isolation | Denoise |
|--------|----------|-----------------|---------|
| Lecture (🎓) | ON (-16 LUFS) | ON | ON |
| Clean Recording (🎙️) | ON (-16 LUFS) | OFF | OFF |
| Custom (🔧) | user choice | user choice | user choice |

- Selecting a preset sets the toggles automatically.
- Switching to Custom preserves the current toggle state.
- In Lecture and Clean modes, toggles are visible but disabled (grayed out) so the user can see what each preset does.
- If all toggles are off in Custom mode, the "Process" button is disabled (nothing to do).

### Loudnorm Parameters (Custom mode)

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| Target LUFS | -24 to -10 | -16 | Target integrated loudness |

Voice Isolation and Denoise have no user-facing parameters — they either run or don't.

### Cancellation

The pipeline checks a cancellation flag between each step. If cancelled:
- Current step is allowed to finish (Demucs cannot be interrupted mid-inference)
- Subsequent steps are skipped
- Partial result is served if at least one step completed
- UI shows "Cancelled" status

## API Endpoints

### `POST /api/audiolab/process`

Upload an audio file and start preprocessing.

**Request:** multipart/form-data
- `file`: audio file (same extensions as transcription: mp3, m4a, mp4, wav, ogg, flac, webm, aac, wma, opus)
- `preset`: `"lecture"` | `"clean"` | `"custom"`
- `loudnorm`: boolean (only for custom)
- `loudnorm_target`: float (only for custom, default -16)
- `voice_isolation`: boolean (only for custom)
- `denoise`: boolean (only for custom)

**Response:**
```json
{
  "job_id": "uuid",
  "status": "processing"
}
```

### `POST /api/audiolab/cancel/{job_id}`

Cancel a running Audio Lab job. Returns 200 if cancelled, 404 if job not found, 409 if already done.

### `GET /api/audiolab/preview/{job_id}?which=original|processed`

Serves the audio file for the browser `<audio>` element via `FileResponse` (Starlette's `FileResponse` handles HTTP Range headers automatically via `stat_result` for seeking support).

### `POST /api/audiolab/send-to-transcribe`

**Request:**
```json
{
  "job_id": "uuid"
}
```

Backend directly creates a transcription job using the processed file — no re-upload through the browser. Returns the new transcription job_id. Frontend subscribes to that job via WebSocket and switches to the Transcribe tab.

**Response:**
```json
{
  "transcription_job_id": "uuid",
  "filename": "Lezione 1 Revisione (processed).wav"
}
```

### `GET /api/audiolab/download/{job_id}`

Downloads the processed WAV file (16kHz version for transcription use).

## Audio Lab Job State

Audio Lab jobs are stored in an in-memory dict `_audiolab_jobs: dict[str, AudioLabJob]`. They do not persist across server restarts (ephemeral by design — the user can always re-process).

```python
@dataclass
class AudioLabJob:
    id: str
    status: str  # "processing" | "done" | "error" | "cancelled"
    original_path: Path
    processed_path: Path | None
    processed_48k_path: Path | None  # for A/B player (better quality)
    progress: float  # 0.0 - 1.0
    current_step: str  # "decode" | "loudnorm" | "demucs" | "deepfilter" | "resample"
    message: str
    cancelled: bool  # cancellation flag checked between steps
    stats: dict | None  # {original_lufs, processed_lufs, duration_sec, original_size, processed_size}
    original_filename: str  # preserve original name for display
```

## Frontend

### Tab Layout

```
┌─────────────────────────────────────────────┐
│  Drop zone                                  │
│  "Drop an audio file here or browse"        │
│  (same extensions as Transcribe tab)        │
├─────────────────────────────────────────────┤
│  Preset: [🎓 Lecture] [🎙️ Clean] [🔧 Custom]│
├─────────────────────────────────────────────┤
│  ☑ Loudnorm    [-16 LUFS ──●──────]        │
│  ☑ Voice Isolation (Demucs)                 │
│  ☑ Denoise (DeepFilterNet)                  │
│  (toggles visible in all modes,             │
│   editable only in Custom)                  │
├─────────────────────────────────────────────┤
│  [ ▶ Process ] [ ✕ Cancel ]  [progress bar] │
├─────────────────────────────────────────────┤
│  Player A/B:                                │
│  ▶ advancement bar ──●────── 1:23 / 5:00   │
│  [ A Original | B Processed ]   🔊 volume   │
│                                              │
│  Stats: -32 LUFS → -16 LUFS | 5:23 | 12MB  │
├─────────────────────────────────────────────┤
│  [ 📥 Download ]  [ → Transcribe ]          │
└─────────────────────────────────────────────┘
```

### A/B Player Implementation

- Two `<audio>` elements: one for original (48kHz), one for processed (48kHz)
- Single seekbar controls both — `currentTime` synced on every seek/play
- **A button**: mutes processed, unmutes original
- **B button**: mutes original, unmutes processed
- Switch is instantaneous (no reload, no gap) — just mute/unmute
- Keyboard shortcuts: `Space` = play/pause, `A`/`B` = switch source — **only active when Audio Lab tab is focused and no input element has focus**
- Volume slider affects both equally
- Stats bar shows before/after LUFS, file duration, file size

### Progress Reporting

Via WebSocket (same connection as transcription):

| Message Type | Payload |
|-------------|---------|
| `audiolab_progress` | `{job_id, progress: 0-1, step: "decode\|loudnorm\|demucs\|deepfilter\|resample", message}` |
| `audiolab_done` | `{job_id, original_url, processed_url, stats: {duration_sec, original_lufs, processed_lufs, original_size, processed_size}}` |
| `audiolab_error` | `{job_id, message}` |

### "→ Transcribe" Flow

1. Frontend calls `POST /api/audiolab/send-to-transcribe` with the job_id
2. Backend creates a transcription job directly using the processed WAV (no browser re-upload)
3. Backend returns `{transcription_job_id, filename}`
4. Frontend adds the file to `files[]` with the returned job_id and subscribes via WebSocket
5. Frontend switches to the Transcribe tab (`currentSection = 'transcribe'`)
6. File appears in the queue ready to be transcribed (or auto-starts if queue is idle)

## File Structure

### New files
- `app/core/preprocess.py` — pipeline logic (loudnorm, demucs, deepfilternet)
- `tests/test_preprocess.py` — unit tests

### Modified files
- `app/main.py` — new `/api/audiolab/*` endpoints + AudioLabJob state
- `app/templates/index.html` — Audio Lab tab UI
- `app/static/app.js` — Audio Lab Alpine.js state and methods
- `app/locales/en.json` — English strings
- `app/locales/it.json` — Italian strings
- `requirements.txt` — add demucs, deepfilternet

### Temporary files
- `audiolab_cache/<job_id>/original.wav` — decoded original (48kHz, for player)
- `audiolab_cache/<job_id>/processed_48k.wav` — processed at 48kHz (for player)
- `audiolab_cache/<job_id>/processed.wav` — final 16kHz output (for transcription/download)
- Stored in system temp dir (`tempfile.gettempdir() / "transcriber_audiolab"`)
- Cleaned up: on new file upload (previous job's files deleted), on server shutdown (FastAPI shutdown event), and any files older than 24h on server startup

## Dependencies

| Package | Size | Purpose |
|---------|------|---------|
| `demucs` | ~300MB model (first use) + deps (~500MB total with torchaudio, julius, etc.) | Voice isolation via htdemucs |
| `deepfilternet` | ~80MB model | Neural noise reduction |
| `torch` | already installed (XPU build) | Required by both |
| `torchaudio` | already installed (XPU build) | Required by demucs |
| `ffmpeg` | already required | Loudnorm filter, audio decode, resample |

Note: torch and torchaudio are already present because the user has qwen3_asr set up. For users without local providers, demucs will pull in torch as a dependency (~2GB additional download). This is documented in the install process.

## Error Handling

- **FFmpeg not found:** Show error "ffmpeg is required for Audio Lab" (same check as transcription)
- **Demucs/DeepFilterNet import fails:** Skip unavailable steps with a warning. Loudnorm always works (FFmpeg only). Show banner: "Install demucs/deepfilternet for voice isolation and denoise"
- **Processing fails mid-pipeline:** Serve partial result with warning (e.g., loudnorm succeeded but demucs failed → serve loudnorm-only output, show which steps completed)
- **File too large:** No hard limit — warn if > 500MB that processing may take several minutes
- **Disk full:** Catch IOError during WAV write, report clearly

## Testing Strategy

- Unit tests for `preprocess.py` using short synthetic audio (pydub-generated sine waves)
- Mock `demucs` and `deepfilternet` in tests to avoid downloading models
- Test sample rate handling: verify 48kHz→44.1kHz→48kHz→16kHz chain
- Test cancellation: verify partial results are served
- Test cleanup: verify temp files are removed
- Integration test: upload file → process → verify output exists and is valid WAV
- Frontend: manual testing of A/B player sync, preset switching, progress display, keyboard shortcuts (verify no conflict with text inputs)
