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

Three sequential steps, each independently toggleable:

```
input.m4a
  │
  ▼ FFmpeg decode → WAV 16kHz mono
  │
  ▼ Step 1: Loudnorm (FFmpeg loudnorm filter, EBU R128)
  │   Pass 1: analyze (measure integrated LUFS, true peak, LRA)
  │   Pass 2: apply normalization to target LUFS (default -16)
  │
  ▼ Step 2: Voice Isolation (Demucs htdemucs)
  │   Separates into 4 stems: vocals, drums, bass, other
  │   Keeps only "vocals" stem, discards everything else
  │
  ▼ Step 3: Denoise (DeepFilterNet)
  │   Removes residual noise from the isolated vocal signal
  │
  ▼ output.wav (16-bit, 16kHz mono)
```

**Key design choices:**
- Input is always decoded to WAV 16kHz mono first — this is the optimal format for all speech recognition models and ensures consistent behavior across the pipeline.
- Each step reads WAV and writes WAV. No lossy re-encoding between steps.
- The original file is never modified. All processing creates new files.
- Output is WAV (not MP3) to avoid quality loss before transcription.

### Presets

| Preset | Loudnorm | Voice Isolation | Denoise |
|--------|----------|-----------------|---------|
| Lecture (🎓) | ON (-16 LUFS) | ON | ON |
| Clean Recording (🎙️) | ON (-16 LUFS) | OFF | OFF |
| Custom (🔧) | user choice | user choice | user choice |

- Selecting a preset sets the toggles automatically.
- Switching to Custom preserves the current toggle state.
- In Lecture and Clean modes, toggles are visible but disabled (grayed out) so the user can see what each preset does.

### Loudnorm Parameters (Custom mode)

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| Target LUFS | -24 to -10 | -16 | Target integrated loudness |

Voice Isolation and Denoise have no user-facing parameters — they either run or don't.

## API Endpoints

### `POST /api/audiolab/process`

Upload an audio file and start preprocessing.

**Request:** multipart/form-data
- `file`: audio file
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

### `GET /api/audiolab/status/{job_id}`

**Response:**
```json
{
  "status": "processing|done|error",
  "progress": 0.65,
  "current_step": "demucs",
  "message": "Isolating voice..."
}
```

### `GET /api/audiolab/preview/{job_id}?which=original|processed`

Serves the audio file for the browser `<audio>` element. Supports HTTP Range requests for seeking.

### `POST /api/audiolab/send-to-transcribe`

**Request:**
```json
{
  "job_id": "uuid"
}
```

Copies the processed file to the transcription upload queue and switches the frontend to the Transcribe tab.

### `GET /api/audiolab/download/{job_id}`

Downloads the processed WAV file.

## Frontend

### Tab Layout

```
┌─────────────────────────────────────────────┐
│  Drop zone                                  │
│  "Drop an audio file here or browse"        │
├─────────────────────────────────────────────┤
│  Preset: [🎓 Lecture] [🎙️ Clean] [🔧 Custom]│
├─────────────────────────────────────────────┤
│  ☑ Loudnorm    [-16 LUFS ──●──────]        │
│  ☑ Voice Isolation (Demucs)                 │
│  ☑ Denoise (DeepFilterNet)                  │
│  (toggles visible in all modes,             │
│   editable only in Custom)                  │
├─────────────────────────────────────────────┤
│  [ ▶ Process ]               [progress bar] │
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

- Two `<audio>` elements: one for original, one for processed
- Single seekbar controls both — `currentTime` synced on every seek/play
- **A button**: mutes processed, unmutes original
- **B button**: mutes original, unmutes processed
- Switch is instantaneous (no reload, no gap) — just mute/unmute
- Keyboard shortcuts: `Space` = play/pause, `A`/`B` = switch source
- Volume slider affects both equally
- Stats bar shows before/after LUFS, file duration, file size

### Progress Reporting

Via WebSocket (same connection as transcription):

| Message Type | Payload |
|-------------|---------|
| `audiolab_progress` | `{job_id, progress: 0-1, step: "loudnorm\|demucs\|deepfilter", message}` |
| `audiolab_done` | `{job_id, original_url, processed_url, stats: {duration_sec, original_lufs, processed_lufs, original_size, processed_size}}` |
| `audiolab_error` | `{job_id, message}` |

### "→ Transcribe" Flow

1. Frontend calls `POST /api/audiolab/send-to-transcribe` with the job_id
2. Backend copies the processed WAV to a temp upload path
3. Backend returns the file info (name, path)
4. Frontend adds the file to the `files[]` array in the Transcribe tab
5. Frontend switches to the Transcribe tab (`currentSection = 'transcribe'`)
6. File appears in the queue ready to be transcribed

## File Structure

### New files
- `app/core/preprocess.py` — pipeline logic (loudnorm, demucs, deepfilternet)
- `tests/test_preprocess.py` — unit tests

### Modified files
- `app/main.py` — new `/api/audiolab/*` endpoints
- `app/templates/index.html` — Audio Lab tab UI
- `app/static/app.js` — Audio Lab Alpine.js state and methods
- `app/locales/en.json` — English strings
- `app/locales/it.json` — Italian strings
- `requirements.txt` — add demucs, deepfilternet

### Temporary files
- `audiolab_cache/<job_id>/original.wav` — decoded original
- `audiolab_cache/<job_id>/processed.wav` — final output
- Cleaned up when user loads a new file or on server restart

## Dependencies

| Package | Size | Purpose |
|---------|------|---------|
| `demucs` | ~300MB model (downloaded on first use) | Voice isolation via htdemucs |
| `deepfilternet` | ~80MB model | Neural noise reduction |
| `torch` | already installed | Required by both |
| `ffmpeg` | already required | Loudnorm filter, audio decoding |

## Error Handling

- **FFmpeg not found:** Show error "ffmpeg is required for Audio Lab" (same check as transcription)
- **Demucs/DeepFilterNet import fails:** Show error with install instructions
- **Processing fails mid-pipeline:** Return partial result if possible (e.g., loudnorm succeeded but demucs failed → serve loudnorm-only output with warning)
- **File too large:** No hard limit — warn if > 500MB that processing may take several minutes

## Testing Strategy

- Unit tests for `preprocess.py` using short synthetic audio (pydub-generated sine waves)
- Mock `demucs` and `deepfilternet` in tests to avoid downloading models
- Integration test: upload file → process → verify output exists and is valid WAV
- Frontend: manual testing of A/B player sync, preset switching, progress display
