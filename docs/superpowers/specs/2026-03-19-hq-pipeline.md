# HQ Audio Pipeline — Spec

**Date:** 2026-03-19
**Status:** Approved
**Scope:** Reorder audio pipeline for all presets, add chunked Demucs processing, add "Polish" step group (high-pass, de-esser, EQ, compressor, limiter), add HQ preset.

---

## 1. Problem

The current pipeline order (loudnorm → voice isolation → denoise) is suboptimal:
- Loudnorm amplifies noise before denoise removes it
- Denoise runs after voice isolation has already stripped harmonics, producing a thinner sound
- Demucs loads the entire file into GPU VRAM, causing OOM on files longer than ~30 minutes on a 16GB Intel Arc GPU
- There is no option for audio "polishing" (EQ warmth, compression, de-essing) that would make the output pleasant for human listening while remaining optimal for ASR

## 2. Solution

### 2.1 Pipeline Reorder (all presets)

New order for all presets:

```
Decode 48kHz mono
  → Denoise (afftdn / DeepFilterNet)
  → Voice Isolation (Demucs, chunked)
  → Polish (high-pass, de-esser, EQ, compressor, limiter)  [only if enabled]
  → Loudnorm (target LUFS)
  → output 48kHz + resample 16kHz
```

Key changes:
- **Denoise before voice isolation** — works on the original signal, richer in harmonics
- **Loudnorm at the end** — normalizes the final result, not an intermediate signal
- Each step only runs if enabled in the config

### 2.2 Chunked Demucs

Split long audio into 5-minute chunks with 5-second overlap, process each chunk individually on GPU, crossfade overlap regions, reassemble. Completely transparent to the user.

- Chunk size: 5 minutes (300 seconds × sample_rate samples)
- Overlap: 5 seconds (5 × sample_rate samples)
- Step between chunks: chunk_size - overlap
- Crossfade: linear ramp in the overlap region
- Falls back to CPU on OOM (existing safety net remains)
- For files shorter than 5 minutes, processes as a single chunk (no overhead)

### 2.3 Polish Step Group

Five ffmpeg audio filters executed in a single ffmpeg pass:

1. **High-pass 80Hz** — removes rumble, table bumps, HVAC vibrations
2. **De-esser** — attenuates harsh sibilants (4-8kHz) using boost→compress→cut technique
3. **EQ coloration** — lowshelf +2dB@250Hz (body), peak +1.5dB@3kHz (presence), highshelf +1dB@10kHz (air)
4. **Gentle compressor** — ratio 2:1, threshold 0.1 linear (~-20dB), attack 20ms, release 200ms
5. **Limiter** — brick wall at -1dBTP (linear 0.891)

All five activate/deactivate together via a single `polish` boolean.

### 2.4 HQ Preset

New preset between Lecture and Clean that enables all processing including Polish.

## 3. Architecture

### 3.1 Updated Pipeline Flow

```
PreprocessConfig.polish = True/False
PreprocessConfig.denoise = True/False
PreprocessConfig.voice_isolation = True/False
PreprocessConfig.loudnorm = True/False
                    │
        ┌───────────┴───────────────────┐
        ▼                               │
   decode_to_wav()                      │
        │                               │
        ▼ [if denoise]                  │
   apply_denoise()                      │
        │                               │
        ▼ [if voice_isolation]          │
   apply_voice_isolation()              │
   (internally: _run_demucs_chunked)    │
        │                               │
        ▼ [if polish]                   │
   _apply_polish()                      │
        │                               │
        ▼ [if loudnorm]                 │
   apply_loudnorm()                     │
        │                               │
        ▼                               │
   output 48kHz + resample 16kHz        │
        └───────────────────────────────┘
```

### 3.2 Chunked Demucs Detail

```
Input WAV (e.g., 1 hour = 158,760,000 samples @ 44.1kHz)
        │
        ▼
Split into chunks:
  chunk 0: samples [0 .. 13,230,000]           (5 min)
  chunk 1: samples [13,009,500 .. 26,239,500]   (5 min, overlap 5s)
  chunk 2: samples [26,019,000 .. 39,249,000]   (5 min, overlap 5s)
  ...
        │
        ▼ (each chunk)
  Pad to stereo → normalize → send to GPU → Demucs → extract vocals → to CPU
        │
        ▼
Reassemble with linear crossfade in overlap zones:
  overlap region: fade_out(chunk_n) × (1-t) + fade_in(chunk_n+1) × t
        │
        ▼
Mono vocals numpy array at model sample rate
```

## 4. Backend Changes

### 4.1 `app/core/preprocess.py`

#### PreprocessConfig

```python
@dataclass
class PreprocessConfig:
    loudnorm: bool = True
    loudnorm_target: float = -16.0
    voice_isolation: bool = True
    denoise: bool = True
    denoise_engine: str = "ffmpeg"
    polish: bool = False  # NEW
```

#### New function: `_apply_polish()`

```python
def _apply_polish(wav_path: Path, output_path: Path) -> Path:
    """Apply audio polish: high-pass, de-esser, EQ, compressor, limiter."""
    ffmpeg = _require_ffmpeg()
    af = ",".join([
        # High-pass: remove rumble below 80Hz
        "highpass=f=80:poles=2",
        # De-esser: boost sibilant band → compress → cut back
        "equalizer=f=6000:width_type=s:width=2.0:g=3",
        "acompressor=threshold=0.1:ratio=4:attack=5:release=50",
        "equalizer=f=6000:width_type=s:width=2.0:g=-3",
        # EQ: body (shelf) + presence (peak) + air (shelf)
        "lowshelf=f=250:width_type=s:width=0.8:g=2",
        "equalizer=f=3000:width_type=s:width=1.0:g=1.5",
        "highshelf=f=10000:width_type=s:width=0.8:g=1",
        # Gentle compressor
        "acompressor=threshold=0.1:ratio=2:attack=20:release=200",
        # Limiter -1dBTP
        "alimiter=limit=0.891:level=disabled",
    ])
    cmd = [
        ffmpeg, "-y", "-i", str(wav_path),
        "-af", af,
        "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le",
        str(output_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        stderr_text = result.stderr.decode(errors="replace")
        raise RuntimeError(f"ffmpeg polish failed: {stderr_text}")
    return output_path.resolve()
```

#### Refactored: `_run_demucs()` → `_run_demucs_chunked()`

Replace the current `_run_demucs()` with a chunked version:

```python
DEMUCS_CHUNK_SEC = 300    # 5 minutes
DEMUCS_OVERLAP_SEC = 5    # 5 seconds crossfade

def _run_demucs_chunked(wav_path: Path) -> np.ndarray:
    """Run Demucs with chunked processing to avoid GPU OOM."""
    model, device = _get_demucs_model()

    wav, sr = torchaudio.load(str(wav_path))
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    if sr != model.samplerate:
        wav = torchaudio.transforms.Resample(sr, model.samplerate)(wav)
    sr = model.samplerate

    total_samples = wav.shape[1]
    chunk_samples = DEMUCS_CHUNK_SEC * sr
    overlap_samples = DEMUCS_OVERLAP_SEC * sr
    step_samples = chunk_samples - overlap_samples

    # For short files, process as single chunk
    if total_samples <= chunk_samples:
        return _process_single_demucs(model, wav, device)

    vocals = np.zeros(total_samples, dtype=np.float32)

    for start in range(0, total_samples, step_samples):
        end = min(start + chunk_samples, total_samples)
        chunk = wav[:, start:end]
        chunk_vocals = _process_demucs_chunk(model, chunk, device)

        # Crossfade into output
        if start == 0:
            vocals[start:end] = chunk_vocals
        else:
            overlap_end = min(start + overlap_samples, total_samples)
            ol_len = overlap_end - start
            fade_in = np.linspace(0, 1, ol_len, dtype=np.float32)
            fade_out = 1.0 - fade_in
            vocals[start:overlap_end] = (
                vocals[start:overlap_end] * fade_out +
                chunk_vocals[:ol_len] * fade_in
            )
            if overlap_end < end:
                vocals[overlap_end:end] = chunk_vocals[ol_len:end - start]

        if end >= total_samples:
            break

    return vocals
```

Helper to extract model init from _run_demucs:

```python
def _get_demucs_model():
    """Get or create cached Demucs model."""
    model_name = "htdemucs"
    if model_name not in _demucs_model_cache:
        model = get_model(model_name)
        device = _select_device()
        model.to(device)
        if device != "cpu":
            try:
                model = model.to(memory_format=torch.channels_last)
            except Exception:
                pass
        _demucs_model_cache[model_name] = (model, device)
    return _demucs_model_cache[model_name]

def _process_demucs_chunk(model, chunk_wav, device) -> np.ndarray:
    """Process a single chunk through Demucs, with OOM fallback."""
    ref = chunk_wav.mean(0)
    chunk_wav = (chunk_wav - ref.mean()) / ref.std()
    chunk_wav = chunk_wav.unsqueeze(0).to(device)

    try:
        with torch.no_grad():
            if device != "cpu":
                with torch.autocast(device_type=device, dtype=torch.float16):
                    sources = apply_model(model, chunk_wav, device=device)
            else:
                sources = apply_model(model, chunk_wav, device=device)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            logger.warning("GPU OOM on chunk — falling back to CPU")
            if hasattr(torch, "xpu"):
                torch.xpu.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
            model_cpu = model.cpu()
            chunk_cpu = chunk_wav.cpu()
            with torch.no_grad():
                sources = apply_model(model_cpu, chunk_cpu, device="cpu")
            model.to(device)
        else:
            raise

    vocals_idx = model.sources.index("vocals")
    return sources[0, vocals_idx].mean(0).cpu().float().numpy()
```

#### `run_pipeline()` — new order

```python
# Step 0: Decode
current = decode_to_wav(input_path, ...)

# Step 0.1: LUFS analysis (always — needed for stats even if loudnorm off)
measured = analyze_lufs(current)

# Cancellation checkpoint between every step (existing pattern preserved)

# Step 1: Denoise (before voice isolation — works on original harmonics)
if config.denoise:
    denoised = work_dir / "step1_denoised.wav"
    apply_denoise(current, denoised, engine=config.denoise_engine)
    current = denoised

# Step 3: Voice Isolation (chunked Demucs)
if config.voice_isolation:
    isolated = work_dir / "step2_vocals.wav"
    apply_voice_isolation(current, isolated)
    current = isolated

# Step 4: Polish
if config.polish:
    polished = work_dir / "step3_polished.wav"
    _apply_polish(current, polished)
    current = polished

# Step 5: Loudnorm (at the end — uses measured stats from analyze_lufs)
if config.loudnorm:
    normed = work_dir / "step4_loudnorm.wav"
    apply_loudnorm(current, normed, target_lufs=config.loudnorm_target,
                   measured=measured)
    current = normed
```

### 4.2 `app/main.py`

#### `/api/audiolab/process` endpoint

Add `polish` form parameter:

```python
polish: bool = Form(False),
```

Add HQ preset branch:

```python
if preset == "lecture":
    config = PreprocessConfig(loudnorm=True, loudnorm_target=-16.0,
                              voice_isolation=True, denoise=True,
                              denoise_engine=denoise_engine, polish=False)
elif preset == "hq":
    config = PreprocessConfig(loudnorm=True, loudnorm_target=-16.0,
                              voice_isolation=True, denoise=True,
                              denoise_engine=denoise_engine, polish=True)
elif preset == "clean":
    config = PreprocessConfig(loudnorm=True, loudnorm_target=-16.0,
                              voice_isolation=False, denoise=False,
                              denoise_engine=denoise_engine, polish=False)
else:  # custom
    config = PreprocessConfig(
        loudnorm=loudnorm, loudnorm_target=loudnorm_target,
        voice_isolation=voice_isolation, denoise=denoise,
        denoise_engine=denoise_engine, polish=polish,
    )
```

### 4.3 No changes needed

- `app/settings.py` — polish is not a global preference, it's per-preset
- `/api/audiolab/deps` — no new dependencies required (all ffmpeg-based)
- Install endpoints — no changes

## 5. Frontend Changes

### 5.1 `app/static/app.js`

#### New state

```javascript
alPolish: false,
```

#### `alSetPreset()` updated

```javascript
alSetPreset(p) {
    this.alPreset = p;
    if (p === 'lecture') {
        this.alLoudnorm = true; this.alVoiceIsolation = true;
        this.alDenoise = true; this.alPolish = false;
    } else if (p === 'hq') {
        this.alLoudnorm = true; this.alVoiceIsolation = true;
        this.alDenoise = true; this.alPolish = true;
    } else if (p === 'clean') {
        this.alLoudnorm = true; this.alVoiceIsolation = false;
        this.alDenoise = false; this.alPolish = false;
    }
}
```

#### `alCanProcess()` updated

Include polish in the check:

```javascript
alCanProcess() {
    return this.alFile && (this.alLoudnorm || this.alVoiceIsolation ||
                           this.alDenoise || this.alPolish);
}
```

#### `alProcess()` change

Send `polish` in FormData (only in custom mode, matching existing pattern):

```javascript
if (this.alPreset === 'custom') {
    // ... existing params ...
    fd.append("polish", this.alPolish);
}
```

### 5.2 `app/templates/index.html`

#### Preset buttons

Add HQ button between Lecture and Clean:

```
[🎓 Lecture]  [✨ HQ]  [🎙️ Clean]  [🔧 Custom]
```

#### Polish toggle

Below Denoise toggle, visible only in Custom mode:

```
[Polish ☐]  Voce calda (EQ, compressore, de-esser)
```

Simple checkbox with description, no sub-toggles.

## 6. Locale Strings

### English (`en.json`)

```json
{
  "audiolab_preset_hq": "HQ",
  "audiolab_polish": "Polish",
  "audiolab_polish_desc": "Warm voice (EQ, compressor, de-esser)",
  "audiolab_step_polish": "Polish"
}
```

### Italian (`it.json`)

```json
{
  "audiolab_preset_hq": "HQ",
  "audiolab_polish": "Rifinitura",
  "audiolab_polish_desc": "Voce calda (EQ, compressore, de-esser)",
  "audiolab_step_polish": "Rifinitura"
}
```

## 7. Tests

### `tests/test_preprocess.py` — new tests

1. **`test_apply_polish_creates_output()`** — verifies ffmpeg polish produces valid WAV
2. **`test_apply_polish_48k_mono()`** — verifies output is 48kHz mono
3. **`test_demucs_chunked_short_file()`** — file < 5min processes as single chunk
4. **`test_pipeline_new_order_denoise_before_isolation()`** — verifies denoise runs before voice isolation via mock call order
5. **`test_pipeline_loudnorm_last()`** — verifies loudnorm runs last
6. **`test_pipeline_hq_preset_enables_polish()`** — verifies polish step runs with HQ config
7. **`test_pipeline_lecture_no_polish()`** — verifies Lecture preset does not run polish

### `tests/test_main.py` — new tests

1. **`test_audiolab_process_hq_preset()`** — verifies "hq" preset is accepted
2. **`test_audiolab_process_polish_param()`** — verifies polish parameter is accepted

## 8. Files Modified

| File | Change |
|------|--------|
| `app/core/preprocess.py` | Add `_apply_polish()`, refactor `_run_demucs()` → chunked, reorder `run_pipeline()`, add `polish` to config |
| `app/main.py` | Add `polish` form param, add "hq" preset branch |
| `app/templates/index.html` | Add HQ preset button, Polish toggle |
| `app/static/app.js` | Add `alPolish` state, update `alSetPreset()`, send polish in FormData |
| `app/locales/en.json` | Add 4 new keys |
| `app/locales/it.json` | Add 4 new keys |
| `tests/test_preprocess.py` | Add 7 new tests |
| `tests/test_main.py` | Add 2 new tests |

## 9. Out of Scope

- Per-step toggles for individual polish filters
- Adjustable EQ/compressor parameters in UI
- GPU-accelerated polish (ffmpeg filters are CPU-only, but very fast)
- Chunk size configuration (hardcoded 5min + 5s overlap)
- Changes to LUFS target defaults
