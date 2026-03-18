# Dual Denoise Engine — Spec

**Date:** 2026-03-19
**Status:** Draft
**Scope:** Add ffmpeg `afftdn` as a lightweight denoise engine alongside DeepFilterNet, with user-selectable engine and installation guide UI.

---

## 1. Problem

DeepFilterNet requires Rust + Visual Studio Build Tools (~8 GB) to compile on Windows. This is an unacceptable barrier for the target audience (university students). The denoise feature is currently unusable without these prerequisites.

## 2. Solution

Add `ffmpeg afftdn` as a **zero-install** denoise engine that works out of the box. Keep DeepFilterNet as an **advanced option** for users willing to install build tools. The user explicitly chooses which engine to use via a global setting.

## 3. Architecture

### 3.1 Denoise Engine Dispatcher

```
PreprocessConfig.denoise = True
PreprocessConfig.denoise_engine = "ffmpeg" | "deepfilter"
                                      │
                          ┌───────────┴───────────┐
                          ▼                       ▼
              _apply_denoise_ffmpeg()    _apply_denoise_deepfilter()
              (ffmpeg afftdn filter)     (DeepFilterNet, unchanged)
                          │                       │
                          └───────────┬───────────┘
                                      ▼
                              48kHz mono WAV output
```

### 3.2 Engine Characteristics

| Property | ffmpeg afftdn | DeepFilterNet |
|----------|--------------|---------------|
| Install | Zero (ffmpeg already required) | Rust + VS Build Tools + pip |
| Quality (stationary noise) | Good | Excellent |
| Quality (non-stationary noise) | Poor | Excellent |
| Speed (1h file, CPU) | ~10 seconds | ~3-10 minutes |
| Default | **Yes** | No |

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
    denoise_engine: str = "ffmpeg"  # NEW: "ffmpeg" or "deepfilter"
```

#### New function: `_apply_denoise_ffmpeg()`

```python
def _apply_denoise_ffmpeg(wav_path: Path, output_path: Path) -> Path:
    """Denoise using ffmpeg's afftdn filter, optimized for speech."""
    ffmpeg = _find_ffmpeg()
    cmd = [
        ffmpeg, "-y", "-i", str(wav_path),
        "-af", "afftdn=nf=-25:tn=1",
        "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le",
        str(output_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    logger.info("Denoise (ffmpeg): %s → %s", wav_path.name, output_path.name)
    return output_path.resolve()
```

**afftdn parameters:**
- `nf=-25`: noise floor at -25 dB (good for lecture-room background noise)
- `tn=1`: enable noise tracking (adapts to changing noise levels)

#### Renamed: `apply_denoise()` → dispatcher

```python
def apply_denoise(wav_path: Path, output_path: Path, engine: str = "ffmpeg") -> Path:
    """Denoise audio using the specified engine."""
    if engine == "deepfilter":
        if not _DEEPFILTER_AVAILABLE:
            raise RuntimeError(
                "DeepFilterNet is not installed. "
                "Install it with: pip install deepfilternet"
            )
        return _apply_denoise_deepfilter(wav_path, output_path)
    else:
        return _apply_denoise_ffmpeg(wav_path, output_path)
```

Current `apply_denoise()` body moves to `_apply_denoise_deepfilter()` (private).

#### `run_pipeline()` change

```python
if config.denoise:
    engine_label = "deepfilter" if config.denoise_engine == "deepfilter" else "ffmpeg-denoise"
    _progress(0.7, engine_label, "Removing noise...")
    denoised = work_dir / "step3_denoised.wav"
    apply_denoise(current, denoised, engine=config.denoise_engine)
    current = denoised
    steps_completed.append(engine_label)
```

### 4.2 `app/main.py`

#### `/api/audiolab/process` endpoint

Add `denoise_engine` form parameter:

```python
denoise_engine: str = Form("ffmpeg"),
```

Pass to PreprocessConfig for all presets (preset determines whether denoise is on/off; engine is always from form param).

#### `/api/audiolab/deps` endpoint

No changes needed — already returns `deepfilter: bool`.

### 4.3 No changes to install endpoint

The existing `/api/audiolab/install/deepfilter` endpoint works as-is. The installation guide is purely frontend.

## 5. Frontend Changes

### 5.1 `app/static/app.js`

#### New state

```javascript
alDenoiseEngine: "ffmpeg",  // "ffmpeg" or "deepfilter" — global preference
```

#### Persistence

`alDenoiseEngine` is saved/loaded from settings (existing `/api/settings` mechanism) so it persists across sessions.

#### `alProcess()` change

Send `denoise_engine` in FormData:

```javascript
fd.append("denoise_engine", this.alDenoiseEngine);
```

#### `alMissingDep()` change

Only check for deepfilter dependency when engine is "deepfilter":

```javascript
alMissingDep() {
    if (this.alVoiceIsolation && !this.alDeps.demucs) return 'demucs';
    if (this.alDenoise && this.alDenoiseEngine === 'deepfilter' && !this.alDeps.deepfilter) return 'deepfilter';
    return null;
},
```

#### New method: `alShowInstallGuide()`

Opens a modal with DeepFilterNet installation instructions.

### 5.2 `app/templates/index.html`

#### Engine selector (below denoise toggle)

Visible only when denoise is enabled. Two pill buttons:

```
[Denoise ☑]
  [ ffmpeg (leggero) ]  [ DeepFilterNet (avanzato) 🔒 ]  How to install ↗
```

- `ffmpeg` pill: always selectable, default active
- `DeepFilterNet` pill: selectable only if `alDeps.deepfilter` is true; if not installed, clicking opens install guide
- Lock icon 🔒 shown only when DeepFilterNet is not installed
- "How to install ↗" link: visible only when DeepFilterNet is not installed, opens install guide modal

#### Install guide modal

A simple modal with step-by-step instructions:

```
┌─────────────────────────────────────────────────┐
│  Install DeepFilterNet                      [×] │
│                                                 │
│  DeepFilterNet provides state-of-the-art noise  │
│  reduction using deep learning. It requires     │
│  build tools to compile.                        │
│                                                 │
│  Step 1: Install Visual Studio Build Tools      │
│  Download from visualstudio.microsoft.com       │
│  Select "Desktop development with C++"          │
│                                                 │
│  Step 2: Install Rust                           │
│  Visit rustup.rs and run the installer          │
│                                                 │
│  Step 3: Restart this app                       │
│  Close and reopen Transcriber                   │
│                                                 │
│  Step 4: Click "Install" below                  │
│  [ Install DeepFilterNet ]                      │
│                                                 │
│  ⚠ Requires ~8 GB disk space total              │
└─────────────────────────────────────────────────┘
```

The "Install DeepFilterNet" button in the modal triggers the existing `/api/audiolab/install/deepfilter` endpoint and shows the existing install progress modal.

## 6. Locale Strings

### New keys (both en.json and it.json)

```json
{
  "audiolab_denoise_engine_ffmpeg": "ffmpeg (lightweight)",
  "audiolab_denoise_engine_deepfilter": "DeepFilterNet (advanced)",
  "audiolab_denoise_howto": "How to install",
  "audiolab_deepfilter_guide_title": "Install DeepFilterNet",
  "audiolab_deepfilter_guide_intro": "DeepFilterNet provides state-of-the-art noise reduction using deep learning. It requires build tools to compile.",
  "audiolab_deepfilter_guide_step1_title": "Step 1: Install Visual Studio Build Tools",
  "audiolab_deepfilter_guide_step1_body": "Download from visualstudio.microsoft.com and select \"Desktop development with C++\"",
  "audiolab_deepfilter_guide_step2_title": "Step 2: Install Rust",
  "audiolab_deepfilter_guide_step2_body": "Visit rustup.rs and run the installer",
  "audiolab_deepfilter_guide_step3_title": "Step 3: Restart this app",
  "audiolab_deepfilter_guide_step3_body": "Close and reopen Transcriber",
  "audiolab_deepfilter_guide_step4_title": "Step 4: Install the package",
  "audiolab_deepfilter_guide_install_btn": "Install DeepFilterNet",
  "audiolab_deepfilter_guide_warning": "Requires ~8 GB disk space total"
}
```

Italian translations follow the same pattern with localized text.

## 7. Tests

### New tests in `tests/test_preprocess.py`

1. **`test_apply_denoise_ffmpeg_creates_output()`** — verifies ffmpeg denoise produces valid WAV
2. **`test_apply_denoise_ffmpeg_48k()`** — verifies output is 48kHz mono
3. **`test_apply_denoise_dispatcher_ffmpeg()`** — calls `apply_denoise(engine="ffmpeg")`, verifies ffmpeg path taken
4. **`test_apply_denoise_dispatcher_deepfilter()`** — calls `apply_denoise(engine="deepfilter")` with mocked DeepFilterNet
5. **`test_apply_denoise_dispatcher_deepfilter_unavailable()`** — raises RuntimeError when DeepFilterNet not installed and engine="deepfilter"

### Existing tests

Existing `TestApplyDenoise` tests are updated to use `engine="deepfilter"` explicitly.

### New test in `tests/test_main.py`

1. **`test_audiolab_process_with_denoise_engine()`** — verifies `denoise_engine` param is accepted

## 8. Files Modified

| File | Change |
|------|--------|
| `app/core/preprocess.py` | Add `_apply_denoise_ffmpeg()`, refactor `apply_denoise()` as dispatcher, add `denoise_engine` to `PreprocessConfig` |
| `app/main.py` | Add `denoise_engine` form param to `/api/audiolab/process` |
| `app/templates/index.html` | Add engine selector pills, install guide modal, "How to install" link |
| `app/static/app.js` | Add `alDenoiseEngine` state, `alShowInstallGuide()`, persist engine preference |
| `app/locales/en.json` | Add ~13 new denoise engine/guide strings |
| `app/locales/it.json` | Add ~13 new denoise engine/guide strings (Italian) |
| `tests/test_preprocess.py` | Add 5 new tests, update existing denoise tests |
| `tests/test_main.py` | Add 1 new test |

## 9. Out of Scope

- Adjustable `afftdn` parameters in UI (hardcoded for speech)
- Auto-detection of which engine to use
- Other denoise engines (noisereduce, etc.)
- GPU acceleration for DeepFilterNet
