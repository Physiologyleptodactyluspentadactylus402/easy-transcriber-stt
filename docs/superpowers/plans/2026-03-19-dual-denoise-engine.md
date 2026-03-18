# Dual Denoise Engine Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add ffmpeg `afftdn` as a zero-install denoise engine alongside DeepFilterNet, with user-selectable engine choice and installation guide UI.

**Architecture:** The existing `apply_denoise()` becomes a dispatcher that routes to `_apply_denoise_ffmpeg()` (new, uses ffmpeg afftdn filter) or `_apply_denoise_deepfilter()` (renamed from current implementation). The denoise engine is a global user preference stored in settings, orthogonal to presets. The frontend adds engine selector pills below the denoise toggle and a "How to install" guide modal for DeepFilterNet.

**Tech Stack:** Python/FastAPI backend, Alpine.js/Tailwind CSS frontend, ffmpeg CLI, DeepFilterNet (optional)

**Spec:** `docs/superpowers/specs/2026-03-19-dual-denoise-engine.md`

---

## Task 1: Add `_apply_denoise_ffmpeg()` and refactor dispatcher

**Files:**
- Modify: `app/core/preprocess.py:449-464` (refactor `apply_denoise`)
- Modify: `app/core/preprocess.py:487-493` (add `denoise_engine` to `PreprocessConfig`)
- Modify: `app/core/preprocess.py:599-604` (update `run_pipeline` denoise step)
- Test: `tests/test_preprocess.py`

- [ ] **Step 1: Write failing tests for ffmpeg denoise**

Add to `tests/test_preprocess.py` after the existing `TestApplyDenoise` class (after line 258). Add `_apply_denoise_ffmpeg` to imports at line 5.

```python
# Add to imports (line 5):
from app.core.preprocess import (
    PreprocessConfig, PipelineResult, analyze_lufs, apply_denoise,
    apply_loudnorm, apply_voice_isolation, decode_to_wav, run_pipeline,
    _apply_denoise_ffmpeg,
)


class TestApplyDenoiseFFmpeg:
    """Tests for ffmpeg afftdn denoise engine."""

    def test_output_file_is_created(self, audio_tone_10s, tmp_path):
        output = tmp_path / "denoised.wav"
        result = _apply_denoise_ffmpeg(audio_tone_10s, output)
        assert result.exists()
        assert result == output.resolve()

    def test_output_is_48k_mono(self, audio_tone_10s, tmp_path):
        output = tmp_path / "denoised.wav"
        _apply_denoise_ffmpeg(audio_tone_10s, output)
        import soundfile as sf
        data, sr = sf.read(str(output))
        assert sr == 48000
        assert data.ndim == 1  # mono


class TestApplyDenoiseDispatcher:
    """Tests for the apply_denoise dispatcher."""

    def test_dispatcher_ffmpeg(self, audio_tone_10s, tmp_path):
        output = tmp_path / "denoised.wav"
        result = apply_denoise(audio_tone_10s, output, engine="ffmpeg")
        assert result.exists()

    @patch("app.core.preprocess._DEEPFILTER_AVAILABLE", True)
    @patch("app.core.preprocess._run_deepfilter")
    def test_dispatcher_deepfilter(self, mock_df, audio_tone_10s, tmp_path):
        import numpy as np
        mock_df.return_value = np.random.randn(48000 * 10).astype(np.float32)
        output = tmp_path / "denoised.wav"
        result = apply_denoise(audio_tone_10s, output, engine="deepfilter")
        assert result.exists()
        mock_df.assert_called_once()

    def test_dispatcher_deepfilter_unavailable(self, audio_tone_10s, tmp_path):
        output = tmp_path / "denoised.wav"
        with pytest.raises(RuntimeError, match="DeepFilterNet is not installed"):
            apply_denoise(audio_tone_10s, output, engine="deepfilter")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_preprocess.py::TestApplyDenoiseFFmpeg -v && python -m pytest tests/test_preprocess.py::TestApplyDenoiseDispatcher -v`
Expected: ImportError for `_apply_denoise_ffmpeg`, test failures

- [ ] **Step 3: Implement ffmpeg denoise and refactor dispatcher**

In `app/core/preprocess.py`:

**A) Replace the entire `PreprocessConfig` dataclass (lines 487-493):**

```python
@dataclass
class PreprocessConfig:
    loudnorm: bool = True
    loudnorm_target: float = -16.0
    voice_isolation: bool = True
    denoise: bool = True
    denoise_engine: str = "ffmpeg"
```

**B) Add `_apply_denoise_ffmpeg()` before the current `apply_denoise()` (before line 449):**

```python
def _apply_denoise_ffmpeg(wav_path: Path, output_path: Path) -> Path:
    """Denoise using ffmpeg's afftdn filter, optimized for speech."""
    ffmpeg = _require_ffmpeg()
    cmd = [
        ffmpeg, "-y", "-i", str(wav_path),
        "-af", "afftdn=nf=-25:tn=1",
        "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le",
        str(output_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        stderr_text = result.stderr.decode(errors="replace")
        logger.error("ffmpeg afftdn failed: %s", stderr_text)
        raise RuntimeError(f"ffmpeg denoise failed: {stderr_text}")
    logger.info("Denoise (ffmpeg): %s → %s", wav_path.name, output_path.name)
    return output_path.resolve()
```

**C) Rename current `apply_denoise()` body to `_apply_denoise_deepfilter()` and create dispatcher:**

```python
def _apply_denoise_deepfilter(wav_path: Path, output_path: Path) -> Path:
    """Denoise using DeepFilterNet (state-of-the-art neural noise reduction)."""
    enhanced = _run_deepfilter(wav_path)
    sf.write(str(output_path), enhanced, 48000, subtype="PCM_16")
    logger.info("Denoise (deepfilter): %s → %s", wav_path.name, output_path.name)
    return output_path.resolve()


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

**D) Update `run_pipeline()` denoise step (line 599-604):**

```python
    if config.denoise:
        engine_label = "deepfilter" if config.denoise_engine == "deepfilter" else "ffmpeg_denoise"
        _progress(0.7, engine_label, "Removing noise...")
        denoised = work_dir / "step3_denoised.wav"
        apply_denoise(current, denoised, engine=config.denoise_engine)
        current = denoised
        steps_completed.append(engine_label)
```

- [ ] **Step 4: Update existing TestApplyDenoise tests (BEFORE running tests)**

The existing `TestApplyDenoise` tests (lines 219-258) test the DeepFilterNet path. Since the default engine is now `"ffmpeg"`, these tests will break without updating. Update them to call `apply_denoise(..., engine="deepfilter")` explicitly. The `test_raises_when_deepfilter_unavailable` test should use `apply_denoise(..., engine="deepfilter")`.

**Important:** This step MUST be done before running tests, because Step 3 changed the default engine from deepfilter to ffmpeg, which would cause the existing tests to route to the wrong backend.

- [ ] **Step 5: Run all tests to verify they pass**

Run: `python -m pytest tests/test_preprocess.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add app/core/preprocess.py tests/test_preprocess.py
git commit -m "feat: add ffmpeg afftdn denoise engine with dispatcher"
```

---

## Task 2: Add `denoise_engine` to settings persistence

**Files:**
- Modify: `app/settings.py:14-31` (`__init__`), `app/settings.py:33-43` (`save`)
- Modify: `app/main.py:91-103` (GET /api/settings), `app/main.py:105-116` (PATCH /api/settings)
- Modify: `app/main.py:394-427` (`/api/audiolab/process` endpoint)
- Test: `tests/test_main.py`

- [ ] **Step 1: Write failing test for denoise_engine in process endpoint**

Add to `tests/test_main.py` after the last test (after line 213):

```python
def test_audiolab_process_accepts_denoise_engine(client, tmp_path):
    """POST /api/audiolab/process accepts denoise_engine param."""
    wav = tmp_path / "test.wav"
    import numpy as np, soundfile as sf
    sf.write(str(wav), np.zeros(16000, dtype=np.float32), 16000)
    with open(wav, "rb") as f:
        resp = client.post(
            "/api/audiolab/process",
            files={"file": ("test.wav", f, "audio/wav")},
            data={"preset": "custom", "denoise": "true", "denoise_engine": "ffmpeg"},
        )
    assert resp.status_code == 200
    assert "job_id" in resp.json()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_main.py::test_audiolab_process_accepts_denoise_engine -v`
Expected: FAIL (unexpected form field or similar)

- [ ] **Step 3: Add `denoise_engine` to Settings class**

In `app/settings.py`:

**In `__init__` (around line 31), add:**
```python
self.denoise_engine: str = data.get("denoise_engine", "ffmpeg")
```

**In `save()` (around line 37), add to the data dict:**
```python
"denoise_engine": self.denoise_engine,
```

- [ ] **Step 4: Update `app/main.py` settings endpoints**

**In GET /api/settings (line 91-103), add to response dict:**
```python
"denoise_engine": settings.denoise_engine,
```

**In PATCH /api/settings (line 105-116), add to allowed set:**
```python
allowed = {
    "language", "output_dir", "chunk_size_sec",
    "default_provider", "default_model",
    "default_output_formats", "wizard_complete",
    "denoise_engine",
}
```

- [ ] **Step 5: Update `/api/audiolab/process` endpoint**

**Add `denoise_engine` form param (line 402):**
```python
denoise_engine: str = Form("ffmpeg"),
```

**Update all three preset branches (lines 415-427) to pass `denoise_engine`:**
```python
if preset == "lecture":
    config = PreprocessConfig(loudnorm=True, loudnorm_target=-16.0,
                              voice_isolation=True, denoise=True,
                              denoise_engine=denoise_engine)
elif preset == "clean":
    config = PreprocessConfig(loudnorm=True, loudnorm_target=-16.0,
                              voice_isolation=False, denoise=False,
                              denoise_engine=denoise_engine)
else:  # custom
    config = PreprocessConfig(
        loudnorm=loudnorm,
        loudnorm_target=loudnorm_target,
        voice_isolation=voice_isolation,
        denoise=denoise,
        denoise_engine=denoise_engine,
    )
```

- [ ] **Step 6: Run all tests to verify they pass**

Run: `python -m pytest tests/test_main.py -v && python -m pytest tests/test_preprocess.py -v`
Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
git add app/settings.py app/main.py tests/test_main.py
git commit -m "feat: add denoise_engine to settings and audiolab endpoint"
```

---

## Task 3: Add locale strings

**Files:**
- Modify: `app/locales/en.json:101,117`
- Modify: `app/locales/it.json:101,117`

- [ ] **Step 1: Update `app/locales/en.json`**

**Update existing key (line 101):**
```json
"audiolab_denoise": "Denoise",
```
(Remove the old `"Denoise (DeepFilterNet)"`)

**Add new keys after line 117 (`audiolab_step_resample`):**
```json
"audiolab_denoise_engine_ffmpeg": "ffmpeg (lightweight)",
"audiolab_denoise_engine_deepfilter": "DeepFilterNet (advanced)",
"audiolab_denoise_howto": "How to install",
"audiolab_step_ffmpeg_denoise": "FFmpeg Denoise",
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
"audiolab_deepfilter_guide_warning": "Requires ~8 GB disk space total",
```

- [ ] **Step 2: Update `app/locales/it.json`**

**Update existing key (line 101):**
```json
"audiolab_denoise": "Riduzione rumore",
```

**Add new keys after line 117:**
```json
"audiolab_denoise_engine_ffmpeg": "ffmpeg (leggero)",
"audiolab_denoise_engine_deepfilter": "DeepFilterNet (avanzato)",
"audiolab_denoise_howto": "Come installare",
"audiolab_step_ffmpeg_denoise": "Riduzione rumore FFmpeg",
"audiolab_deepfilter_guide_title": "Installa DeepFilterNet",
"audiolab_deepfilter_guide_intro": "DeepFilterNet offre una riduzione del rumore all'avanguardia tramite deep learning. Richiede strumenti di compilazione.",
"audiolab_deepfilter_guide_step1_title": "Passo 1: Installa Visual Studio Build Tools",
"audiolab_deepfilter_guide_step1_body": "Scarica da visualstudio.microsoft.com e seleziona \"Sviluppo di applicazioni desktop con C++\"",
"audiolab_deepfilter_guide_step2_title": "Passo 2: Installa Rust",
"audiolab_deepfilter_guide_step2_body": "Visita rustup.rs e avvia l'installer",
"audiolab_deepfilter_guide_step3_title": "Passo 3: Riavvia l'app",
"audiolab_deepfilter_guide_step3_body": "Chiudi e riapri Transcriber",
"audiolab_deepfilter_guide_step4_title": "Passo 4: Installa il pacchetto",
"audiolab_deepfilter_guide_install_btn": "Installa DeepFilterNet",
"audiolab_deepfilter_guide_warning": "Richiede ~8 GB di spazio su disco in totale",
```

- [ ] **Step 3: Verify JSON is valid**

Run: `python -c "import json; json.load(open('app/locales/en.json')); json.load(open('app/locales/it.json')); print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add app/locales/en.json app/locales/it.json
git commit -m "feat: add dual denoise engine locale strings (en + it)"
```

---

## Task 4: Frontend — engine selector and install guide modal

**Files:**
- Modify: `app/static/app.js:66,75,125-136,586-595,601-608,610-633`
- Modify: `app/templates/index.html:569-593`

- [ ] **Step 1: Add `alDenoiseEngine` state and persistence to `app/static/app.js`**

**Add state variable (after line 66 `alDenoise: true`):**
```javascript
alDenoiseEngine: "ffmpeg",
alShowGuide: false,
```

**In `loadSettings()` method (lines 125-136), after setting other values from `this.settings`, add:**
```javascript
if (this.settings.denoise_engine) this.alDenoiseEngine = this.settings.denoise_engine;
```

**Note:** `saveSettings(patch)` takes a `patch` argument and sends it directly — do NOT modify the method body. Instead, callers pass the data to save. See Step 4 below for how `alSelectDenoiseEngine` calls it.

- [ ] **Step 2: Update `alMissingDep()` (lines 601-608)**

Replace the current deepfilter check to only trigger when engine is "deepfilter":

```javascript
alMissingDep() {
    if (this.alVoiceIsolation && !this.alDeps.demucs) return 'demucs';
    if (this.alDenoise && this.alDenoiseEngine === 'deepfilter' && !this.alDeps.deepfilter) return 'deepfilter';
    return null;
},
```

- [ ] **Step 3: Update `alProcess()` FormData (lines 610-633)**

Add `denoise_engine` to the FormData being sent:
```javascript
fd.append("denoise_engine", this.alDenoiseEngine);
```

- [ ] **Step 4: Add `alSelectDenoiseEngine()` method**

Add a new method to handle engine selection with install guard:

```javascript
alSelectDenoiseEngine(engine) {
    if (engine === 'deepfilter' && !this.alDeps.deepfilter) {
        this.alShowGuide = true;
        return;
    }
    this.alDenoiseEngine = engine;
    this.saveSettings({ denoise_engine: engine });
},
alInstallFromGuide() {
    this.alShowGuide = false;
    this.alOpenInstallModal('deepfilter');
},
```

- [ ] **Step 5: Update `app/templates/index.html` — replace denoise Install badge with engine selector**

Replace the denoise toggle section (lines 582-593) with:

```html
<!-- Denoise toggle -->
<div class="flex items-center justify-between">
  <label class="flex items-center gap-2 text-sm text-gray-300">
    <input type="checkbox" x-model="alDenoise" :disabled="alPreset!=='custom'"
           class="w-4 h-4 rounded border-gray-600 bg-gray-800 text-indigo-500">
    <span x-text="t('audiolab_denoise')"></span>
  </label>
</div>
<!-- Engine selector (visible when denoise enabled) -->
<div x-show="alDenoise" class="ml-6 mt-1 flex items-center gap-2 flex-wrap">
  <button @click="alSelectDenoiseEngine('ffmpeg')"
          :class="alDenoiseEngine==='ffmpeg' ? 'bg-indigo-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'"
          class="text-xs px-3 py-1 rounded-full transition-colors cursor-pointer">
    <span x-text="t('audiolab_denoise_engine_ffmpeg')"></span>
  </button>
  <button @click="alSelectDenoiseEngine('deepfilter')"
          :class="alDenoiseEngine==='deepfilter' ? 'bg-indigo-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'"
          class="text-xs px-3 py-1 rounded-full transition-colors cursor-pointer">
    <span x-text="t('audiolab_denoise_engine_deepfilter')"></span>
    <span x-show="!alDeps.deepfilter" class="ml-1 opacity-60">&#128274;</span>
  </button>
  <a x-show="!alDeps.deepfilter" @click.prevent="alShowGuide=true"
     href="#" class="text-xs text-indigo-400 hover:text-indigo-300 transition-colors cursor-pointer">
    <span x-text="t('audiolab_denoise_howto')"></span> &#8599;
  </a>
</div>
```

- [ ] **Step 6: Add install guide modal to `app/templates/index.html`**

Add before the closing `</section>` of the Audio Lab section:

```html
<!-- DeepFilterNet install guide modal -->
<div x-show="alShowGuide" x-cloak
     class="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
     @click.self="alShowGuide=false" @keydown.escape.window="alShowGuide=false">
  <div class="bg-gray-800 rounded-xl shadow-2xl border border-gray-700 w-full max-w-md mx-4 p-6">
    <div class="flex justify-between items-center mb-4">
      <h3 class="text-lg font-semibold text-white" x-text="t('audiolab_deepfilter_guide_title')"></h3>
      <button @click="alShowGuide=false" class="text-gray-400 hover:text-white text-xl cursor-pointer">&times;</button>
    </div>
    <p class="text-sm text-gray-400 mb-4" x-text="t('audiolab_deepfilter_guide_intro')"></p>
    <ol class="space-y-3 text-sm">
      <li>
        <div class="text-white font-medium" x-text="t('audiolab_deepfilter_guide_step1_title')"></div>
        <div class="text-gray-400 mt-0.5" x-text="t('audiolab_deepfilter_guide_step1_body')"></div>
      </li>
      <li>
        <div class="text-white font-medium" x-text="t('audiolab_deepfilter_guide_step2_title')"></div>
        <div class="text-gray-400 mt-0.5" x-text="t('audiolab_deepfilter_guide_step2_body')"></div>
      </li>
      <li>
        <div class="text-white font-medium" x-text="t('audiolab_deepfilter_guide_step3_title')"></div>
        <div class="text-gray-400 mt-0.5" x-text="t('audiolab_deepfilter_guide_step3_body')"></div>
      </li>
      <li>
        <div class="text-white font-medium" x-text="t('audiolab_deepfilter_guide_step4_title')"></div>
        <button @click="alInstallFromGuide()"
                class="mt-2 w-full py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium transition-colors cursor-pointer"
                x-text="t('audiolab_deepfilter_guide_install_btn')"></button>
      </li>
    </ol>
    <p class="mt-4 text-xs text-amber-400/80">&#9888; <span x-text="t('audiolab_deepfilter_guide_warning')"></span></p>
  </div>
</div>
```

- [ ] **Step 7: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 8: Commit**

```bash
git add app/static/app.js app/templates/index.html
git commit -m "feat: add denoise engine selector UI and DeepFilterNet install guide"
```

---

## Task 5: Full integration test and cleanup

- [ ] **Step 1: Run the full test suite**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS (existing + new)

- [ ] **Step 2: Manual smoke test checklist**

Start the server with `python start.py` and verify in browser:

1. Audio Lab tab loads without errors
2. Denoise toggle shows engine selector pills when enabled
3. "ffmpeg (lightweight)" is selected by default
4. "DeepFilterNet (advanced)" shows lock icon when not installed
5. Clicking locked DeepFilterNet pill opens install guide modal
6. "How to install" link opens the same guide modal
7. Guide modal has 4 steps and Install button
8. Clicking Install in guide opens the existing install progress modal
9. Processing a file with ffmpeg engine works (denoise step completes)
10. Engine preference persists after page reload

- [ ] **Step 3: Final commit if any fixes needed**

```bash
git add -A
git commit -m "fix: address integration test findings"
```
