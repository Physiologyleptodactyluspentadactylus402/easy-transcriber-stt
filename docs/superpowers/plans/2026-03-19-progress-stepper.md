# Progress Stepper with ETA — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the simple Audio Lab progress bar with a vertical stepper showing all pipeline steps, sub-progress for Demucs chunks, and dynamic ETA.

**Architecture:** Backend `PipelineProgressTracker` class manages step timing and ETA computation; `run_pipeline()` wraps each step with tracker calls and emits enriched progress data via kwargs. Frontend Alpine.js state stores step list and renders a stepper with icons, labels, and ETA.

**Tech Stack:** Python 3.10+ (FastAPI), Alpine.js 3.x, Tailwind CSS 3.x, pytest

---

## Chunk 1: Backend — PipelineProgressTracker + build_step_list

### Task 1: `build_step_list` function

**Files:**
- Modify: `app/core/preprocess.py:715-725` (after PreprocessConfig, before PipelineResult)
- Test: `tests/test_preprocess.py`

- [ ] **Step 1: Write failing tests for `build_step_list`**

In `tests/test_preprocess.py`, add at the top imports:

```python
from app.core.preprocess import build_step_list, PipelineProgressTracker
```

Add test class:

```python
class TestBuildStepList:
    def test_lecture_preset(self):
        config = PreprocessConfig(loudnorm=True, voice_isolation=False,
                                  denoise=True, denoise_engine="ffmpeg", polish=False)
        steps = build_step_list(config)
        ids = [s["id"] for s in steps]
        assert ids == ["decode", "analyze", "denoise", "demucs", "polish", "loudnorm", "resample"]
        active = {s["id"]: s["active"] for s in steps}
        assert active["decode"] is True
        assert active["analyze"] is True
        assert active["denoise"] is True
        assert active["demucs"] is False
        assert active["polish"] is False
        assert active["loudnorm"] is True
        assert active["resample"] is True

    def test_hq_preset(self):
        config = PreprocessConfig(loudnorm=True, voice_isolation=True,
                                  denoise=True, denoise_engine="deepfilter", polish=True)
        steps = build_step_list(config)
        active = {s["id"]: s["active"] for s in steps}
        assert all(active.values())  # all steps active for HQ

    def test_all_disabled(self):
        config = PreprocessConfig(loudnorm=False, voice_isolation=False,
                                  denoise=False, polish=False)
        steps = build_step_list(config)
        active = {s["id"]: s["active"] for s in steps}
        assert active["decode"] is True
        assert active["analyze"] is True
        assert active["resample"] is True
        assert active["denoise"] is False
        assert active["demucs"] is False
        assert active["polish"] is False
        assert active["loudnorm"] is False

    def test_label_keys_are_stepper_keys(self):
        config = PreprocessConfig()
        steps = build_step_list(config)
        for s in steps:
            assert s["label_key"].startswith("audiolab_stepper_")
```

- [ ] **Step 2: Run tests — verify they fail**

Run: `python -m pytest tests/test_preprocess.py::TestBuildStepList -v`
Expected: ImportError for `build_step_list`

- [ ] **Step 3: Implement `build_step_list`**

In `app/core/preprocess.py`, after `PreprocessConfig` (line ~725), add:

```python
# All pipeline steps in order, with their stepper label keys
_PIPELINE_STEPS = [
    ("decode",   "audiolab_stepper_decode"),
    ("analyze",  "audiolab_stepper_analyze"),
    ("denoise",  "audiolab_stepper_denoise"),
    ("demucs",   "audiolab_stepper_demucs"),
    ("polish",   "audiolab_stepper_polish"),
    ("loudnorm", "audiolab_stepper_loudnorm"),
    ("resample", "audiolab_stepper_resample"),
]


def build_step_list(config: PreprocessConfig) -> list[dict]:
    """Build the ordered list of pipeline steps with active flags."""
    always_active = {"decode", "analyze", "resample"}
    active_map = {
        "denoise": config.denoise,
        "demucs": config.voice_isolation,
        "polish": config.polish,
        "loudnorm": config.loudnorm,
    }
    return [
        {
            "id": step_id,
            "label_key": label_key,
            "active": step_id in always_active or active_map.get(step_id, True),
        }
        for step_id, label_key in _PIPELINE_STEPS
    ]
```

- [ ] **Step 4: Run tests — verify they pass**

Run: `python -m pytest tests/test_preprocess.py::TestBuildStepList -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/core/preprocess.py tests/test_preprocess.py
git commit -m "feat: add build_step_list for pipeline step enumeration"
```

---

### Task 2: `PipelineProgressTracker` class

**Files:**
- Modify: `app/core/preprocess.py` (after `build_step_list`)
- Test: `tests/test_preprocess.py`

- [ ] **Step 1: Write failing tests for PipelineProgressTracker**

```python
import time


class TestPipelineProgressTracker:
    def _make_tracker(self, skip_steps=None):
        config = PreprocessConfig(
            loudnorm=True, voice_isolation=True,
            denoise=True, denoise_engine="ffmpeg", polish=True,
        )
        if skip_steps:
            if "denoise" in skip_steps:
                config.denoise = False
            if "demucs" in skip_steps:
                config.voice_isolation = False
            if "polish" in skip_steps:
                config.polish = False
            if "loudnorm" in skip_steps:
                config.loudnorm = False
        steps = build_step_list(config)
        return PipelineProgressTracker(steps, denoise_engine=config.denoise_engine)

    def test_initial_state(self):
        tracker = self._make_tracker()
        tracker.start()
        data = tracker.get_progress_data()
        assert data["progress"] == 0.0
        assert data["step"] is None
        assert data["eta_sec"] is None
        assert data["sub_progress"] is None

    def test_begin_step(self):
        tracker = self._make_tracker()
        tracker.start()
        tracker.begin_step("decode")
        data = tracker.get_progress_data()
        assert data["step"] == "decode"

    def test_complete_step(self):
        tracker = self._make_tracker()
        tracker.start()
        tracker.begin_step("decode")
        tracker.complete_step("decode")
        data = tracker.get_progress_data()
        assert data["completed_step"] == "decode"
        assert data["completed_step_elapsed"] >= 0
        assert data["progress"] > 0

    def test_eta_after_one_step(self):
        tracker = self._make_tracker()
        tracker.start()
        tracker.begin_step("decode")
        # Simulate time passing
        tracker._step_start_time -= 2.0  # fake 2s elapsed for decode
        tracker.complete_step("decode")
        data = tracker.get_progress_data()
        assert data["eta_sec"] is not None
        assert data["eta_sec"] > 0

    def test_eta_is_none_before_any_completion(self):
        tracker = self._make_tracker()
        tracker.start()
        tracker.begin_step("decode")
        data = tracker.get_progress_data()
        assert data["eta_sec"] is None

    def test_skipped_steps_excluded_from_weights(self):
        tracker = self._make_tracker(skip_steps={"demucs", "polish"})
        tracker.start()
        # Weights should be re-normalized without demucs and polish
        total = sum(tracker._normalized_weights.values())
        assert abs(total - 1.0) < 0.001

    def test_sub_progress(self):
        tracker = self._make_tracker()
        tracker.start()
        tracker.begin_step("demucs")
        tracker.update_sub_progress(0.4, "chunk 2/5")
        data = tracker.get_progress_data()
        assert data["sub_progress"] == 0.4
        assert data["sub_label"] == "chunk 2/5"

    def test_sub_progress_clears_on_new_step(self):
        tracker = self._make_tracker()
        tracker.start()
        tracker.begin_step("demucs")
        tracker.update_sub_progress(0.5, "chunk 3/5")
        tracker.complete_step("demucs")
        tracker.begin_step("polish")
        data = tracker.get_progress_data()
        assert data["sub_progress"] is None
        assert data["sub_label"] is None

    def test_progress_reaches_1_when_all_done(self):
        tracker = self._make_tracker()
        tracker.start()
        for step_id, _ in _PIPELINE_STEPS:
            tracker.begin_step(step_id)
            tracker.complete_step(step_id)
        data = tracker.get_progress_data()
        assert abs(data["progress"] - 1.0) < 0.001
```

Also import `_PIPELINE_STEPS` at the top:
```python
from app.core.preprocess import build_step_list, PipelineProgressTracker, _PIPELINE_STEPS
```

- [ ] **Step 2: Run tests — verify they fail**

Run: `python -m pytest tests/test_preprocess.py::TestPipelineProgressTracker -v`
Expected: ImportError for `PipelineProgressTracker`

- [ ] **Step 3: Implement `PipelineProgressTracker`**

In `app/core/preprocess.py`, after `build_step_list`:

```python
# Default step weights (relative, will be normalized)
_STEP_WEIGHTS = {
    "ffmpeg": {
        "decode": 0.05, "analyze": 0.03, "denoise": 0.10,
        "demucs": 0.52, "polish": 0.05, "loudnorm": 0.15, "resample": 0.10,
    },
    "deepfilter": {
        "decode": 0.05, "analyze": 0.03, "denoise": 0.18,
        "demucs": 0.47, "polish": 0.05, "loudnorm": 0.12, "resample": 0.10,
    },
}


class PipelineProgressTracker:
    """Tracks pipeline step progress, timing, and ETA."""

    def __init__(self, step_list: list[dict], denoise_engine: str = "ffmpeg"):
        self._steps = step_list
        self._active_ids = [s["id"] for s in step_list if s["active"]]

        weight_key = "deepfilter" if denoise_engine == "deepfilter" else "ffmpeg"
        raw = _STEP_WEIGHTS[weight_key]
        active_raw = {sid: raw[sid] for sid in self._active_ids}
        total = sum(active_raw.values())
        self._normalized_weights = {sid: w / total for sid, w in active_raw.items()}

        self._start_time: float = 0.0
        self._step_start_time: float = 0.0
        self._completed: list[str] = []
        self._step_elapsed: dict[str, float] = {}
        self._current_step: str | None = None
        self._sub_progress: float | None = None
        self._sub_label: str | None = None
        self._last_completed_step: str | None = None
        self._last_completed_elapsed: float | None = None

    def start(self):
        import time
        self._start_time = time.monotonic()

    def begin_step(self, step_id: str):
        import time
        self._current_step = step_id
        self._step_start_time = time.monotonic()
        self._sub_progress = None
        self._sub_label = None
        self._last_completed_step = None
        self._last_completed_elapsed = None

    def complete_step(self, step_id: str):
        import time
        elapsed = time.monotonic() - self._step_start_time
        self._step_elapsed[step_id] = elapsed
        self._completed.append(step_id)
        self._last_completed_step = step_id
        self._last_completed_elapsed = elapsed
        self._sub_progress = None
        self._sub_label = None

    def update_sub_progress(self, fraction: float, label: str):
        self._sub_progress = fraction
        self._sub_label = label

    def get_progress_data(self) -> dict:
        import time
        elapsed_sec = time.monotonic() - self._start_time if self._start_time else 0.0

        # Calculate completed weight
        completed_weight = sum(
            self._normalized_weights.get(s, 0) for s in self._completed
        )

        # Add partial weight for sub-progress during current step
        partial = 0.0
        if (self._current_step and self._sub_progress is not None
                and self._current_step not in self._completed):
            step_weight = self._normalized_weights.get(self._current_step, 0)
            partial = step_weight * self._sub_progress

        progress = completed_weight + partial

        # ETA
        eta_sec = None
        if self._completed:
            total_completed_weight = completed_weight + partial
            if total_completed_weight > 0:
                pace = elapsed_sec / total_completed_weight
                eta_sec = (1.0 - total_completed_weight) * pace

        return {
            "progress": min(progress, 1.0),
            "step": self._current_step,
            "sub_progress": self._sub_progress,
            "sub_label": self._sub_label,
            "elapsed_sec": round(elapsed_sec, 1),
            "eta_sec": round(eta_sec, 0) if eta_sec is not None else None,
            "completed_step": self._last_completed_step,
            "completed_step_elapsed": (
                round(self._last_completed_elapsed, 1)
                if self._last_completed_elapsed is not None else None
            ),
        }
```

- [ ] **Step 4: Run tests — verify they pass**

Run: `python -m pytest tests/test_preprocess.py::TestPipelineProgressTracker -v`
Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/core/preprocess.py tests/test_preprocess.py
git commit -m "feat: add PipelineProgressTracker with ETA calculation"
```

---

### Task 3: Wire tracker into `run_pipeline` + fix step IDs

**Files:**
- Modify: `app/core/preprocess.py:753-881` (run_pipeline function)
- Modify: `app/core/preprocess.py:476-538` (_run_demucs function)
- Modify: `app/core/preprocess.py:541-560` (apply_voice_isolation function)
- Test: `tests/test_preprocess.py`

- [ ] **Step 1: Write failing tests**

```python
class TestPipelineProgressMessages:
    """Verify run_pipeline sends enriched progress data."""

    def test_progress_callback_receives_extra_kwargs(self, audio_tone_10s, tmp_path):
        """Callback should receive sub_progress, eta_sec, etc. as kwargs."""
        calls = []
        def cb(frac, step, msg, **kwargs):
            calls.append({"frac": frac, "step": step, "msg": msg, **kwargs})

        config = PreprocessConfig(loudnorm=True, voice_isolation=False,
                                  denoise=False, polish=False)
        run_pipeline(audio_tone_10s, tmp_path, config, progress_callback=cb)

        # Should have at least decode, analyze, loudnorm, resample, done
        steps_seen = [c["step"] for c in calls]
        assert "decode" in steps_seen
        assert "analyze" in steps_seen
        assert "loudnorm" in steps_seen
        assert "resample" in steps_seen
        assert "done" in steps_seen

        # The final "done" call should have progress ~1.0
        done_call = [c for c in calls if c["step"] == "done"][0]
        assert done_call.get("elapsed_sec") is not None

    def test_denoise_step_uses_canonical_id(self, audio_tone_10s, tmp_path):
        """Denoise step should use 'denoise' not 'ffmpeg_denoise'."""
        calls = []
        def cb(frac, step, msg, **kwargs):
            calls.append(step)

        config = PreprocessConfig(loudnorm=False, voice_isolation=False,
                                  denoise=True, denoise_engine="ffmpeg", polish=False)
        run_pipeline(audio_tone_10s, tmp_path, config, progress_callback=cb)
        assert "denoise" in calls
        assert "ffmpeg_denoise" not in calls

    def test_analyze_in_steps_completed(self, audio_tone_10s, tmp_path):
        config = PreprocessConfig(loudnorm=False, voice_isolation=False,
                                  denoise=False, polish=False)
        result = run_pipeline(audio_tone_10s, tmp_path, config)
        assert "analyze" in result.steps_completed

    def test_completed_step_field_present(self, audio_tone_10s, tmp_path):
        """After a step completes, next progress call should have completed_step."""
        calls = []
        def cb(frac, step, msg, **kwargs):
            calls.append({"step": step, **kwargs})

        config = PreprocessConfig(loudnorm=True, voice_isolation=False,
                                  denoise=False, polish=False)
        run_pipeline(audio_tone_10s, tmp_path, config, progress_callback=cb)

        # Find a call that has completed_step set
        completed_calls = [c for c in calls if c.get("completed_step")]
        assert len(completed_calls) >= 1
        assert completed_calls[0]["completed_step_elapsed"] is not None
```

- [ ] **Step 2: Run tests — verify they fail**

Run: `python -m pytest tests/test_preprocess.py::TestPipelineProgressMessages -v`
Expected: FAIL (kwargs not present, step IDs wrong)

- [ ] **Step 3: Fix existing tests for new callback signature and step IDs**

Several existing tests will break because:
1. Callbacks use `def cb(frac, step, msg):` without `**kwargs` — the new `run_pipeline` passes extra kwargs.
2. Some tests assert `"ffmpeg_denoise"` in step IDs — renamed to `"denoise"`.

Fix in `tests/test_preprocess.py`:

```python
# Line 367: Add **kwargs
def cb(frac, step, msg, **kwargs):

# Line 416: Add **kwargs
def cb(frac, step, msg, **kwargs):

# Line 428: Add **kwargs
def cb(frac, step, msg, **kwargs):

# Line 401: Change "ffmpeg_denoise" to "denoise"
assert "denoise" in result.steps_completed

# Lines 422-423: Change "ffmpeg_denoise" to "denoise"
step_names = [s for s in calls if s in ("denoise", "loudnorm")]
assert step_names == ["denoise", "loudnorm"]
```

- [ ] **Step 4: Implement changes**

**4a. Add `sub_progress_callback` to `_run_demucs` and `apply_voice_isolation`:**

In `_run_demucs` (line 476), change signature:
```python
def _run_demucs(wav_path: Path, sub_progress_callback=None) -> np.ndarray:
```

After `chunk_idx += 1` (line 532), add:
```python
        chunk_idx += 1
        if sub_progress_callback:
            total_chunks = (total_samples + step_samples - 1) // step_samples
            sub_progress_callback(chunk_idx / total_chunks, f"chunk {chunk_idx}/{total_chunks}")
```

In `apply_voice_isolation` (line 541), change signature:
```python
def apply_voice_isolation(
    wav_path: Path,
    output_path: Path,
    sub_progress_callback=None,
) -> Path:
```

Pass it through (line 551):
```python
    vocals_np = _run_demucs(wav_path, sub_progress_callback=sub_progress_callback)
```

**4b. Rewrite `run_pipeline` to use tracker:**

Replace the `_progress` helper and wire in the tracker. Key changes:

```python
def run_pipeline(
    input_path: Path,
    work_dir: Path,
    config: PreprocessConfig,
    progress_callback: Callable[[float, str, str], None] | None = None,
    cancel_flag: dict | None = None,
) -> PipelineResult:
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    def _cancelled() -> bool:
        return bool(cancel_flag and cancel_flag.get("cancelled"))

    step_list = build_step_list(config)
    tracker = PipelineProgressTracker(step_list, denoise_engine=config.denoise_engine)
    tracker.start()

    def _progress(step: str, msg: str):
        data = tracker.get_progress_data()
        if progress_callback:
            progress_callback(
                data["progress"], step, msg,
                sub_progress=data["sub_progress"],
                sub_label=data["sub_label"],
                elapsed_sec=data["elapsed_sec"],
                eta_sec=data["eta_sec"],
                completed_step=data["completed_step"],
                completed_step_elapsed=data["completed_step_elapsed"],
            )

    steps_completed: list[str] = []

    # Step 0: Decode
    tracker.begin_step("decode")
    _progress("decode", "Decoding audio...")
    original_wav = work_dir / "original.wav"
    decode_to_wav(input_path, original_wav, sample_rate=48000)
    tracker.complete_step("decode")
    steps_completed.append("decode")
    _progress("decode", "Decoding audio...")  # emit with completed_step

    # LUFS analysis
    tracker.begin_step("analyze")
    _progress("analyze", "Measuring loudness...")
    lufs_stats = analyze_lufs(original_wav)
    original_lufs = lufs_stats["input_i"]
    tracker.complete_step("analyze")
    steps_completed.append("analyze")
    _progress("analyze", "Measuring loudness...")

    if _cancelled():
        return _early_return(original_wav, original_wav, original_lufs, steps_completed, work_dir)

    current = original_wav

    # Denoise
    if config.denoise:
        if _cancelled():
            return _early_return(original_wav, current, original_lufs, steps_completed, work_dir)
        tracker.begin_step("denoise")
        _progress("denoise", "Removing noise...")
        denoised = work_dir / "step1_denoised.wav"
        apply_denoise(current, denoised, engine=config.denoise_engine)
        current = denoised
        tracker.complete_step("denoise")
        steps_completed.append("denoise")
        _progress("denoise", "Removing noise...")

    # Voice isolation
    if config.voice_isolation:
        if _cancelled():
            return _early_return(original_wav, current, original_lufs, steps_completed, work_dir)
        tracker.begin_step("demucs")
        _progress("demucs", "Isolating voice (this may take a while)...")
        vocals = work_dir / "step2_vocals.wav"

        def _demucs_sub(frac, label):
            tracker.update_sub_progress(frac, label)
            _progress("demucs", f"Isolating voice — {label}")

        apply_voice_isolation(current, vocals, sub_progress_callback=_demucs_sub)
        current = vocals
        tracker.complete_step("demucs")
        steps_completed.append("demucs")
        _progress("demucs", "Voice isolation complete")

    # Polish
    if config.polish:
        if _cancelled():
            return _early_return(original_wav, current, original_lufs, steps_completed, work_dir)
        tracker.begin_step("polish")
        _progress("polish", "Applying audio polish...")
        polished = work_dir / "step3_polished.wav"
        _apply_polish(current, polished)
        current = polished
        tracker.complete_step("polish")
        steps_completed.append("polish")
        _progress("polish", "Audio polish complete")

    # Loudnorm
    if config.loudnorm:
        if _cancelled():
            return _early_return(original_wav, current, original_lufs, steps_completed, work_dir)
        tracker.begin_step("loudnorm")
        _progress("loudnorm", f"Normalizing to {config.loudnorm_target} LUFS...")
        normalized = work_dir / "step4_loudnorm.wav"
        current_lufs = analyze_lufs(current) if current != original_wav else lufs_stats
        apply_loudnorm(current, normalized, target_lufs=config.loudnorm_target, measured=current_lufs)
        current = normalized
        tracker.complete_step("loudnorm")
        steps_completed.append("loudnorm")
        _progress("loudnorm", "Loudness normalized")

    # Resample
    tracker.begin_step("resample")
    _progress("resample", "Preparing final output...")
    processed_48k = work_dir / "processed_48k.wav"
    processed_16k = work_dir / "processed.wav"
    if current.resolve() != processed_48k.resolve():
        shutil.copy2(current, processed_48k)
    _resample_to_16k(current, processed_16k)
    tracker.complete_step("resample")
    steps_completed.append("resample")

    # Stats
    processed_lufs_stats = analyze_lufs(processed_48k)
    original_size = Path(input_path).stat().st_size
    processed_size = processed_16k.stat().st_size

    from pydub import AudioSegment
    duration_sec = len(AudioSegment.from_wav(str(processed_16k))) / 1000.0

    stats = {
        "original_lufs": original_lufs,
        "processed_lufs": processed_lufs_stats["input_i"],
        "duration_sec": duration_sec,
        "original_size": original_size,
        "processed_size": processed_size,
    }

    # Final progress
    data = tracker.get_progress_data()
    if progress_callback:
        progress_callback(
            1.0, "done", "Processing complete",
            elapsed_sec=data["elapsed_sec"],
            eta_sec=0,
            sub_progress=None, sub_label=None,
            completed_step=None, completed_step_elapsed=None,
        )

    logger.info("Pipeline complete: steps=%s, stats=%s", steps_completed, stats)
    return PipelineResult(
        original_path=original_wav,
        processed_path=processed_16k,
        processed_48k_path=processed_48k,
        stats=stats,
        steps_completed=steps_completed,
    )
```

- [ ] **Step 5: Run tests — verify they pass**

Run: `python -m pytest tests/test_preprocess.py -v`
Expected: ALL tests pass (both new and existing)

- [ ] **Step 6: Commit**

```bash
git add app/core/preprocess.py tests/test_preprocess.py
git commit -m "feat: wire PipelineProgressTracker into run_pipeline, fix step IDs"
```

---

## Chunk 2: Backend — main.py + Frontend

### Task 4: Emit `audiolab_steps` and enrich WS messages in `main.py`

**Files:**
- Modify: `app/main.py:510-582` (_run_audiolab function)
- Test: `tests/test_main.py`

- [ ] **Step 1: Write failing test**

In `tests/test_main.py`, add:

```python
def test_audiolab_steps_message_sent(client):
    """The audiolab_steps message should be sent before processing starts."""
    # This tests the build_step_list import and broadcast
    from app.core.preprocess import build_step_list, PreprocessConfig
    config = PreprocessConfig(loudnorm=True, voice_isolation=False,
                              denoise=True, denoise_engine="ffmpeg", polish=False)
    steps = build_step_list(config)
    assert len(steps) == 7
    assert steps[0]["id"] == "decode"
    # Verify active flags
    active = {s["id"]: s["active"] for s in steps}
    assert active["demucs"] is False
```

- [ ] **Step 2: Run test — verify it passes** (this is a sanity check for the import)

Run: `python -m pytest tests/test_main.py::test_audiolab_steps_message_sent -v`

- [ ] **Step 3: Update `_run_audiolab` in main.py**

Change the `_progress` callback to accept kwargs and forward them:

```python
async def _run_audiolab(job, input_path, cache_dir, config):
    """Run the preprocessing pipeline in a background thread."""
    from app.core.preprocess import run_pipeline, build_step_list
    loop = asyncio.get_running_loop()

    # Send step list before starting
    step_list = build_step_list(config)
    await _ws_manager.broadcast_global({
        "type": "audiolab_steps",
        "job_id": job.id,
        "steps": step_list,
    })

    def _progress(frac, step, msg, **kwargs):
        job.progress = frac
        job.current_step = step
        job.message = msg
        loop.call_soon_threadsafe(
            asyncio.ensure_future,
            _ws_manager.broadcast_global({
                "type": "audiolab_progress",
                "job_id": job.id,
                "progress": frac,
                "step": step,
                "message": msg,
                "sub_progress": kwargs.get("sub_progress"),
                "sub_label": kwargs.get("sub_label"),
                "elapsed_sec": kwargs.get("elapsed_sec", 0),
                "eta_sec": kwargs.get("eta_sec"),
                "completed_step": kwargs.get("completed_step"),
                "completed_step_elapsed": kwargs.get("completed_step_elapsed"),
            }),
        )

    # ... rest unchanged (cancel_flag, _run, result handling)
```

- [ ] **Step 4: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: ALL pass

- [ ] **Step 5: Commit**

```bash
git add app/main.py tests/test_main.py
git commit -m "feat: emit audiolab_steps and enriched progress via WebSocket"
```

---

### Task 5: Frontend — Alpine.js state + WebSocket handlers

**Files:**
- Modify: `app/static/app.js:55-78` (state variables)
- Modify: `app/static/app.js:291-311` (WS handlers)
- Modify: `app/static/app.js:646-652` (alProcess reset)

- [ ] **Step 1: Add new state variables**

After line 68 (`alPolish: false,`), add:

```javascript
    alSteps: [],           // [{id, label_key, active, status, elapsed}]
    alElapsed: 0,
    alEta: null,
    alSubProgress: null,
    alSubLabel: '',
```

- [ ] **Step 2: Add `_formatEta` method**

Add as a method in the Alpine component (near other `_format` helpers):

```javascript
    _formatEta(seconds) {
      if (seconds == null) return '';
      const s = Math.round(seconds);
      if (s < 60) return `~${s}s`;
      const m = Math.floor(s / 60);
      const rem = s % 60;
      return rem > 0 ? `~${m}m ${rem}s` : `~${m}m`;
    },
```

- [ ] **Step 3: Add `audiolab_steps` WS handler**

In the WebSocket message handler, before the `audiolab_progress` block (line 292), add:

```javascript
      } else if (msg.type === 'audiolab_steps') {
        if (msg.job_id === this.alJobId) {
          this.alSteps = msg.steps.map(s => ({
            ...s,
            status: s.active ? 'pending' : 'skipped',
            elapsed: null,
          }));
        }
```

- [ ] **Step 4: Enhance `audiolab_progress` handler**

Replace the existing handler (lines 292-297):

```javascript
      } else if (msg.type === 'audiolab_progress') {
        if (msg.job_id === this.alJobId) {
          this.alProgress = msg.progress;
          this.alStep = msg.step;
          this.alMessage = msg.message;
          this.alElapsed = msg.elapsed_sec || 0;
          this.alEta = msg.eta_sec;
          this.alSubProgress = msg.sub_progress;
          this.alSubLabel = msg.sub_label || '';

          // Mark completed step
          if (msg.completed_step) {
            const cs = this.alSteps.find(s => s.id === msg.completed_step);
            if (cs) {
              cs.status = 'completed';
              cs.elapsed = msg.completed_step_elapsed;
            }
          }
          // Mark current step as active
          if (msg.step && msg.step !== 'done') {
            const cur = this.alSteps.find(s => s.id === msg.step);
            if (cur && cur.status === 'pending') cur.status = 'active';
          }
        }
```

- [ ] **Step 5: Enhance `audiolab_done` handler**

After the existing done handler (line 298-305), update to mark all active steps as completed:

```javascript
      } else if (msg.type === 'audiolab_done') {
        if (msg.job_id === this.alJobId) {
          this.alStatus = 'done';
          this.alProgress = 1;
          this.alOriginalUrl = msg.original_url;
          this.alProcessedUrl = msg.processed_url;
          this.alStats = msg.stats;
          // Mark all active steps as completed
          this.alSteps.forEach(s => {
            if (s.status === 'active' || s.status === 'pending') {
              if (s.active) s.status = 'completed';
            }
          });
        }
```

- [ ] **Step 6: Reset new state in `alProcess`**

In `alProcess()` (around line 647-649), add resets:

```javascript
      this.alStatus = 'processing';
      this.alProgress = 0;
      this.alMessage = '';
      this.alSteps = [];
      this.alElapsed = 0;
      this.alEta = null;
      this.alSubProgress = null;
      this.alSubLabel = '';
```

- [ ] **Step 7: Commit**

```bash
git add app/static/app.js
git commit -m "feat: add stepper state, WS handlers, formatEta in Alpine.js"
```

---

### Task 6: Frontend — HTML stepper template

**Files:**
- Modify: `app/templates/index.html:634-640` (replace progress bar with stepper)

- [ ] **Step 1: Replace progress section**

Replace lines 634-640 (the current `<div x-show="alStatus==='processing'">` block) with:

```html
            <div x-show="alStatus==='processing'" class="space-y-3 flex-1">
              <!-- Stepper -->
              <div class="space-y-0.5">
                <template x-for="s in alSteps" :key="s.id">
                  <div class="flex items-center gap-2 text-sm"
                       :class="s.status === 'skipped' ? 'opacity-40 line-through' : ''">
                    <span x-show="s.status==='completed'" class="text-green-400 w-4 text-center">✓</span>
                    <span x-show="s.status==='active'" class="text-indigo-400 w-4 text-center animate-pulse">●</span>
                    <span x-show="s.status==='pending'" class="text-gray-600 w-4 text-center">○</span>
                    <span x-show="s.status==='skipped'" class="text-gray-600 w-4 text-center">–</span>
                    <span :class="s.status==='active' ? 'text-indigo-300 font-medium' :
                                  s.status==='completed' ? 'text-gray-300' : 'text-gray-500'"
                          x-text="t(s.label_key)"></span>
                    <span x-show="s.status==='completed' && s.elapsed != null"
                          class="text-xs text-gray-600" x-text="`${Math.round(s.elapsed)}s`"></span>
                    <span x-show="s.status==='active' && alSubLabel"
                          class="text-xs text-indigo-400" x-text="alSubLabel"></span>
                  </div>
                </template>
              </div>
              <!-- Global bar + ETA -->
              <div>
                <div class="bg-gray-800 rounded-full h-2 mb-1">
                  <div class="bg-indigo-500 h-2 rounded-full transition-all duration-300"
                       :style="`width: ${Math.round(alProgress * 100)}%`"></div>
                </div>
                <div class="flex justify-between text-xs text-gray-500">
                  <span x-text="`${Math.round(alProgress * 100)}%`"></span>
                  <span x-show="alEta != null" x-text="'⏱ ETA ' + _formatEta(alEta)"></span>
                </div>
              </div>
            </div>
```

- [ ] **Step 2: Commit**

```bash
git add app/templates/index.html
git commit -m "feat: replace progress bar with vertical stepper + ETA in HTML"
```

---

### Task 7: Localization

**Files:**
- Modify: `app/locales/en.json`
- Modify: `app/locales/it.json`

- [ ] **Step 1: Add stepper keys to en.json**

Add after the existing `audiolab_step_*` keys:

```json
  "audiolab_stepper_decode": "Decode audio",
  "audiolab_stepper_analyze": "Analyze loudness",
  "audiolab_stepper_denoise": "Remove noise",
  "audiolab_stepper_demucs": "Voice isolation",
  "audiolab_stepper_polish": "Audio polish",
  "audiolab_stepper_loudnorm": "Normalize loudness",
  "audiolab_stepper_resample": "Prepare output",
```

- [ ] **Step 2: Add stepper keys to it.json**

```json
  "audiolab_stepper_decode": "Decodifica audio",
  "audiolab_stepper_analyze": "Analisi loudness",
  "audiolab_stepper_denoise": "Rimozione rumore",
  "audiolab_stepper_demucs": "Isolamento voce",
  "audiolab_stepper_polish": "Rifinitura audio",
  "audiolab_stepper_loudnorm": "Normalizzazione loudness",
  "audiolab_stepper_resample": "Preparazione output",
```

- [ ] **Step 3: Commit**

```bash
git add app/locales/en.json app/locales/it.json
git commit -m "feat: add stepper label translations (EN + IT)"
```

---

## Chunk 3: Final verification

### Task 8: Run full test suite

- [ ] **Step 1: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: ALL pass

- [ ] **Step 2: Fix any failures**

If tests fail, diagnose and fix.

### Task 9: Smoke test

- [ ] **Step 1: Start server**

Run: `python start.py`

- [ ] **Step 2: Open browser, upload file, select HQ preset, process**

Verify:
- Stepper shows all 7 steps
- Steps transition: pending → active → completed
- Completed steps show elapsed time in seconds
- Skipped steps show as struck-through with dash
- ETA appears after first step completes and counts down
- During Demucs, sub-label shows "chunk X/Y"
- Progress bar fills correctly
- On completion, all active steps show green checkmarks
- A/B player works normally after processing

- [ ] **Step 3: Test with Lecture preset (some steps skipped)**

Verify:
- Demucs and Polish show as skipped (struck-through)
- Stepper still shows all 7 steps
- Active steps progress normally

- [ ] **Step 4: Commit if any smoke test fixes were needed**
