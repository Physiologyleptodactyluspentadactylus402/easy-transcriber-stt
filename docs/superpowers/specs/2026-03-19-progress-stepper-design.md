# Audio Lab Progress Stepper with ETA

**Date:** 2026-03-19
**Status:** Approved

## Problem

The Audio Lab pipeline can take several minutes (up to 5+ minutes for long files with Demucs). Currently the UI shows only a generic progress bar and a text message. Users have no visibility into:

1. Which steps have completed and how long each took
2. Which step is currently running (and sub-progress for chunked operations)
3. Which steps are still pending
4. Which steps were skipped (disabled in config)
5. Estimated time remaining

## Solution

Replace the simple progress bar with a **vertical stepper** showing all pipeline steps, plus a global progress bar and dynamic ETA.

## Design Decisions

- **Vertical stepper** layout chosen over segmented bar or minimal bar (more informative)
- **Skipped steps shown** as struck-through/grayed (pipeline always fully visible)
- **Dynamic ETA** based on real elapsed times per completed step, weighted proportionally
- **Sub-progress** for Demucs chunks (chunk X/Y)
- **Canonical step ID `"denoise"`** used regardless of engine (ffmpeg or deepfilter) — `run_pipeline()` will be updated to use `"denoise"` consistently
- **`analyze` step is always active** — it cannot be skipped (LUFS measurement is always needed)
- **ETA also interpolated during Demucs** using sub-progress, not just at step boundaries

## WebSocket Protocol Changes

### New message: `audiolab_steps`

Sent once at the start of processing, before any `audiolab_progress` messages.

```json
{
  "type": "audiolab_steps",
  "job_id": "abc123",
  "steps": [
    {"id": "decode",   "label_key": "audiolab_stepper_decode",   "active": true},
    {"id": "analyze",  "label_key": "audiolab_stepper_analyze",  "active": true},
    {"id": "denoise",  "label_key": "audiolab_stepper_denoise",  "active": true},
    {"id": "demucs",   "label_key": "audiolab_stepper_demucs",   "active": false},
    {"id": "polish",   "label_key": "audiolab_stepper_polish",   "active": false},
    {"id": "loudnorm", "label_key": "audiolab_stepper_loudnorm", "active": true},
    {"id": "resample", "label_key": "audiolab_stepper_resample", "active": true}
  ]
}
```

- `active: false` means the step is disabled in the current config and will be skipped.
- `label_key` references an i18n key for the step name.

### Enhanced message: `audiolab_progress`

Existing fields preserved, new fields added:

```json
{
  "type": "audiolab_progress",
  "job_id": "abc123",
  "progress": 0.35,
  "step": "demucs",
  "message": "Isolating voice...",
  "sub_progress": 0.4,
  "sub_label": "chunk 2/5",
  "elapsed_sec": 45.2,
  "eta_sec": 82
}
```

New fields:
- `sub_progress` (float 0-1, optional): sub-step progress within current step (e.g., Demucs chunk progress). `null` if not applicable.
- `sub_label` (string, optional): human-readable sub-progress label (e.g., "chunk 2/5"). `null` if not applicable.
- `elapsed_sec` (float): total seconds elapsed since pipeline start.
- `eta_sec` (float|null): estimated seconds remaining. `null` if not enough data to estimate.

### Step completion signal

When a step finishes, a progress message is sent with the *next* step as `step` (or `"done"` for the final message). The frontend tracks which steps are complete by observing step transitions.

To make this explicit, we add a `completed_step` field:

```json
{
  "type": "audiolab_progress",
  "step": "demucs",
  "completed_step": "denoise",
  "completed_step_elapsed": 12.3,
  ...
}
```

- `completed_step` (string, optional): ID of the step that just finished.
- `completed_step_elapsed` (float, optional): how long that step took in seconds.

## ETA Calculation

### Step weights

Each step has a relative weight representing its expected share of total processing time:

| Step | Weight (ffmpeg denoise) | Weight (deepfilter denoise) |
|------|------------------------|-----------------------------|
| decode | 0.05 | 0.05 |
| analyze | 0.03 | 0.03 |
| denoise | 0.10 | 0.18 |
| demucs | 0.52 | 0.47 |
| polish | 0.05 | 0.05 |
| loudnorm | 0.15 | 0.12 |
| resample | 0.10 | 0.10 |

Raw weights sum to 1.0 for each column. Weights of skipped steps are excluded and the remaining weights are re-normalized to sum to 1.0.

### Algorithm

```
active_weights = {step: weight for step in steps if step.active}
total_weight = sum(active_weights.values())
normalized = {step: w / total_weight for step, w in active_weights.items()}

# After completing step N:
completed_weight = sum(normalized[s] for s in completed_steps)
completed_time = total_elapsed_sec

if completed_weight > 0:
    pace = completed_time / completed_weight      # seconds per unit weight
    remaining_weight = 1.0 - completed_weight
    eta_sec = remaining_weight * pace
else:
    eta_sec = audio_duration_sec * 4               # rough initial estimate
```

The ETA is recalculated after each step completes, and also **during Demucs** using sub-progress interpolation:

```
# During Demucs step with sub_progress = chunk_done / chunk_total:
demucs_weight = normalized["demucs"]
partial_demucs_weight = demucs_weight * sub_progress
current_completed_weight = completed_weight + partial_demucs_weight
pace = elapsed_sec / current_completed_weight
eta_sec = (1.0 - current_completed_weight) * pace
```

ETA display: show `null` (hidden) until at least one step has completed. Before that, no ETA is shown.

### Sub-progress for Demucs

Demucs processes audio in 5-minute chunks. During Demucs, the progress callback fires after each chunk:

```
sub_progress = chunks_completed / total_chunks
sub_label = f"chunk {chunks_completed}/{total_chunks}"
```

The overall `progress` fraction interpolates within the Demucs weight range based on `sub_progress`.

## Backend Changes

### `app/core/preprocess.py`

1. **New function `build_step_list(config: PreprocessConfig) -> list[dict]`**
   - Returns the ordered list of steps with `id`, `label_key`, and `active` based on config.
   - Used by both the WebSocket initial message and internally for ETA calculation.

2. **New class `PipelineProgressTracker`**
   - Initialized with `step_list` and `audio_duration_sec`.
   - Methods:
     - `start()` — records pipeline start time.
     - `begin_step(step_id)` — marks step as in-progress.
     - `complete_step(step_id)` — records step duration, recalculates ETA.
     - `update_sub_progress(fraction, label)` — for chunk-level progress.
     - `get_progress_data() -> dict` — returns all fields for the WebSocket message.
   - Holds step weights, elapsed times, and computes ETA.

3. **Modify `run_pipeline()`**
   - Create `PipelineProgressTracker` at start.
   - Call `tracker.begin_step()` / `tracker.complete_step()` around each pipeline phase.
   - Pass `tracker.update_sub_progress` into `_run_demucs()` for chunk callbacks.
   - The existing `_progress()` callback signature changes from `(fraction, step, message)` to `(tracker_data: dict)` — or we keep the old signature and add tracker data as extra kwargs.

4. **Modify `_run_demucs()`**
   - Accept optional `sub_progress_callback(fraction, label)` parameter.
   - Call it after each chunk completes.

5. **Modify `apply_voice_isolation()`**
   - Accept and forward `sub_progress_callback` to `_run_demucs()`.

6. **Fix `run_pipeline()` step ID consistency**
   - Change denoise step to always use `"denoise"` as the step ID (not `"ffmpeg_denoise"` / `"deepfilter"`).
   - Add `steps_completed.append("analyze")` after LUFS analysis (currently missing).
   - Append `"decode"` to `steps_completed` is already done.

### `app/main.py`

1. **In `_run_audiolab()`**:
   - Call `build_step_list(config)` and broadcast `audiolab_steps` message before starting pipeline.
   - Update the progress callback to forward the enriched tracker data fields.

### Progress callback compatibility

The existing callback signature is `(fraction: float, step: str, message: str)`. To maintain backward compatibility:

- Add optional keyword arguments: `sub_progress=None, sub_label=None, elapsed_sec=0, eta_sec=None, completed_step=None, completed_step_elapsed=None`.
- `app/main.py` reads these kwargs and includes them in the WebSocket broadcast.

## Frontend Changes

### `app/static/app.js`

New state variables:
```javascript
alSteps: [],           // [{id, label_key, active, status, elapsed}]
alElapsed: 0,          // total elapsed seconds
alEta: null,           // estimated remaining seconds
alSubProgress: null,   // 0-1 sub-progress within current step
alSubLabel: '',        // "chunk 2/5"
```

Step statuses (computed from messages):
- `"pending"` — not yet started
- `"active"` — currently running
- `"completed"` — done (with `elapsed` seconds)
- `"skipped"` — `active: false` from step list

New WebSocket handlers:
- `audiolab_steps`: Initialize `alSteps` array, set inactive steps to `"skipped"`.
- `audiolab_progress` (enhanced): Update current step status, store sub-progress, elapsed, ETA. When `completed_step` is present, mark that step as `"completed"` with its elapsed time.

New Alpine method (inside `app()` return object, NOT a standalone function):
- `_formatEta(seconds)`: Formats ETA as "~Xm Ys" or "< 1m" for short durations.

State reset in `alProcess()`:
- Reset `alSteps`, `alElapsed`, `alEta`, `alSubProgress`, `alSubLabel` when starting a new job (alongside existing `alProgress` and `alMessage` resets).

### `app/templates/index.html`

Replace the current progress section (lines 634-640) with the stepper:

```html
<div x-show="alStatus==='processing'" class="space-y-3">
  <!-- Stepper -->
  <div class="space-y-1">
    <template x-for="s in alSteps" :key="s.id">
      <div class="flex items-center gap-2 text-sm" :class="s.status === 'skipped' ? 'opacity-40 line-through' : ''">
        <!-- Icon -->
        <span x-show="s.status==='completed'" class="text-green-400">✓</span>
        <span x-show="s.status==='active'" class="text-indigo-400 animate-spin">⟳</span>
        <span x-show="s.status==='pending'" class="text-gray-600">○</span>
        <span x-show="s.status==='skipped'" class="text-gray-600">–</span>
        <!-- Label -->
        <span :class="s.status==='active' ? 'text-indigo-300 font-medium' : s.status==='completed' ? 'text-gray-300' : 'text-gray-500'"
              x-text="t(s.label_key)"></span>
        <!-- Elapsed / sub-progress -->
        <span x-show="s.status==='completed'" class="text-xs text-gray-600" x-text="s.elapsed ? `${Math.round(s.elapsed)}s` : ''"></span>
        <span x-show="s.status==='active' && alSubLabel" class="text-xs text-indigo-400" x-text="alSubLabel"></span>
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
      <span x-show="alEta !== null" x-text="'⏱ ETA ' + _formatEta(alEta)"></span>
    </div>
  </div>
</div>
```

## Localization

### New keys needed

Stepper labels use short noun phrases (not action phrases with "..."). Some keys already exist with action-phrase text used for progress messages — we add new **stepper-specific** keys:

| Key | EN | IT |
|-----|----|----|
| `audiolab_stepper_decode` | Decode audio | Decodifica audio |
| `audiolab_stepper_analyze` | Analyze loudness | Analisi loudness |
| `audiolab_stepper_denoise` | Remove noise | Rimozione rumore |
| `audiolab_stepper_demucs` | Voice isolation | Isolamento voce |
| `audiolab_stepper_polish` | Audio polish | Rifinitura audio |
| `audiolab_stepper_loudnorm` | Normalize loudness | Normalizzazione loudness |
| `audiolab_stepper_resample` | Prepare output | Preparazione output |

The existing `audiolab_step_*` keys (with "..." suffixes) remain unchanged — they are used for the progress message text.

The `label_key` in `audiolab_steps` messages references the new `audiolab_stepper_*` keys.

## Testing

1. **Unit tests for `PipelineProgressTracker`**: ETA calculation with various step completion patterns, weight normalization with skipped steps.
2. **Unit tests for `build_step_list`**: Correct step list for each preset (lecture, clean, hq, custom).
3. **Integration test**: Verify `audiolab_steps` message is sent before first `audiolab_progress`.
4. **Integration test**: Verify `completed_step` and `completed_step_elapsed` fields are present in progress messages.
5. **Frontend**: Manual smoke test — verify stepper renders, updates, shows ETA.

## Final State

When the pipeline completes (`audiolab_done` message), the frontend marks all active steps as `"completed"`. This ensures the stepper shows all green checkmarks at completion, even if the last `completed_step` message was missed.

## What Does NOT Change

- Pipeline logic (order, algorithms, chunking)
- A/B player, download, send-to-transcribe
- Presets and options
- File formats or API endpoints
