"""Tests for app.core.preprocess — decode_to_wav, analyze_lufs, apply_loudnorm, apply_voice_isolation."""
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from app.core.preprocess import (
    PreprocessConfig, PipelineResult, analyze_lufs, apply_denoise,
    apply_loudnorm, apply_voice_isolation, decode_to_wav, run_pipeline,
    _apply_denoise_ffmpeg, _apply_polish,
)


# ---------------------------------------------------------------------------
# Task 1 — decode_to_wav
# ---------------------------------------------------------------------------


class TestDecodeToWav:
    def test_output_file_is_created(self, audio_tone_10s, tmp_path):
        out = tmp_path / "out.wav"
        decode_to_wav(audio_tone_10s, out)
        assert out.exists()

    def test_output_is_valid_wav(self, audio_tone_10s, tmp_path):
        out = tmp_path / "out.wav"
        decode_to_wav(audio_tone_10s, out)
        with wave.open(str(out)) as wf:
            assert wf.getnchannels() == 1  # mono
            assert wf.getframerate() == 48000

    def test_custom_sample_rate(self, audio_tone_10s, tmp_path):
        out = tmp_path / "out_16k.wav"
        decode_to_wav(audio_tone_10s, out, sample_rate=16000)
        with wave.open(str(out)) as wf:
            assert wf.getframerate() == 16000

    def test_raises_on_missing_input(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            decode_to_wav(Path("/nonexistent/file.mp3"), tmp_path / "out.wav")

    def test_raises_on_ffmpeg_error(self, tmp_path):
        bad_input = tmp_path / "not_audio.txt"
        bad_input.write_text("this is not audio")
        with pytest.raises(RuntimeError):
            decode_to_wav(bad_input, tmp_path / "out.wav")

    def test_output_duration_approx_10s(self, audio_tone_10s, tmp_path):
        out = tmp_path / "out.wav"
        decode_to_wav(audio_tone_10s, out)
        with wave.open(str(out)) as wf:
            duration = wf.getnframes() / wf.getframerate()
        assert abs(duration - 10.0) < 0.1


# ---------------------------------------------------------------------------
# Task 1 — analyze_lufs
# ---------------------------------------------------------------------------


class TestAnalyzeLufs:
    def test_returns_dict_with_required_keys(self, audio_tone_10s, tmp_path):
        wav = tmp_path / "tone.wav"
        decode_to_wav(audio_tone_10s, wav)
        result = analyze_lufs(wav)
        required_keys = {"input_i", "input_tp", "input_lra", "input_thresh", "target_offset"}
        assert required_keys.issubset(result.keys())

    def test_values_are_floats(self, audio_tone_10s, tmp_path):
        wav = tmp_path / "tone.wav"
        decode_to_wav(audio_tone_10s, wav)
        result = analyze_lufs(wav)
        for key, value in result.items():
            assert isinstance(value, float), f"Expected float for {key}, got {type(value)}"

    def test_lufs_is_negative(self, audio_tone_10s, tmp_path):
        """A -20 dBFS 440 Hz tone should have a negative integrated LUFS."""
        wav = tmp_path / "tone.wav"
        decode_to_wav(audio_tone_10s, wav)
        result = analyze_lufs(wav)
        assert result["input_i"] < 0.0

    def test_lufs_approx_minus20(self, audio_tone_10s, tmp_path):
        """Tone at -20 dBFS should measure in the range -26 to -14 LUFS.

        LUFS (EBU R128) uses K-weighting which attenuates certain frequencies,
        so a pure 440 Hz sine at -20 dBFS typically measures around -24 LUFS —
        a few LU below its dBFS level.  We use a generous ±6 LU window.
        """
        wav = tmp_path / "tone.wav"
        decode_to_wav(audio_tone_10s, wav)
        result = analyze_lufs(wav)
        assert abs(result["input_i"] - (-20.0)) < 6.0

    def test_raises_on_missing_wav(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            analyze_lufs(tmp_path / "nonexistent.wav")


# ---------------------------------------------------------------------------
# Task 2 — apply_loudnorm
# ---------------------------------------------------------------------------


class TestApplyLoudnorm:
    def test_output_file_is_created(self, audio_tone_10s, tmp_path):
        wav_in = tmp_path / "in.wav"
        wav_out = tmp_path / "out.wav"
        decode_to_wav(audio_tone_10s, wav_in)
        apply_loudnorm(wav_in, wav_out)
        assert wav_out.exists()

    def test_output_is_48k_mono_16bit(self, audio_tone_10s, tmp_path):
        wav_in = tmp_path / "in.wav"
        wav_out = tmp_path / "out.wav"
        decode_to_wav(audio_tone_10s, wav_in)
        apply_loudnorm(wav_in, wav_out)
        with wave.open(str(wav_out)) as wf:
            assert wf.getnchannels() == 1
            assert wf.getframerate() == 48000
            assert wf.getsampwidth() == 2  # 16-bit = 2 bytes

    def test_output_within_2_lufs_of_target(self, audio_tone_10s, tmp_path):
        wav_in = tmp_path / "in.wav"
        wav_out = tmp_path / "out.wav"
        decode_to_wav(audio_tone_10s, wav_in)
        target = -16.0
        apply_loudnorm(wav_in, wav_out, target_lufs=target)
        result = analyze_lufs(wav_out)
        assert abs(result["input_i"] - target) < 2.0, (
            f"Expected LUFS close to {target}, got {result['input_i']}"
        )

    def test_accepts_precomputed_measured(self, audio_tone_10s, tmp_path):
        """Passing measured= skips the internal analyze_lufs call."""
        wav_in = tmp_path / "in.wav"
        wav_out = tmp_path / "out.wav"
        decode_to_wav(audio_tone_10s, wav_in)
        measured = analyze_lufs(wav_in)
        # Should not raise, should produce output
        apply_loudnorm(wav_in, wav_out, measured=measured)
        assert wav_out.exists()

    def test_raises_on_missing_input(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            apply_loudnorm(tmp_path / "nonexistent.wav", tmp_path / "out.wav")

    def test_custom_target_lufs(self, audio_tone_10s, tmp_path):
        """Verify a different target (e.g., -23 LUFS) is also honoured within 2 LU."""
        wav_in = tmp_path / "in.wav"
        wav_out = tmp_path / "out.wav"
        decode_to_wav(audio_tone_10s, wav_in)
        target = -23.0
        apply_loudnorm(wav_in, wav_out, target_lufs=target)
        result = analyze_lufs(wav_out)
        assert abs(result["input_i"] - target) < 2.0


# ---------------------------------------------------------------------------
# Task 3 — apply_voice_isolation
# ---------------------------------------------------------------------------


class TestApplyVoiceIsolation:
    def test_output_file_is_created(self, tmp_path):
        from pydub import AudioSegment
        from pydub.generators import Sine
        tone = Sine(440).to_audio_segment(duration=2000)
        wav_in = tmp_path / "input.wav"
        tone.export(str(wav_in), format="wav")
        wav_out = tmp_path / "vocals.wav"

        # Mock demucs to avoid downloading the model
        num_samples = 2 * 44100  # 2s at 44.1kHz (demucs native rate)
        vocals_np = np.sin(np.linspace(0, 440 * 2 * np.pi * 2, num_samples)).astype(np.float32)

        with patch("app.core.preprocess._DEMUCS_AVAILABLE", True), \
             patch("app.core.preprocess._run_demucs") as mock_run:
            mock_run.return_value = vocals_np
            result = apply_voice_isolation(wav_in, wav_out)

        assert result.exists()

    def test_output_is_48k(self, tmp_path):
        from pydub import AudioSegment
        from pydub.generators import Sine
        tone = Sine(440).to_audio_segment(duration=2000)
        wav_in = tmp_path / "input.wav"
        tone.export(str(wav_in), format="wav")
        wav_out = tmp_path / "vocals.wav"

        num_samples = 2 * 44100
        vocals_np = np.sin(np.linspace(0, 440 * 2 * np.pi * 2, num_samples)).astype(np.float32)

        with patch("app.core.preprocess._DEMUCS_AVAILABLE", True), \
             patch("app.core.preprocess._run_demucs") as mock_run:
            mock_run.return_value = vocals_np
            apply_voice_isolation(wav_in, wav_out)

        seg = AudioSegment.from_wav(str(wav_out))
        assert seg.frame_rate == 48000  # resampled back to 48kHz

    def test_raises_when_demucs_unavailable(self, tmp_path):
        from pydub import AudioSegment
        wav_in = tmp_path / "input.wav"
        AudioSegment.silent(duration=1000).export(str(wav_in), format="wav")

        with patch("app.core.preprocess._DEMUCS_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="[Dd]emucs"):
                apply_voice_isolation(wav_in, tmp_path / "out.wav")


# ---------------------------------------------------------------------------
# Task 4 — apply_denoise
# ---------------------------------------------------------------------------


class TestApplyDenoise:
    def test_output_file_is_created(self, tmp_path):
        from pydub import AudioSegment
        from pydub.generators import Sine
        tone = Sine(440).to_audio_segment(duration=2000)
        wav_in = tmp_path / "input_48k.wav"
        tone.export(str(wav_in), format="wav", parameters=["-ar", "48000"])
        wav_out = tmp_path / "denoised.wav"

        with patch("app.core.preprocess._DEEPFILTER_AVAILABLE", True), \
             patch("app.core.preprocess._run_deepfilter") as mock_run:
            mock_run.return_value = np.zeros(2 * 48000, dtype=np.float32)
            result = apply_denoise(wav_in, wav_out, engine="deepfilter")

        assert result.exists()

    def test_output_is_48k(self, tmp_path):
        from pydub import AudioSegment
        from pydub.generators import Sine
        tone = Sine(440).to_audio_segment(duration=2000)
        wav_in = tmp_path / "input_48k.wav"
        tone.export(str(wav_in), format="wav", parameters=["-ar", "48000"])
        wav_out = tmp_path / "denoised.wav"

        with patch("app.core.preprocess._DEEPFILTER_AVAILABLE", True), \
             patch("app.core.preprocess._run_deepfilter") as mock_run:
            mock_run.return_value = np.zeros(2 * 48000, dtype=np.float32)
            apply_denoise(wav_in, wav_out, engine="deepfilter")

        seg = AudioSegment.from_wav(str(wav_out))
        assert seg.frame_rate == 48000

    def test_raises_when_deepfilter_unavailable(self, tmp_path):
        from pydub import AudioSegment
        wav_in = tmp_path / "input.wav"
        AudioSegment.silent(duration=1000).export(str(wav_in), format="wav")

        with patch("app.core.preprocess._DEEPFILTER_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="[Dd]eep[Ff]ilter"):
                apply_denoise(wav_in, tmp_path / "out.wav", engine="deepfilter")


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

    @patch("app.core.preprocess._DEEPFILTER_AVAILABLE", False)
    def test_dispatcher_deepfilter_unavailable(self, audio_tone_10s, tmp_path):
        output = tmp_path / "denoised.wav"
        with pytest.raises(RuntimeError, match="DeepFilterNet is not installed"):
            apply_denoise(audio_tone_10s, output, engine="deepfilter")


class TestResampleTo16k:
    def test_output_is_16k_mono(self, audio_tone_10s, tmp_path):
        from app.core.preprocess import _resample_to_16k
        wav_in = tmp_path / "in.wav"
        decode_to_wav(audio_tone_10s, wav_in, sample_rate=48000)
        wav_out = tmp_path / "out_16k.wav"
        _resample_to_16k(wav_in, wav_out)
        with wave.open(str(wav_out)) as wf:
            assert wf.getframerate() == 16000
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2


class TestRunPipeline:
    def test_loudnorm_only(self, audio_tone_10s, tmp_path):
        config = PreprocessConfig(
            loudnorm=True,
            loudnorm_target=-16.0,
            voice_isolation=False,
            denoise=False,
        )
        result = run_pipeline(audio_tone_10s, tmp_path / "work", config)
        assert result.processed_path.exists()
        assert result.processed_48k_path.exists()
        assert result.original_path.exists()
        assert result.stats is not None
        assert "original_lufs" in result.stats
        assert "processed_lufs" in result.stats
        assert result.cancelled is False
        # Final output is 16kHz
        with wave.open(str(result.processed_path)) as wf:
            assert wf.getframerate() == 16000
        # 48k version for player
        with wave.open(str(result.processed_48k_path)) as wf:
            assert wf.getframerate() == 48000

    def test_no_steps(self, audio_tone_10s, tmp_path):
        config = PreprocessConfig(
            loudnorm=False,
            voice_isolation=False,
            denoise=False,
        )
        result = run_pipeline(audio_tone_10s, tmp_path / "work", config)
        assert result.processed_path.exists()
        assert result.stats is not None
        assert "original_lufs" in result.stats
        assert result.cancelled is False

    def test_cancellation_before_loudnorm(self, audio_tone_10s, tmp_path):
        cancel_flag = {"cancelled": True}
        config = PreprocessConfig(loudnorm=True, voice_isolation=True, denoise=True)
        result = run_pipeline(
            audio_tone_10s, tmp_path / "work", config,
            cancel_flag=cancel_flag,
        )
        assert result.cancelled is True
        assert result.stats is not None
        assert "original_lufs" in result.stats
        assert "decode" in result.steps_completed

    def test_progress_callback(self, audio_tone_10s, tmp_path):
        calls = []
        def cb(frac, step, msg):
            calls.append((frac, step, msg))
        config = PreprocessConfig(loudnorm=True, voice_isolation=False, denoise=False)
        run_pipeline(audio_tone_10s, tmp_path / "work", config, progress_callback=cb)
        assert len(calls) >= 4  # decode, analyze, loudnorm, resample, done
        assert calls[0][1] == "decode"
        assert calls[-1][0] == 1.0

    def test_steps_completed_tracks_steps(self, audio_tone_10s, tmp_path):
        config = PreprocessConfig(loudnorm=True, voice_isolation=False, denoise=False)
        result = run_pipeline(audio_tone_10s, tmp_path / "work", config)
        assert "decode" in result.steps_completed
        assert "loudnorm" in result.steps_completed
        assert "resample" in result.steps_completed
        assert "demucs" not in result.steps_completed
        assert "deepfilter" not in result.steps_completed

    def test_pipeline_polish_only(self, audio_tone_10s, tmp_path):
        config = PreprocessConfig(
            loudnorm=False, voice_isolation=False, denoise=False, polish=True,
        )
        result = run_pipeline(audio_tone_10s, tmp_path / "work", config)
        assert result.processed_path.exists()
        assert "polish" in result.steps_completed
        assert result.cancelled is False

    def test_pipeline_hq_preset(self, audio_tone_10s, tmp_path):
        """HQ preset enables loudnorm + denoise + polish (skip voice_isolation for speed)."""
        config = PreprocessConfig(
            loudnorm=True, loudnorm_target=-16.0,
            voice_isolation=False, denoise=True, polish=True,
        )
        result = run_pipeline(audio_tone_10s, tmp_path / "work", config)
        assert result.processed_path.exists()
        assert "ffmpeg_denoise" in result.steps_completed
        assert "polish" in result.steps_completed
        assert "loudnorm" in result.steps_completed

    def test_pipeline_lecture_no_polish(self, audio_tone_10s, tmp_path):
        """Lecture preset does NOT include polish."""
        config = PreprocessConfig(
            loudnorm=True, voice_isolation=False, denoise=True, polish=False,
        )
        result = run_pipeline(audio_tone_10s, tmp_path / "work", config)
        assert "polish" not in result.steps_completed

    def test_pipeline_order_denoise_before_loudnorm(self, audio_tone_10s, tmp_path):
        """Verify denoise runs before loudnorm in the new pipeline order."""
        calls = []
        def cb(frac, step, msg):
            calls.append(step)
        config = PreprocessConfig(
            loudnorm=True, voice_isolation=False, denoise=True, polish=False,
        )
        run_pipeline(audio_tone_10s, tmp_path / "work", config, progress_callback=cb)
        step_names = [s for s in calls if s in ("ffmpeg_denoise", "loudnorm")]
        assert step_names == ["ffmpeg_denoise", "loudnorm"]

    def test_pipeline_order_polish_before_loudnorm(self, audio_tone_10s, tmp_path):
        """Verify polish runs before loudnorm."""
        calls = []
        def cb(frac, step, msg):
            calls.append(step)
        config = PreprocessConfig(
            loudnorm=True, voice_isolation=False, denoise=False, polish=True,
        )
        run_pipeline(audio_tone_10s, tmp_path / "work", config, progress_callback=cb)
        step_names = [s for s in calls if s in ("polish", "loudnorm")]
        assert step_names == ["polish", "loudnorm"]


class TestApplyPolish:
    def test_output_file_is_created(self, audio_tone_10s, tmp_path):
        output = tmp_path / "polished.wav"
        result = _apply_polish(audio_tone_10s, output)
        assert result.exists()

    def test_output_is_48k_mono(self, audio_tone_10s, tmp_path):
        output = tmp_path / "polished.wav"
        _apply_polish(audio_tone_10s, output)
        with wave.open(str(output)) as wf:
            assert wf.getframerate() == 48000
            assert wf.getnchannels() == 1

    def test_raises_on_missing_input(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _apply_polish(tmp_path / "nonexistent.wav", tmp_path / "out.wav")
