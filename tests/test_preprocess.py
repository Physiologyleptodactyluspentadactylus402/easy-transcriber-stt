"""Tests for app.core.preprocess — decode_to_wav, analyze_lufs, apply_loudnorm, apply_voice_isolation."""
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from app.core.preprocess import analyze_lufs, apply_loudnorm, apply_voice_isolation, decode_to_wav


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
