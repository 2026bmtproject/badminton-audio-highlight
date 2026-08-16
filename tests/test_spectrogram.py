from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np
import pytest

from audio_highlight.audio import AudioSlice, NormalizedAudioSource
from audio_highlight.spectrogram import (
    DEFAULT_SPECTROGRAM_CONFIG,
    SpectrogramConfig,
    audio_slice_to_wav_bytes,
    mel_spectrogram_db,
    render_spectrogram_rgb,
)


def sine_wave(amplitude: float = 0.2) -> np.ndarray:
    time = np.arange(48_000, dtype=np.float32) / 16_000
    return (amplitude * np.sin(2.0 * np.pi * 440.0 * time)).astype(np.float32)


def test_mel_spectrogram_shape_and_fixed_display_range() -> None:
    config = DEFAULT_SPECTROGRAM_CONFIG

    spectrogram = mel_spectrogram_db(sine_wave(), config)
    image = render_spectrogram_rgb(spectrogram, config)

    assert spectrogram.shape == (64, 147)
    assert spectrogram.dtype == np.float32
    assert np.min(spectrogram) >= config.db_min
    assert np.max(spectrogram) <= config.db_max
    assert image.shape == (64, 147, 3)
    assert image.dtype == np.uint8


def test_spectrogram_does_not_normalize_each_window_by_its_maximum() -> None:
    loud = mel_spectrogram_db(sine_wave(0.5))
    quiet = mel_spectrogram_db(sine_wave(0.05))

    assert np.max(loud) - np.max(quiet) == pytest.approx(20.0, abs=0.25)
    assert np.mean(render_spectrogram_rgb(loud)) > np.mean(
        render_spectrogram_rgb(quiet)
    )


def test_spectrogram_configuration_is_centralized_and_validated() -> None:
    assert DEFAULT_SPECTROGRAM_CONFIG == SpectrogramConfig(
        sample_rate=16_000,
        n_fft=1_024,
        hop_length=320,
        n_mels=64,
        fmin=50.0,
        fmax=8_000.0,
        db_min=-80.0,
        db_max=0.0,
    )
    with pytest.raises(ValueError, match="db_min"):
        SpectrogramConfig(db_min=0.0, db_max=0.0)


def test_absolute_audio_slice_and_pcm_wav_copy(tmp_path: Path) -> None:
    samples = np.arange(160_000, dtype=np.float32) / 160_000.0
    cache = tmp_path / "audio.f32le"
    cache.write_bytes(samples.astype("<f4").tobytes())

    with NormalizedAudioSource(cache) as source:
        audio = source.slice_absolute(2.0, 5.0)

    assert (audio.start_sec, audio.end_sec) == (2.0, 5.0)
    assert audio.samples.size == 48_000
    np.testing.assert_array_equal(audio.samples, samples[32_000:80_000])
    original = audio.samples.copy()
    wav_bytes = audio_slice_to_wav_bytes(audio)
    np.testing.assert_array_equal(audio.samples, original)

    with wave.open(io.BytesIO(wav_bytes), "rb") as file:
        assert file.getframerate() == 16_000
        assert file.getnchannels() == 1
        assert file.getsampwidth() == 2
        assert file.getnframes() == 48_000


def test_audio_slice_contract_rejects_relative_or_wrong_length_waveform() -> None:
    samples = np.zeros(47_999, dtype=np.float32)

    with pytest.raises(ValueError, match="expected 48000"):
        AudioSlice(10.0, 13.0, 16_000, 1, samples)
