"""Fixed-reference Mel spectrograms and browser WAV encoding."""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from audio_highlight.audio import AudioSlice, YAMNET_SAMPLE_RATE_HZ


@dataclass(frozen=True, slots=True)
class SpectrogramConfig:
    sample_rate: int = YAMNET_SAMPLE_RATE_HZ
    n_fft: int = 1_024
    hop_length: int = 320
    n_mels: int = 64
    fmin: float = 50.0
    fmax: float = 8_000.0
    db_min: float = -80.0
    db_max: float = 0.0

    def __post_init__(self) -> None:
        if self.sample_rate != YAMNET_SAMPLE_RATE_HZ:
            raise ValueError(f"sample_rate must be {YAMNET_SAMPLE_RATE_HZ}")
        if self.n_fft <= 0 or self.hop_length <= 0 or self.n_mels <= 0:
            raise ValueError("FFT, hop, and Mel dimensions must be positive")
        if not 0 <= self.fmin < self.fmax <= self.sample_rate / 2:
            raise ValueError("Mel frequency range must fit inside the Nyquist range")
        if self.db_min >= self.db_max:
            raise ValueError("db_min must be lower than db_max")


DEFAULT_SPECTROGRAM_CONFIG = SpectrogramConfig()


def mel_spectrogram_db(
    samples: NDArray[np.float32],
    config: SpectrogramConfig = DEFAULT_SPECTROGRAM_CONFIG,
) -> NDArray[np.float32]:
    """Compute absolute-power Mel dB values without per-window normalization."""

    waveform = np.asarray(samples)
    if waveform.ndim != 1 or waveform.dtype != np.float32:
        raise ValueError("waveform must be one-dimensional float32")
    if not np.isfinite(waveform).all():
        raise ValueError("waveform must contain finite samples")
    if waveform.size < config.n_fft:
        waveform = np.pad(waveform, (0, config.n_fft - waveform.size))

    frames = np.lib.stride_tricks.sliding_window_view(waveform, config.n_fft)[
        :: config.hop_length
    ]
    window = np.hanning(config.n_fft).astype(np.float32)
    amplitude = np.abs(np.fft.rfft(frames * window, axis=1)) / np.sum(window)
    power = np.square(amplitude)
    mel_power = _mel_filter_bank(config) @ power.T
    floor_power = 10.0 ** (config.db_min / 10.0)
    decibels = 10.0 * np.log10(np.maximum(mel_power, floor_power))
    return np.clip(decibels, config.db_min, config.db_max).astype(
        np.float32, copy=False
    )


@lru_cache(maxsize=16)
def _mel_filter_bank(config: SpectrogramConfig) -> NDArray[np.float64]:
    frequencies = np.fft.rfftfreq(config.n_fft, d=1.0 / config.sample_rate)
    mel_min = _hz_to_mel(config.fmin)
    mel_max = _hz_to_mel(config.fmax)
    mel_points = np.linspace(mel_min, mel_max, config.n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bank = np.zeros((config.n_mels, frequencies.size), dtype=np.float64)
    for index in range(config.n_mels):
        left, center, right = hz_points[index : index + 3]
        bank[index] = np.maximum(
            0.0,
            np.minimum(
                (frequencies - left) / (center - left),
                (right - frequencies) / (right - center),
            ),
        )
    bank.setflags(write=False)
    return bank


def _hz_to_mel(frequency: float) -> float:
    return 2_595.0 * np.log10(1.0 + frequency / 700.0)


def _mel_to_hz(mel: NDArray[np.float64]) -> NDArray[np.float64]:
    return 700.0 * (np.power(10.0, mel / 2_595.0) - 1.0)


def render_spectrogram_rgb(
    decibels: NDArray[np.float32],
    config: SpectrogramConfig = DEFAULT_SPECTROGRAM_CONFIG,
) -> NDArray[np.uint8]:
    """Apply one fixed dB-to-color mapping shared by every labeling card."""

    values = np.asarray(decibels)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("spectrogram must be a finite two-dimensional array")
    normalized = np.clip(
        (values - config.db_min) / (config.db_max - config.db_min),
        0.0,
        1.0,
    )
    positions = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
    colors = np.asarray(
        [
            [0, 0, 4],
            [87, 16, 110],
            [188, 55, 84],
            [249, 142, 9],
            [252, 255, 164],
        ],
        dtype=np.float64,
    )
    rgb = np.stack(
        [np.interp(normalized, positions, colors[:, channel]) for channel in range(3)],
        axis=-1,
    )
    return np.flipud(np.rint(rgb).astype(np.uint8))


def audio_slice_to_wav_bytes(audio: AudioSlice) -> bytes:
    """Encode a playback-only PCM16 copy while leaving float32 samples unchanged."""

    clipped = np.clip(audio.samples, -1.0, 1.0)
    pcm16 = np.rint(clipped * 32_767.0).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as file:
        file.setnchannels(audio.channels)
        file.setsampwidth(2)
        file.setframerate(audio.sample_rate_hz)
        file.writeframes(pcm16.tobytes())
    return buffer.getvalue()
