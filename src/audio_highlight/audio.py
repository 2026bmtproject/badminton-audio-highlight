"""Interfaces for extracting 16 kHz mono audio from match media."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

YAMNET_SAMPLE_RATE_HZ = 16_000


@dataclass(frozen=True, slots=True)
class AudioClip:
    """Mono waveform samples anchored to absolute match time."""

    samples: Sequence[float]
    sample_rate_hz: int
    start_sec: float

    def __post_init__(self) -> None:
        if self.sample_rate_hz != YAMNET_SAMPLE_RATE_HZ:
            raise ValueError(f"audio must be {YAMNET_SAMPLE_RATE_HZ} Hz")
        if not math.isfinite(self.start_sec) or self.start_sec < 0:
            raise ValueError("start_sec must be a non-negative finite timestamp")

    @property
    def duration_sec(self) -> float:
        return len(self.samples) / self.sample_rate_hz

    @property
    def end_sec(self) -> float:
        return self.start_sec + self.duration_sec


class AudioExtractor(Protocol):
    """Backend boundary for decoding/resampling media; no backend is chosen yet."""

    def extract_mono(
        self,
        media_path: Path,
        *,
        start_sec: float,
        end_sec: float,
        sample_rate_hz: int = YAMNET_SAMPLE_RATE_HZ,
    ) -> AudioClip:
        """Extract an absolute match-time interval as mono waveform audio."""
        ...
