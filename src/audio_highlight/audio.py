"""Normalize match media once, then slice absolute-time waveform windows."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from audio_highlight.windows import AnalysisWindow

YAMNET_SAMPLE_RATE_HZ = 16_000
MONO_CHANNELS = 1
_FLOAT32_BYTES = np.dtype("<f4").itemsize
AUDIO_CACHE_FORMAT_VERSION = 3
FFMPEG_PRECLIP_FILTER = "aformat=sample_rates=16000:channel_layouts=mono"
FFMPEG_NORMALIZATION_FILTER = (
    f"{FFMPEG_PRECLIP_FILTER},"
    "asoftclip=type=hard:threshold=1"
)


class AudioError(RuntimeError):
    """Base error for normalization and waveform slicing."""


class FFmpegNotFoundError(AudioError):
    """Raised when a cache must be built but FFmpeg is unavailable."""


class AudioDecodeError(AudioError):
    """Raised when FFmpeg cannot create normalized audio."""


class AudioRangeError(AudioError):
    """Raised when a requested complete window is outside normalized audio."""


def _decimal_seconds(value: float, field: str) -> Decimal:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{field} must be a non-negative finite timestamp")
    return Decimal(str(value))


def _round_samples(seconds: Decimal, sample_rate_hz: int) -> int:
    return int(
        (seconds * sample_rate_hz).to_integral_value(rounding=ROUND_HALF_EVEN)
    )


def timestamp_to_sample_index(
    timestamp_sec: float,
    sample_rate_hz: int = YAMNET_SAMPLE_RATE_HZ,
) -> int:
    """Convert an absolute timestamp to its nearest sample using ties-to-even."""

    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    return _round_samples(
        _decimal_seconds(timestamp_sec, "timestamp_sec"),
        sample_rate_hz,
    )


def _range_sample_count(start_sec: float, end_sec: float, sample_rate_hz: int) -> int:
    start = _decimal_seconds(start_sec, "start_sec")
    end = _decimal_seconds(end_sec, "end_sec")
    if end <= start:
        raise ValueError("end_sec must be later than start_sec")
    return _round_samples(end - start, sample_rate_hz)


def _validate_waveform(
    start_sec: float,
    end_sec: float,
    sample_rate_hz: int,
    channels: int,
    samples: NDArray[np.float32],
) -> None:
    expected_count = _range_sample_count(start_sec, end_sec, sample_rate_hz)
    if sample_rate_hz != YAMNET_SAMPLE_RATE_HZ:
        raise ValueError(f"audio must be {YAMNET_SAMPLE_RATE_HZ} Hz")
    if channels != MONO_CHANNELS:
        raise ValueError("audio must be mono")
    if samples.ndim != 1 or samples.dtype != np.float32:
        raise ValueError("samples must be a one-dimensional float32 waveform")
    if samples.size != expected_count:
        raise ValueError(
            f"samples has {samples.size} values; expected {expected_count} "
            "for the declared timestamps"
        )


@dataclass(frozen=True, slots=True)
class AudioSlice:
    """Identity-neutral waveform slice addressed only by absolute match time."""

    start_sec: float
    end_sec: float
    sample_rate_hz: int
    channels: int
    samples: NDArray[np.float32]

    def __post_init__(self) -> None:
        _validate_waveform(
            self.start_sec,
            self.end_sec,
            self.sample_rate_hz,
            self.channels,
            self.samples,
        )


@dataclass(frozen=True, slots=True)
class AudioWindow:
    """A complete mono float32 waveform with absolute match timestamps."""

    segment_index: int
    start_sec: float
    end_sec: float
    sample_rate_hz: int
    channels: int
    samples: NDArray[np.float32]

    def __post_init__(self) -> None:
        if self.segment_index < 0:
            raise ValueError("segment_index must be non-negative")
        _validate_waveform(
            self.start_sec,
            self.end_sec,
            self.sample_rate_hz,
            self.channels,
            self.samples,
        )


class NormalizedAudioSource:
    """Random-access view over one decoded 16 kHz mono float32 cache file."""

    __slots__ = ("_samples", "channels", "path", "sample_rate_hz")

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"normalized audio cache not found: {self.path}")
        size_bytes = self.path.stat().st_size
        if size_bytes == 0 or size_bytes % _FLOAT32_BYTES != 0:
            raise AudioDecodeError(
                f"invalid float32 audio cache size ({size_bytes} bytes): {self.path}"
            )
        self.sample_rate_hz = YAMNET_SAMPLE_RATE_HZ
        self.channels = MONO_CHANNELS
        self._samples = np.memmap(self.path, dtype="<f4", mode="r")

    @property
    def sample_count(self) -> int:
        return int(self._samples.size)

    @property
    def duration_sec(self) -> float:
        return self.sample_count / self.sample_rate_hz

    def _read_samples(
        self,
        start_sec: float,
        end_sec: float,
    ) -> NDArray[np.float32]:
        start_sample = timestamp_to_sample_index(start_sec, self.sample_rate_hz)
        expected_count = _range_sample_count(start_sec, end_sec, self.sample_rate_hz)
        end_sample = start_sample + expected_count
        if end_sample > self.sample_count:
            raise AudioRangeError(
                "requested complete audio window exceeds normalized audio: "
                f"samples [{start_sample}, {end_sample}) requested, "
                f"{self.sample_count} available"
            )

        samples = np.array(
            self._samples[start_sample:end_sample],
            dtype=np.float32,
            copy=True,
        )
        if samples.size != expected_count:
            raise AudioRangeError(
                f"expected {expected_count} samples, read {samples.size}; "
                "partial audio is not allowed"
            )
        samples.setflags(write=False)
        return samples

    def slice_absolute(self, start_sec: float, end_sec: float) -> AudioSlice:
        """Read a complete identity-neutral range using absolute match timestamps."""

        return AudioSlice(
            start_sec=start_sec,
            end_sec=end_sec,
            sample_rate_hz=self.sample_rate_hz,
            channels=self.channels,
            samples=self._read_samples(start_sec, end_sec),
        )

    def slice_window(self, window: AnalysisWindow) -> AudioWindow:
        """Read exactly one planned inference window; never return partial audio."""

        return AudioWindow(
            segment_index=window.segment_index,
            start_sec=window.start_sec,
            end_sec=window.end_sec,
            sample_rate_hz=self.sample_rate_hz,
            channels=self.channels,
            samples=self._read_samples(window.start_sec, window.end_sec),
        )

    def close(self) -> None:
        memory_map = getattr(self._samples, "_mmap", None)
        if memory_map is not None:
            memory_map.close()

    def __enter__(self) -> NormalizedAudioSource:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class AudioNormalizer(Protocol):
    """Backend boundary for building one reusable normalized audio source."""

    def normalize(
        self,
        media_path: str | Path,
        cache_path: str | Path,
        *,
        rebuild: bool = False,
    ) -> NormalizedAudioSource:
        ...


@dataclass(frozen=True, slots=True)
class FFmpegAudioNormalizer:
    """Build a reusable mono 16 kHz float32 cache with one FFmpeg process."""

    ffmpeg_executable: str | None = None

    def normalize(
        self,
        media_path: str | Path,
        cache_path: str | Path,
        *,
        rebuild: bool = False,
    ) -> NormalizedAudioSource:
        media = Path(media_path)
        cache = Path(cache_path)
        if not media.is_file():
            raise FileNotFoundError(f"match media not found: {media}")
        if not rebuild and _cache_is_current(media, cache):
            return NormalizedAudioSource(cache)

        executable = self.ffmpeg_executable or shutil.which("ffmpeg")
        if executable is None:
            raise FFmpegNotFoundError(
                "FFmpeg executable not found on PATH; install FFmpeg or pass "
                "FFmpegAudioNormalizer(ffmpeg_executable=...)"
            )

        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.NamedTemporaryFile(
            prefix=f".{cache.name}.",
            suffix=".tmp",
            dir=cache.parent,
            delete=False,
        )
        temporary_path = Path(temporary.name)
        temporary.close()
        command = [
            executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(media),
            "-map",
            "0:a:0",
            "-vn",
            "-af",
            FFMPEG_NORMALIZATION_FILTER,
            "-ac",
            str(MONO_CHANNELS),
            "-ar",
            str(YAMNET_SAMPLE_RATE_HZ),
            "-c:a",
            "pcm_f32le",
            "-f",
            "f32le",
            str(temporary_path),
        ]
        try:
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    shell=False,
                )
            except OSError as exc:
                raise AudioDecodeError(f"failed to start FFmpeg: {exc}") from exc
            if completed.returncode != 0:
                detail = completed.stderr.strip() or "unknown FFmpeg error"
                raise AudioDecodeError(f"FFmpeg audio normalization failed: {detail}")
            size_bytes = temporary_path.stat().st_size
            if size_bytes == 0 or size_bytes % _FLOAT32_BYTES != 0:
                raise AudioDecodeError(
                    f"FFmpeg produced invalid float32 audio ({size_bytes} bytes)"
                )
            os.replace(temporary_path, cache)
            _write_cache_metadata(media, cache)
        finally:
            temporary_path.unlink(missing_ok=True)

        return NormalizedAudioSource(cache)


def _metadata_path(cache_path: Path) -> Path:
    return Path(f"{cache_path}.json")


def _source_signature(media_path: Path) -> dict[str, object]:
    stat = media_path.stat()
    return {
        "source_path": str(media_path.resolve()),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "sample_rate_hz": YAMNET_SAMPLE_RATE_HZ,
        "channels": MONO_CHANNELS,
        "sample_format": "float32le",
        "format_version": AUDIO_CACHE_FORMAT_VERSION,
    }


def _cache_is_current(media_path: Path, cache_path: Path) -> bool:
    if not cache_path.is_file():
        return False
    size_bytes = cache_path.stat().st_size
    if size_bytes == 0 or size_bytes % _FLOAT32_BYTES != 0:
        return False
    try:
        metadata = json.loads(_metadata_path(cache_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return metadata == _source_signature(media_path)


def _write_cache_metadata(media_path: Path, cache_path: Path) -> None:
    metadata_path = _metadata_path(cache_path)
    temporary_path = Path(f"{metadata_path}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(_source_signature(media_path), indent=2),
            encoding="utf-8",
        )
        os.replace(temporary_path, metadata_path)
    finally:
        temporary_path.unlink(missing_ok=True)
