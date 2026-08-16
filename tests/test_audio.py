from __future__ import annotations

import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

import audio_highlight.audio as audio_module
from audio_highlight.audio import (
    AudioRangeError,
    FFmpegAudioNormalizer,
    FFmpegNotFoundError,
    NormalizedAudioSource,
    timestamp_to_sample_index,
)
from audio_highlight.windows import AnalysisWindow

SAMPLE_RATE = 16_000


@pytest.fixture
def normalized_source(tmp_path: Path):
    waveform = np.linspace(-1.0, 1.0, 6 * SAMPLE_RATE, dtype=np.float32)
    path = tmp_path / "synthetic.f32le"
    waveform.tofile(path)
    source = NormalizedAudioSource(path)
    try:
        yield source, waveform
    finally:
        source.close()


def test_timestamp_to_sample_index() -> None:
    assert timestamp_to_sample_index(386.233) == 6_179_728
    assert timestamp_to_sample_index(0.00003125) == 0
    assert timestamp_to_sample_index(0.00009375) == 2


def test_three_seconds_at_16khz_has_48000_samples(normalized_source) -> None:
    source, _ = normalized_source

    audio = source.slice_window(AnalysisWindow(0, 1.0, 4.0))

    assert audio.samples.size == 48_000


def test_absolute_timestamp_slicing(normalized_source) -> None:
    source, waveform = normalized_source

    audio = source.slice_window(AnalysisWindow(4, 2.0, 5.0))

    np.testing.assert_array_equal(audio.samples, waveform[32_000:80_000])


def test_output_is_mono_float32_at_16khz(normalized_source) -> None:
    source, _ = normalized_source

    audio = source.slice_window(AnalysisWindow(0, 0.0, 3.0))

    assert audio.channels == 1
    assert audio.samples.ndim == 1
    assert audio.samples.dtype == np.float32
    assert audio.sample_rate_hz == SAMPLE_RATE
    assert not audio.samples.flags.writeable


def test_multiple_windows_share_one_normalized_source(normalized_source) -> None:
    source, _ = normalized_source

    first = source.slice_window(AnalysisWindow(0, 0.0, 3.0))
    second = source.slice_window(AnalysisWindow(1, 1.0, 4.0))

    assert source.sample_count == 96_000
    assert first.samples.size == second.samples.size == 48_000


def test_float_timestamp_drift_does_not_change_sample_count(normalized_source) -> None:
    source, _ = normalized_source
    start = 0.1 + 0.2

    audio = source.slice_window(AnalysisWindow(0, start, start + 3.0))

    assert audio.samples.size == 48_000
    assert audio.start_sec == start


def test_requested_range_beyond_audio_raises(normalized_source) -> None:
    source, _ = normalized_source

    with pytest.raises(AudioRangeError, match="exceeds normalized audio"):
        source.slice_window(AnalysisWindow(0, 4.0, 7.0))


def test_partial_waveform_is_never_returned(normalized_source) -> None:
    source, _ = normalized_source

    with pytest.raises(AudioRangeError):
        source.slice_window(AnalysisWindow(0, 5.0, 8.0))


def test_segment_index_is_preserved(normalized_source) -> None:
    source, _ = normalized_source

    audio = source.slice_window(AnalysisWindow(19, 0.0, 3.0))

    assert audio.segment_index == 19


def test_overlapping_windows_read_overlapping_samples(normalized_source) -> None:
    source, _ = normalized_source

    first = source.slice_window(AnalysisWindow(0, 1.0, 4.0))
    second = source.slice_window(AnalysisWindow(1, 2.0, 5.0))

    np.testing.assert_array_equal(first.samples[16_000:], second.samples[:-16_000])


def test_analysis_window_is_not_modified(normalized_source) -> None:
    source, _ = normalized_source
    window = AnalysisWindow(7, 1.0, 4.0)
    original = asdict(window)

    source.slice_window(window)

    assert asdict(window) == original


def test_ffmpeg_normalization_runs_once_then_reuses_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media = tmp_path / "match.mp4"
    media.write_bytes(b"synthetic media placeholder")
    cache = tmp_path / "match.f32le"
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        np.zeros(3 * SAMPLE_RATE, dtype=np.float32).tofile(command[-1])
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(audio_module.shutil, "which", lambda _: "ffmpeg")
    monkeypatch.setattr(audio_module.subprocess, "run", fake_run)
    normalizer = FFmpegAudioNormalizer()

    first = normalizer.normalize(media, cache)
    first.close()
    second = normalizer.normalize(media, cache)
    second.close()

    assert len(calls) == 1
    assert Path(f"{cache}.json").is_file()
    assert calls[0][calls[0].index("-ac") + 1] == "1"
    assert calls[0][calls[0].index("-ar") + 1] == "16000"
    assert calls[0][calls[0].index("-f") + 1] == "f32le"
    assert calls[0][calls[0].index("-af") + 1] == (
        "aformat=sample_rates=16000:channel_layouts=mono,"
        "asoftclip=type=hard:threshold=1"
    )


def test_missing_ffmpeg_has_clear_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media = tmp_path / "match.mp4"
    media.write_bytes(b"synthetic media placeholder")
    monkeypatch.setattr(audio_module.shutil, "which", lambda _: None)

    with pytest.raises(FFmpegNotFoundError, match="not found on PATH"):
        FFmpegAudioNormalizer().normalize(media, tmp_path / "match.f32le")
