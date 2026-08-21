"""Read-only hard-clipping audit over canonical normalized match audio."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from audio_highlight.audio import (
    AUDIO_CACHE_FORMAT_VERSION,
    FFMPEG_NORMALIZATION_FILTER,
    FFMPEG_PRECLIP_FILTER,
    MONO_CHANNELS,
    YAMNET_SAMPLE_RATE_HZ,
    AudioDecodeError,
    FFmpegNotFoundError,
    NormalizedAudioSource,
)
from audio_highlight.contracts import load_segments_artifact
from audio_highlight.labeling import (
    LabelStore,
    build_candidate_windows,
    load_manifest,
    segments_sha256,
)
from audio_highlight.windows import InferenceConfig

EXPERIMENT_ID = "clipping_v1"
RMS_EPSILON = 1e-12
RAIL_THRESHOLDS = (0.0001, 0.001, 0.01)
LOUDNESS_GROUPS = (
    ("top_1_percent", 0.01),
    ("top_5_percent", 0.05),
    ("top_10_percent", 0.10),
)
_FLOAT32_BYTES = np.dtype("<f4").itemsize
_SCAN_CHUNK_SAMPLES = 1_000_000


class ClippingAuditError(ValueError):
    """Raised when clipping inputs, alignment, or diagnostic state is invalid."""


@dataclass(frozen=True, slots=True)
class RailStatistics:
    total_samples: int
    positive_rail_samples: int
    negative_rail_samples: int
    rail_samples: int
    rail_ratio: float
    max_abs: float
    clipped_run_count: int
    max_clipped_run_samples: int
    max_clipped_run_ms: float


@dataclass(frozen=True, slots=True)
class PreclipStatistics:
    total_samples: int
    max_abs: float
    samples_abs_gt_1: int
    samples_abs_gt_1_01: int
    samples_abs_gt_1_05: int
    samples_abs_gt_1_10: int
    overflow_ratio_gt_1: float
    max_excess: float


@dataclass(frozen=True, slots=True)
class WindowAudit:
    match_id: str
    segment_index: int
    window_index_in_segment: int
    start_sec: float
    end_sec: float
    rail_sample_count: int
    rail_ratio: float
    has_any_rail: bool
    max_abs: float
    rms: float
    log_rms_db: float
    preclip_max_abs: float | None
    preclip_overflow_count: int | None
    preclip_overflow_ratio: float | None
    label: int | None


@dataclass(frozen=True, slots=True)
class MatchClippingAudit:
    match_id: str
    whole_match: dict[str, Any]
    candidate_windows: dict[str, Any]
    labeled_stratification: dict[str, Any]
    loudness_concentration: dict[str, Any]
    windows: tuple[WindowAudit, ...]
    top_windows: tuple[WindowAudit, ...]
    input_artifacts: dict[str, dict[str, str]]


@dataclass(frozen=True, slots=True)
class ClippingAuditResult:
    experiment_id: str
    matches: tuple[MatchClippingAudit, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ClippingAuditArtifactPaths:
    summary_csv: Path
    metrics_json: Path
    window_audit_csv: Path
    top_clipped_windows_csv: Path
    metadata_json: Path
    clipping_by_match_plot: Path
    clipping_vs_rms_plots: tuple[Path, ...]
    labeled_plot: Path | None


def compute_rail_statistics(
    samples: object,
    *,
    sample_rate_hz: int = YAMNET_SAMPLE_RATE_HZ,
    chunk_samples: int = _SCAN_CHUNK_SAMPLES,
) -> RailStatistics:
    """Count exact float rails and consecutive runs without tolerance."""

    values = np.asarray(samples)
    if values.ndim != 1 or values.size == 0:
        raise ClippingAuditError("rail audit requires a non-empty 1-D waveform")
    if not np.issubdtype(values.dtype, np.floating):
        raise ClippingAuditError("rail audit waveform must be floating point")
    if sample_rate_hz <= 0 or chunk_samples <= 0:
        raise ClippingAuditError("sample rate and scan chunk size must be positive")

    positive = 0
    negative = 0
    maximum = 0.0
    run_count = 0
    max_run = 0
    trailing_run = 0
    previous_ended_on_rail = False
    for start in range(0, values.size, chunk_samples):
        chunk = np.asarray(values[start : start + chunk_samples])
        if not np.isfinite(chunk).all():
            raise ClippingAuditError("rail audit waveform contains NaN or Inf")
        positive_mask = chunk == np.float32(1.0)
        negative_mask = chunk == np.float32(-1.0)
        positive += int(np.count_nonzero(positive_mask))
        negative += int(np.count_nonzero(negative_mask))
        maximum = max(maximum, float(np.max(np.abs(chunk))))
        rail_mask = positive_mask | negative_mask
        lengths, starts_on_rail, ends_on_rail = _boolean_run_lengths(rail_mask)
        if lengths.size:
            local_lengths = lengths.astype(np.int64, copy=True)
            if previous_ended_on_rail and starts_on_rail:
                local_lengths[0] += trailing_run
                run_count += int(local_lengths.size - 1)
            else:
                run_count += int(local_lengths.size)
            max_run = max(max_run, int(np.max(local_lengths)))
            if ends_on_rail:
                trailing_run = int(local_lengths[-1])
            else:
                trailing_run = 0
        else:
            trailing_run = 0
        previous_ended_on_rail = ends_on_rail

    rail_samples = positive + negative
    total = int(values.size)
    return RailStatistics(
        total_samples=total,
        positive_rail_samples=positive,
        negative_rail_samples=negative,
        rail_samples=rail_samples,
        rail_ratio=rail_samples / total,
        max_abs=maximum,
        clipped_run_count=run_count,
        max_clipped_run_samples=max_run,
        max_clipped_run_ms=max_run * 1000.0 / sample_rate_hz,
    )


def compute_preclip_statistics(
    samples: object,
    *,
    chunk_samples: int = _SCAN_CHUNK_SAMPLES,
) -> PreclipStatistics:
    """Measure overflow severity before the production hard-clipping filter."""

    values = np.asarray(samples)
    if values.ndim != 1 or values.size == 0:
        raise ClippingAuditError("preclip audit requires a non-empty 1-D waveform")
    if not np.issubdtype(values.dtype, np.floating) or chunk_samples <= 0:
        raise ClippingAuditError("preclip waveform and chunk size are invalid")
    counts = {threshold: 0 for threshold in (1.0, 1.01, 1.05, 1.10)}
    maximum = 0.0
    for start in range(0, values.size, chunk_samples):
        chunk = np.asarray(values[start : start + chunk_samples])
        if not np.isfinite(chunk).all():
            raise ClippingAuditError("preclip waveform contains NaN or Inf")
        absolute = np.abs(chunk)
        maximum = max(maximum, float(np.max(absolute)))
        for threshold in counts:
            counts[threshold] += int(np.count_nonzero(absolute > threshold))
    total = int(values.size)
    return PreclipStatistics(
        total_samples=total,
        max_abs=maximum,
        samples_abs_gt_1=counts[1.0],
        samples_abs_gt_1_01=counts[1.01],
        samples_abs_gt_1_05=counts[1.05],
        samples_abs_gt_1_10=counts[1.10],
        overflow_ratio_gt_1=counts[1.0] / total,
        max_excess=max(maximum - 1.0, 0.0),
    )


def waveform_rms(samples: object) -> tuple[float, float]:
    """Return RMS and finite log-RMS dB for one canonical waveform."""

    values = np.asarray(samples)
    if (
        values.ndim != 1
        or values.size == 0
        or not np.issubdtype(values.dtype, np.floating)
        or not np.isfinite(values).all()
    ):
        raise ClippingAuditError("RMS requires a non-empty finite waveform")
    floating = np.asarray(values, dtype=np.float64)
    rms = math.sqrt(float(np.mean(floating * floating)))
    return rms, 20.0 * math.log10(rms + RMS_EPSILON)


def summarize_candidate_windows(windows: Sequence[WindowAudit]) -> dict[str, Any]:
    if not windows:
        raise ClippingAuditError("candidate-window audit must not be empty")
    ratios = np.asarray([item.rail_ratio for item in windows], dtype=np.float64)
    nonzero = ratios[ratios > 0]
    overflow = [
        item.preclip_overflow_ratio
        for item in windows
        if item.preclip_overflow_ratio is not None
    ]
    return {
        "total_window_count": len(windows),
        "windows_with_any_rail": int(np.count_nonzero(ratios > 0)),
        "fraction_windows_with_any_rail": float(np.mean(ratios > 0)),
        "windows_rail_ratio_ge_0_0001": int(np.count_nonzero(ratios >= 0.0001)),
        "windows_rail_ratio_ge_0_001": int(np.count_nonzero(ratios >= 0.001)),
        "windows_rail_ratio_ge_0_01": int(np.count_nonzero(ratios >= 0.01)),
        "max_window_rail_ratio": float(np.max(ratios)),
        "median_nonzero_window_rail_ratio": (
            float(np.median(nonzero)) if nonzero.size else None
        ),
        "p95_nonzero_window_rail_ratio": (
            float(np.quantile(nonzero, 0.95)) if nonzero.size else None
        ),
        "fraction_windows_with_preclip_overflow": (
            float(np.mean(np.asarray(overflow) > 0)) if overflow else None
        ),
    }


def summarize_labeled_windows(windows: Sequence[WindowAudit]) -> dict[str, Any]:
    labeled = [item for item in windows if item.label is not None]
    if not labeled or any(item.label not in {0, 1} for item in labeled):
        raise ClippingAuditError("labeled-window audit requires binary labels")
    return {
        "cheer": _labeled_group([item for item in labeled if item.label == 1]),
        "no_cheer": _labeled_group([item for item in labeled if item.label == 0]),
    }


def summarize_loudness_concentration(
    windows: Sequence[WindowAudit],
) -> dict[str, Any]:
    if not windows:
        raise ClippingAuditError("loudness concentration requires windows")
    ordered = sorted(
        windows,
        key=lambda item: (
            -item.log_rms_db,
            item.segment_index,
            item.window_index_in_segment,
            item.start_sec,
        ),
    )
    groups: dict[str, Any] = {}
    for name, fraction in LOUDNESS_GROUPS:
        count = max(1, math.ceil(len(ordered) * fraction))
        groups[name] = _unlabeled_group(ordered[:count])
    groups["all_windows"] = _unlabeled_group(ordered)
    return groups


def select_top_clipped_windows(
    windows: Sequence[WindowAudit], *, limit: int = 20
) -> tuple[WindowAudit, ...]:
    if limit <= 0:
        raise ClippingAuditError("top clipping limit must be positive")
    preclip_available = all(
        item.preclip_overflow_ratio is not None and item.preclip_max_abs is not None
        for item in windows
    )
    if preclip_available:
        ordered = sorted(
            windows,
            key=lambda item: (
                -float(item.preclip_overflow_ratio),
                -float(item.preclip_max_abs),
                -item.rail_ratio,
                item.segment_index,
                item.window_index_in_segment,
                item.start_sec,
            ),
        )
    else:
        ordered = sorted(
            windows,
            key=lambda item: (
                -item.rail_ratio,
                item.segment_index,
                item.window_index_in_segment,
                item.start_sec,
            ),
        )
    return tuple(ordered[:limit])


def run_clipping_audit(
    matches: Sequence[str],
    *,
    artifact_root: str | Path = "artifacts",
    local_data_root: str | Path = "local_data",
    ffmpeg_executable: str | None = None,
) -> ClippingAuditResult:
    """Run post-cache and temporary preclip audits without replacing inputs."""

    match_ids = tuple(matches)
    if (
        not match_ids
        or any(not match_id for match_id in match_ids)
        or len(set(match_ids)) != len(match_ids)
    ):
        raise ClippingAuditError("matches must be unique non-empty identifiers")
    artifacts = Path(artifact_root)
    local_data = Path(local_data_root)
    executable = ffmpeg_executable or shutil.which("ffmpeg")
    if executable is None:
        raise FFmpegNotFoundError("FFmpeg is required for preclip dry-run audit")
    ffmpeg_version = _ffmpeg_version(executable)
    planner = InferenceConfig()
    results: list[MatchClippingAudit] = []

    with tempfile.TemporaryDirectory(prefix="audio-highlight-clipping-") as temporary:
        temporary_root = Path(temporary)
        for match_id in match_ids:
            cache_path = artifacts / match_id / "audio" / "audio.f32le"
            labels_path = artifacts / match_id / "labeling" / "labels.csv"
            manifest_path = artifacts / match_id / "labeling" / "sample_manifest.json"
            media_path = local_data / match_id / "match.mp4"
            segments_path = _resolve_segments_path(local_data / match_id)
            source = NormalizedAudioSource(cache_path)
            preclip_path = temporary_root / f"{match_id}.preclip.f32le"
            _decode_preclip(media_path, preclip_path, executable)
            preclip_source = NormalizedAudioSource(preclip_path)
            try:
                if preclip_source.sample_count != source.sample_count:
                    raise ClippingAuditError(
                        f"{match_id}: post-cache and preclip sample counts differ"
                    )
                post_values = np.memmap(cache_path, dtype="<f4", mode="r")
                preclip_values = np.memmap(preclip_path, dtype="<f4", mode="r")
                try:
                    rail = compute_rail_statistics(post_values)
                    preclip = compute_preclip_statistics(preclip_values)
                finally:
                    _close_memmap(post_values)
                    _close_memmap(preclip_values)
                artifact = load_segments_artifact(segments_path)
                candidates = build_candidate_windows(
                    artifact,
                    planner,
                    media_duration_sec=source.duration_sec,
                )
                manifest = load_manifest(manifest_path)
                if manifest.match_id != match_id or manifest.planner != planner:
                    raise ClippingAuditError(
                        f"{match_id}: manifest identity or planner mismatch"
                    )
                if manifest.segments_sha256 != segments_sha256(segments_path):
                    raise ClippingAuditError(
                        f"{match_id}: manifest and segments SHA-256 differ"
                    )
                if manifest.candidate_window_count != len(candidates):
                    raise ClippingAuditError(
                        f"{match_id}: manifest candidate count mismatch"
                    )
                decisions = LabelStore(labels_path, manifest).decisions
                if len(decisions) != manifest.sample_size:
                    raise ClippingAuditError(f"{match_id}: labels are incomplete")
                label_map = {
                    (
                        decision.segment_index,
                        decision.window_index_in_segment,
                        decision.window_start_sec,
                        decision.window_end_sec,
                    ): decision.has_cheer
                    for decision in decisions.values()
                    if not decision.is_ambiguous
                }
                windows: list[WindowAudit] = []
                for candidate in candidates:
                    post_audio = source.slice_absolute(
                        candidate.start_sec, candidate.end_sec
                    )
                    preclip_audio = preclip_source.slice_absolute(
                        candidate.start_sec, candidate.end_sec
                    )
                    post_window = compute_rail_statistics(post_audio.samples)
                    preclip_window = compute_preclip_statistics(preclip_audio.samples)
                    rms, log_rms_db = waveform_rms(post_audio.samples)
                    identity = (
                        candidate.segment_index,
                        candidate.window_index_in_segment,
                        candidate.start_sec,
                        candidate.end_sec,
                    )
                    windows.append(
                        WindowAudit(
                            match_id=match_id,
                            segment_index=candidate.segment_index,
                            window_index_in_segment=(
                                candidate.window_index_in_segment
                            ),
                            start_sec=candidate.start_sec,
                            end_sec=candidate.end_sec,
                            rail_sample_count=post_window.rail_samples,
                            rail_ratio=post_window.rail_ratio,
                            has_any_rail=post_window.rail_samples > 0,
                            max_abs=post_window.max_abs,
                            rms=rms,
                            log_rms_db=log_rms_db,
                            preclip_max_abs=preclip_window.max_abs,
                            preclip_overflow_count=(
                                preclip_window.samples_abs_gt_1
                            ),
                            preclip_overflow_ratio=(
                                preclip_window.overflow_ratio_gt_1
                            ),
                            label=label_map.get(identity),
                        )
                    )
                labeled_count = sum(item.label is not None for item in windows)
                if labeled_count != len(label_map):
                    raise ClippingAuditError(
                        f"{match_id}: labeled identities do not align to candidates"
                    )
                window_tuple = tuple(windows)
                whole = {
                    "duration_sec": source.duration_sec,
                    **_prefixed_rail_dict(rail),
                    **_prefixed_preclip_dict(preclip),
                }
                inputs = {
                    "normalized_audio_cache": _artifact_record(cache_path),
                    "segments": _artifact_record(segments_path),
                    "labels": _artifact_record(labels_path),
                    "sample_manifest": _artifact_record(manifest_path),
                }
                results.append(
                    MatchClippingAudit(
                        match_id=match_id,
                        whole_match=whole,
                        candidate_windows=summarize_candidate_windows(window_tuple),
                        labeled_stratification=summarize_labeled_windows(window_tuple),
                        loudness_concentration=(
                            summarize_loudness_concentration(window_tuple)
                        ),
                        windows=window_tuple,
                        top_windows=select_top_clipped_windows(window_tuple),
                        input_artifacts=inputs,
                    )
                )
            finally:
                preclip_source.close()
                source.close()

    metadata = {
        "experiment_id": EXPERIMENT_ID,
        "sample_rate_hz": YAMNET_SAMPLE_RATE_HZ,
        "channels": MONO_CHANNELS,
        "sample_format": "float32le",
        "audio_cache_format_version": AUDIO_CACHE_FORMAT_VERSION,
        "canonical_cache_waveform": "post_clipped",
        "stereo_to_mono_layer": "FFmpeg aformat channel_layouts=mono",
        "actual_clipping_rule": "FFmpeg asoftclip=type=hard:threshold=1",
        "canonical_ffmpeg_filter": FFMPEG_NORMALIZATION_FILTER,
        "rail_detection_rule": "exact_float32_equality_x_eq_plus_or_minus_1",
        "rail_detection_tolerance": None,
        "preclip_measurement_available": True,
        "preclip_dry_run_filter": FFMPEG_PRECLIP_FILTER,
        "preclip_dry_run_output": "temporary_pcm_f32le_not_persisted",
        "ffmpeg_version": ffmpeg_version,
        "analysis_planner": asdict(planner),
        "matches": list(match_ids),
        "input_artifacts": {
            result.match_id: result.input_artifacts for result in results
        },
    }
    return ClippingAuditResult(EXPERIMENT_ID, tuple(results), metadata)


def write_clipping_audit_artifacts(
    result: ClippingAuditResult,
    output_dir: str | Path,
) -> ClippingAuditArtifactPaths:
    """Write audit tables, JSON, and diagnostic plots without touching inputs."""

    output = Path(output_dir)
    _validate_output_path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.csv"
    metrics_path = output / "metrics.json"
    windows_path = output / "window_audit.csv"
    top_path = output / "top_clipped_windows.csv"
    metadata_path = output / "metadata.json"
    _write_summary(result, summary_path)
    _write_json(_metrics_dict(result), metrics_path)
    _write_windows(
        tuple(window for match in result.matches for window in match.windows),
        windows_path,
    )
    _write_windows(
        tuple(window for match in result.matches for window in match.top_windows),
        top_path,
    )
    _write_json(result.metadata, metadata_path)
    match_plot, rms_plots, labeled_plot = _write_plots(result, output)
    return ClippingAuditArtifactPaths(
        summary_csv=summary_path,
        metrics_json=metrics_path,
        window_audit_csv=windows_path,
        top_clipped_windows_csv=top_path,
        metadata_json=metadata_path,
        clipping_by_match_plot=match_plot,
        clipping_vs_rms_plots=rms_plots,
        labeled_plot=labeled_plot,
    )


def _boolean_run_lengths(
    mask: NDArray[np.bool_],
) -> tuple[NDArray[np.int64], bool, bool]:
    if mask.size == 0:
        return np.asarray([], dtype=np.int64), False, False
    padded = np.empty(mask.size + 2, dtype=np.bool_)
    padded[0] = False
    padded[1:-1] = mask
    padded[-1] = False
    transitions = np.flatnonzero(padded[1:] != padded[:-1])
    lengths = (transitions[1::2] - transitions[0::2]).astype(np.int64)
    return lengths, bool(mask[0]), bool(mask[-1])


def _labeled_group(windows: Sequence[WindowAudit]) -> dict[str, Any]:
    if not windows:
        raise ClippingAuditError("both cheer and no-cheer groups must be non-empty")
    ratios = np.asarray([item.rail_ratio for item in windows], dtype=np.float64)
    log_rms = np.asarray([item.log_rms_db for item in windows], dtype=np.float64)
    overflow = np.asarray(
        [float(item.preclip_overflow_ratio) for item in windows], dtype=np.float64
    )
    peaks = np.asarray(
        [float(item.preclip_max_abs) for item in windows], dtype=np.float64
    )
    return {
        "sample_count": len(windows),
        "windows_with_any_rail": int(np.count_nonzero(ratios > 0)),
        "fraction_with_any_rail": float(np.mean(ratios > 0)),
        "mean_rail_ratio": float(np.mean(ratios)),
        "median_rail_ratio": float(np.median(ratios)),
        "max_rail_ratio": float(np.max(ratios)),
        "mean_log_rms_db": float(np.mean(log_rms)),
        "median_log_rms_db": float(np.median(log_rms)),
        "fraction_with_overflow": float(np.mean(overflow > 0)),
        "mean_preclip_overflow_ratio": float(np.mean(overflow)),
        "max_preclip_overflow_ratio": float(np.max(overflow)),
        "max_preclip_peak": float(np.max(peaks)),
    }


def _unlabeled_group(windows: Sequence[WindowAudit]) -> dict[str, Any]:
    ratios = np.asarray([item.rail_ratio for item in windows], dtype=np.float64)
    overflow = np.asarray(
        [float(item.preclip_overflow_ratio) for item in windows], dtype=np.float64
    )
    peaks = np.asarray(
        [float(item.preclip_max_abs) for item in windows], dtype=np.float64
    )
    return {
        "window_count": len(windows),
        "fraction_with_any_rail": float(np.mean(ratios > 0)),
        "mean_rail_ratio": float(np.mean(ratios)),
        "max_rail_ratio": float(np.max(ratios)),
        "fraction_with_overflow": float(np.mean(overflow > 0)),
        "mean_overflow_ratio": float(np.mean(overflow)),
        "max_preclip_peak": float(np.max(peaks)),
    }


def _prefixed_rail_dict(value: RailStatistics) -> dict[str, Any]:
    return {
        "total_samples": value.total_samples,
        "positive_rail_samples": value.positive_rail_samples,
        "negative_rail_samples": value.negative_rail_samples,
        "rail_samples": value.rail_samples,
        "rail_ratio": value.rail_ratio,
        "post_clip_max_abs": value.max_abs,
        "clipped_run_count": value.clipped_run_count,
        "max_clipped_run_samples": value.max_clipped_run_samples,
        "max_clipped_run_ms": value.max_clipped_run_ms,
    }


def _prefixed_preclip_dict(value: PreclipStatistics) -> dict[str, Any]:
    return {
        "preclip_total_samples": value.total_samples,
        "preclip_max_abs": value.max_abs,
        "samples_abs_gt_1": value.samples_abs_gt_1,
        "samples_abs_gt_1_01": value.samples_abs_gt_1_01,
        "samples_abs_gt_1_05": value.samples_abs_gt_1_05,
        "samples_abs_gt_1_10": value.samples_abs_gt_1_10,
        "overflow_ratio_gt_1": value.overflow_ratio_gt_1,
        "max_excess": value.max_excess,
    }


def _decode_preclip(media: Path, output: Path, executable: str) -> None:
    if not media.is_file():
        raise FileNotFoundError(f"match media not found: {media}")
    output.parent.mkdir(parents=True, exist_ok=True)
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
        FFMPEG_PRECLIP_FILTER,
        "-ac",
        str(MONO_CHANNELS),
        "-ar",
        str(YAMNET_SAMPLE_RATE_HZ),
        "-c:a",
        "pcm_f32le",
        "-f",
        "f32le",
        str(output),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            shell=False,
        )
    except OSError as exc:
        raise AudioDecodeError(f"failed to start FFmpeg preclip dry-run: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "unknown FFmpeg error"
        raise AudioDecodeError(f"FFmpeg preclip dry-run failed: {detail}")
    size = output.stat().st_size
    if size == 0 or size % _FLOAT32_BYTES != 0:
        raise AudioDecodeError(f"FFmpeg produced invalid preclip audio: {size} bytes")


def _ffmpeg_version(executable: str) -> str:
    try:
        completed = subprocess.run(
            [executable, "-version"],
            check=False,
            capture_output=True,
            text=True,
            shell=False,
        )
    except OSError as exc:
        raise AudioDecodeError(f"failed to inspect FFmpeg version: {exc}") from exc
    if completed.returncode != 0:
        raise AudioDecodeError("FFmpeg version inspection failed")
    return completed.stdout.splitlines()[0].strip()


def _resolve_segments_path(match_directory: Path) -> Path:
    canonical = match_directory / "segments.json"
    if canonical.is_file():
        return canonical
    candidates = sorted(match_directory.glob("*.json"))
    if len(candidates) == 1:
        load_segments_artifact(candidates[0])
        return candidates[0]
    raise FileNotFoundError(
        f"cannot uniquely resolve segments JSON in {match_directory}"
    )


def _metrics_dict(result: ClippingAuditResult) -> dict[str, Any]:
    return {
        "experiment_id": result.experiment_id,
        "matches": {
            item.match_id: {
                "whole_match": item.whole_match,
                "candidate_windows": item.candidate_windows,
                "labeled_stratification": item.labeled_stratification,
                "loudness_concentration": item.loudness_concentration,
            }
            for item in result.matches
        },
    }


def _write_summary(result: ClippingAuditResult, path: Path) -> None:
    fieldnames = (
        "match_id",
        "duration_sec",
        "total_samples",
        "rail_samples",
        "rail_ratio",
        "post_clip_max_abs",
        "candidate_window_count",
        "windows_with_any_rail",
        "fraction_windows_with_any_rail",
        "windows_rail_ratio_ge_0_0001",
        "windows_rail_ratio_ge_0_001",
        "windows_rail_ratio_ge_0_01",
        "max_window_rail_ratio",
        "cheer_sample_count",
        "cheer_fraction_with_any_rail",
        "no_cheer_sample_count",
        "no_cheer_fraction_with_any_rail",
        "preclip_max_abs",
        "overflow_ratio",
        "max_excess",
        "fraction_windows_with_preclip_overflow",
        "top_1_percent_fraction_with_any_rail",
        "top_1_percent_fraction_with_overflow",
    )
    rows: list[dict[str, Any]] = []
    for item in result.matches:
        whole = item.whole_match
        candidate = item.candidate_windows
        cheer = item.labeled_stratification["cheer"]
        no_cheer = item.labeled_stratification["no_cheer"]
        top = item.loudness_concentration["top_1_percent"]
        rows.append(
            {
                "match_id": item.match_id,
                "duration_sec": whole["duration_sec"],
                "total_samples": whole["total_samples"],
                "rail_samples": whole["rail_samples"],
                "rail_ratio": whole["rail_ratio"],
                "post_clip_max_abs": whole["post_clip_max_abs"],
                "candidate_window_count": candidate["total_window_count"],
                "windows_with_any_rail": candidate["windows_with_any_rail"],
                "fraction_windows_with_any_rail": candidate[
                    "fraction_windows_with_any_rail"
                ],
                "windows_rail_ratio_ge_0_0001": candidate[
                    "windows_rail_ratio_ge_0_0001"
                ],
                "windows_rail_ratio_ge_0_001": candidate[
                    "windows_rail_ratio_ge_0_001"
                ],
                "windows_rail_ratio_ge_0_01": candidate[
                    "windows_rail_ratio_ge_0_01"
                ],
                "max_window_rail_ratio": candidate["max_window_rail_ratio"],
                "cheer_sample_count": cheer["sample_count"],
                "cheer_fraction_with_any_rail": cheer["fraction_with_any_rail"],
                "no_cheer_sample_count": no_cheer["sample_count"],
                "no_cheer_fraction_with_any_rail": no_cheer[
                    "fraction_with_any_rail"
                ],
                "preclip_max_abs": whole["preclip_max_abs"],
                "overflow_ratio": whole["overflow_ratio_gt_1"],
                "max_excess": whole["max_excess"],
                "fraction_windows_with_preclip_overflow": candidate[
                    "fraction_windows_with_preclip_overflow"
                ],
                "top_1_percent_fraction_with_any_rail": top[
                    "fraction_with_any_rail"
                ],
                "top_1_percent_fraction_with_overflow": top[
                    "fraction_with_overflow"
                ],
            }
        )
    _write_csv(rows, fieldnames, path)


def _write_windows(windows: Sequence[WindowAudit], path: Path) -> None:
    rows = [asdict(item) for item in windows]
    _write_csv(rows, tuple(WindowAudit.__dataclass_fields__), path)


def _write_csv(
    rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str], path: Path
) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(value: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)
            file.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_plots(
    result: ClippingAuditResult, output: Path
) -> tuple[Path, tuple[Path, ...], Path | None]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    match_plot = output / "clipping_by_match.png"
    names = [item.match_id for item in result.matches]
    rail = [item.whole_match["rail_ratio"] for item in result.matches]
    overflow = [
        item.whole_match["overflow_ratio_gt_1"] for item in result.matches
    ]
    x = np.arange(len(names))
    figure, axis = plt.subplots(figsize=(8, 4.8))
    width = 0.38
    axis.bar(x - width / 2, rail, width, label="post-cache rail ratio")
    axis.bar(x + width / 2, overflow, width, label="preclip overflow ratio")
    axis.set_xticks(x, names)
    axis.set_yscale("symlog", linthresh=1e-9)
    axis.set_ylabel("Sample ratio (symlog; zero retained)")
    axis.set_title("Whole-match clipping by match")
    axis.legend()
    figure.tight_layout()
    _save_figure(figure, match_plot)
    plt.close(figure)

    rms_plots: list[Path] = []
    for item in result.matches:
        path = output / f"{item.match_id}_clipping_vs_rms.png"
        figure, axis = plt.subplots(figsize=(6.5, 4.5))
        axis.scatter(
            [window.log_rms_db for window in item.windows],
            [window.preclip_overflow_ratio for window in item.windows],
            s=12,
            alpha=0.6,
        )
        axis.set_yscale("symlog", linthresh=1e-9)
        axis.set(
            xlabel="Canonical log RMS (dB)",
            ylabel="Preclip overflow ratio (symlog; zero retained)",
            title=f"Clipping vs RMS | {item.match_id}",
        )
        figure.tight_layout()
        _save_figure(figure, path)
        plt.close(figure)
        rms_plots.append(path)

    has_labeled_clipping = any(
        group["fraction_with_any_rail"] > 0
        for item in result.matches
        for group in item.labeled_stratification.values()
    )
    labeled_path: Path | None = None
    if has_labeled_clipping:
        labeled_path = output / "labeled_cheer_vs_no_cheer_clipping.png"
        cheer = [
            item.labeled_stratification["cheer"]["fraction_with_any_rail"]
            for item in result.matches
        ]
        no_cheer = [
            item.labeled_stratification["no_cheer"]["fraction_with_any_rail"]
            for item in result.matches
        ]
        figure, axis = plt.subplots(figsize=(8, 4.8))
        axis.bar(x - width / 2, cheer, width, label="cheer")
        axis.bar(x + width / 2, no_cheer, width, label="no cheer")
        axis.set_xticks(x, names)
        axis.set_ylabel("Fraction with any post-cache rail sample")
        axis.set_title("Labeled-window clipping")
        axis.legend()
        figure.tight_layout()
        _save_figure(figure, labeled_path)
        plt.close(figure)
    return match_plot, tuple(rms_plots), labeled_path


def _save_figure(figure: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    try:
        figure.savefig(temporary, dpi=150, format="png")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _artifact_record(path: Path) -> dict[str, str]:
    return {"path": _portable_path(path), "sha256": _sha256(path)}


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.name


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ClippingAuditError(f"cannot hash input artifact {path}: {exc}") from exc
    return digest.hexdigest()


def _validate_output_path(path: Path) -> None:
    parts = tuple(part.lower() for part in path.resolve(strict=False).parts)
    forbidden = {"features", "models", "evaluation", "external_validation", "labeling"}
    if any(part in forbidden for part in parts) or "baseline_comparison" in parts:
        raise ClippingAuditError(
            "clipping audit output must be separate from frozen artifacts"
        )


def _close_memmap(values: np.memmap[Any, Any]) -> None:
    memory_map = getattr(values, "_mmap", None)
    if memory_map is not None:
        memory_map.close()
