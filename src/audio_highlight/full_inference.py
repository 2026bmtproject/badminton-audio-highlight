"""Production-style full-match inference and structural sampling diagnostics."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from audio_highlight.audio import (
    AUDIO_CACHE_FORMAT_VERSION,
    AudioNormalizer,
    AudioSlice,
    FFmpegAudioNormalizer,
    YAMNET_SAMPLE_RATE_HZ,
)
from audio_highlight.classifier import ExportedCheerDetector
from audio_highlight.contracts import load_segments_artifact
from audio_highlight.labeling import (
    CandidateWindow,
    SampleManifest,
    build_candidate_windows,
    load_manifest,
    segments_sha256,
)
from audio_highlight.windows import InferenceConfig
from audio_highlight.yamnet import (
    YAMNET_EMBEDDING_SIZE,
    YAMNET_MODEL_HANDLE,
    YamNetEmbeddingExtractor,
)

PROBABILITY_RTOL = 1e-9
PROBABILITY_ATOL = 1e-12
RELATIVE_POSITION_BIN_EDGES = tuple(index / 10 for index in range(11))


class FullMatchInferenceError(ValueError):
    """Raised when deployment inference or its provenance is inconsistent."""


class WindowEmbedder(Protocol):
    def embed(self, audio: AudioSlice) -> NDArray[np.float32]:
        ...


class ExportedDetector(Protocol):
    threshold: float
    metadata: dict[str, Any]

    def positive_probabilities(self, embeddings: object) -> NDArray[np.float64]:
        ...


@dataclass(frozen=True, slots=True)
class FullMatchWindow:
    match_id: str
    segment_index: int
    window_index_in_segment: int
    candidate_count_in_segment: int
    relative_window_position: float
    start_sec: float
    end_sec: float
    cheer_probability: float
    predicted_cheer: int

    @property
    def identity(self) -> tuple[int, int, float, float]:
        return (
            self.segment_index,
            self.window_index_in_segment,
            self.start_sec,
            self.end_sec,
        )


@dataclass(frozen=True, slots=True)
class SamplingWindow:
    match_id: str
    segment_index: int
    window_index_in_segment: int
    candidate_count_in_segment: int
    relative_window_position: float
    start_sec: float
    end_sec: float
    is_in_labeled_sample: bool


@dataclass(frozen=True, slots=True)
class FullMatchInferenceResult:
    match_id: str
    windows: tuple[FullMatchWindow, ...]
    sampling_windows: tuple[SamplingWindow, ...]
    summary: dict[str, Any]
    sampling_distribution: dict[str, Any]
    metadata: dict[str, Any]
    inference_runtime_sec: float


@dataclass(frozen=True, slots=True)
class FullMatchInferenceArtifactPaths:
    cheer_windows_csv: Path
    metadata_json: Path
    summary_json: Path
    sampling_distribution_json: Path
    sampling_windows_csv: Path
    sampling_relative_position_plot: Path


def infer_full_match(
    *,
    match_id: str,
    video_path: str | Path,
    segments_path: str | Path,
    audio_cache_path: str | Path,
    model_dir: str | Path,
    manifest_path: str | Path | None = None,
    external_predictions_path: str | Path | None = None,
    normalizer: AudioNormalizer | None = None,
    extractor_factory: Callable[[], WindowEmbedder] | None = None,
    detector_loader: Callable[[str | Path], ExportedDetector] | None = None,
    generated_at: str | None = None,
) -> FullMatchInferenceResult:
    """Score every canonical candidate while loading each heavy component once."""

    if not match_id:
        raise FullMatchInferenceError("match_id must not be empty")
    video = Path(video_path)
    segments_file = Path(segments_path)
    audio_cache = Path(audio_cache_path)
    model_directory = Path(model_dir)
    manifest_file = Path(manifest_path) if manifest_path is not None else None
    external_file = (
        Path(external_predictions_path)
        if external_predictions_path is not None
        else None
    )
    if external_file is not None and manifest_file is None:
        raise FullMatchInferenceError(
            "external predictions require a sample manifest for identity alignment"
        )

    artifact = load_segments_artifact(segments_file)
    segment_digest = segments_sha256(segments_file)
    load_detector = detector_loader or ExportedCheerDetector.load
    detector = load_detector(model_directory)
    _validate_detector_contract(detector)
    planner = InferenceConfig()
    started = time.perf_counter()

    source = (normalizer or FFmpegAudioNormalizer()).normalize(video, audio_cache)
    try:
        candidates = build_candidate_windows(
            artifact,
            planner,
            media_duration_sec=source.duration_sec,
        )
        _validate_candidates(candidates)
        if not candidates:
            raise FullMatchInferenceError(
                "canonical planner produced no complete analysis windows"
            )
        extractor = (
            extractor_factory()
            if extractor_factory is not None
            else YamNetEmbeddingExtractor()
        )
        embeddings = np.empty(
            (len(candidates), YAMNET_EMBEDDING_SIZE), dtype=np.float32
        )
        for index, candidate in enumerate(candidates):
            audio = source.slice_absolute(candidate.start_sec, candidate.end_sec)
            embedding = np.asarray(extractor.embed(audio))
            if embedding.shape != (YAMNET_EMBEDDING_SIZE,):
                raise FullMatchInferenceError(
                    f"candidate {index}: embedding must have shape "
                    f"({YAMNET_EMBEDDING_SIZE},)"
                )
            if embedding.dtype != np.float32 or not np.isfinite(embedding).all():
                raise FullMatchInferenceError(
                    f"candidate {index}: embedding must be finite float32"
                )
            embeddings[index] = embedding
    finally:
        source.close()

    probabilities = np.asarray(detector.positive_probabilities(embeddings))
    _validate_probabilities(probabilities, len(candidates))
    windows = tuple(
        FullMatchWindow(
            match_id=match_id,
            segment_index=candidate.segment_index,
            window_index_in_segment=candidate.window_index_in_segment,
            candidate_count_in_segment=candidate.candidate_count_in_segment,
            relative_window_position=candidate.relative_window_position,
            start_sec=candidate.start_sec,
            end_sec=candidate.end_sec,
            cheer_probability=float(probabilities[index]),
            predicted_cheer=int(probabilities[index] >= detector.threshold),
        )
        for index, candidate in enumerate(candidates)
    )
    _validate_output_windows(windows, detector.threshold, planner.window_sec)

    manifest = None
    if manifest_file is not None:
        manifest = load_manifest(manifest_file)
        _validate_manifest(
            manifest,
            match_id=match_id,
            segments_digest=segment_digest,
            planner=planner,
            candidates=candidates,
        )
    sampled_identities = _sampled_identities(manifest, candidates)
    sampling_windows = tuple(
        SamplingWindow(
            match_id=match_id,
            segment_index=candidate.segment_index,
            window_index_in_segment=candidate.window_index_in_segment,
            candidate_count_in_segment=candidate.candidate_count_in_segment,
            relative_window_position=candidate.relative_window_position,
            start_sec=candidate.start_sec,
            end_sec=candidate.end_sec,
            is_in_labeled_sample=_candidate_identity(candidate)
            in sampled_identities,
        )
        for candidate in candidates
    )
    if manifest is not None and sum(
        item.is_in_labeled_sample for item in sampling_windows
    ) != manifest.sample_size:
        raise FullMatchInferenceError(
            "not every manifest identity was found in canonical candidates"
        )

    equivalence = _verify_external_probabilities(
        windows,
        manifest,
        external_file,
    )
    runtime = time.perf_counter() - started
    eligible_segments = len({item.segment_index for item in candidates})
    summary = _build_summary(
        match_id=match_id,
        windows=windows,
        segment_count=len(artifact.segments),
        eligible_segment_count=eligible_segments,
        detector=detector,
        model_sha256=_sha256(model_directory / "model.npz"),
        equivalence=equivalence,
        runtime_sec=runtime,
    )
    sampling_distribution = build_sampling_distribution(sampling_windows)
    metadata = _build_metadata(
        match_id=match_id,
        detector=detector,
        model_directory=model_directory,
        video_path=video,
        segments_path=segments_file,
        audio_cache_path=audio_cache,
        manifest_path=manifest_file,
        external_predictions_path=external_file,
        segment_count=len(artifact.segments),
        candidate_window_count=len(candidates),
        segment_digest=segment_digest,
        planner=planner,
        equivalence=equivalence,
        generated_at=generated_at or datetime.now(UTC).isoformat(),
    )
    return FullMatchInferenceResult(
        match_id=match_id,
        windows=windows,
        sampling_windows=sampling_windows,
        summary=summary,
        sampling_distribution=sampling_distribution,
        metadata=metadata,
        inference_runtime_sec=runtime,
    )


def build_sampling_distribution(
    windows: Sequence[SamplingWindow],
) -> dict[str, Any]:
    """Describe structural sampling differences without labels or predictions."""

    if not windows:
        raise FullMatchInferenceError("sampling diagnostic needs candidate windows")
    all_positions = np.asarray(
        [item.relative_window_position for item in windows], dtype=np.float64
    )
    sampled_positions = np.asarray(
        [
            item.relative_window_position
            for item in windows
            if item.is_in_labeled_sample
        ],
        dtype=np.float64,
    )
    bins = _bin_distribution(all_positions, sampled_positions)
    distances: dict[str, float] | None = None
    if sampled_positions.size:
        from scipy.stats import ks_2samp, wasserstein_distance

        distances = {
            "kolmogorov_smirnov_statistic": float(
                ks_2samp(sampled_positions, all_positions).statistic
            ),
            "wasserstein_distance": float(
                wasserstein_distance(sampled_positions, all_positions)
            ),
        }
    all_counts = Counter(item.segment_index for item in windows)
    sampled_counts = Counter(
        item.segment_index for item in windows if item.is_in_labeled_sample
    )
    return {
        "diagnostic_type": "structural_sampling_distribution",
        "relative_window_position_definition": (
            "0.5 when candidate_count_in_segment == 1; otherwise "
            "window_index_in_segment / (candidate_count_in_segment - 1)"
        ),
        "relative_window_position": {
            "all_candidate_windows": _distribution_statistics(all_positions),
            "sampled_windows": (
                _distribution_statistics(sampled_positions)
                if sampled_positions.size
                else None
            ),
            "fixed_bins": bins,
            "descriptive_distances": distances,
        },
        "segment_weighting": {
            "all_candidate_windows": _count_statistics(all_counts.values()),
            "sampled_windows": (
                _count_statistics(sampled_counts.values())
                if sampled_counts
                else None
            ),
            "interpretation": (
                "segment-diverse evaluation and deployment window population "
                "use different segment weighting; this is not treated as a bug"
            ),
        },
    }


def write_full_match_inference_artifacts(
    result: FullMatchInferenceResult,
    output_dir: str | Path,
) -> FullMatchInferenceArtifactPaths:
    """Serialize a complete artifact set before atomically publishing each file."""

    output = Path(output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    names = (
        "cheer_windows.csv",
        "metadata.json",
        "summary.json",
        "sampling_distribution.json",
        "sampling_windows.csv",
        "sampling_relative_position.png",
    )
    try:
        _write_cheer_windows(staging / names[0], result.windows)
        _write_json(staging / names[1], result.metadata)
        _write_json(staging / names[2], result.summary)
        _write_json(staging / names[3], result.sampling_distribution)
        _write_sampling_windows(staging / names[4], result.sampling_windows)
        _write_sampling_plot(staging / names[5], result.sampling_windows)
        _publish_staged_artifacts(staging, output, names)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return FullMatchInferenceArtifactPaths(
        cheer_windows_csv=output / names[0],
        metadata_json=output / names[1],
        summary_json=output / names[2],
        sampling_distribution_json=output / names[3],
        sampling_windows_csv=output / names[4],
        sampling_relative_position_plot=output / names[5],
    )


def _publish_staged_artifacts(
    staging: Path,
    output: Path,
    names: Sequence[str],
) -> None:
    """Publish the set transactionally, restoring prior outputs on OS failure."""

    output.mkdir(parents=True, exist_ok=True)
    backup = staging / "previous"
    backup.mkdir()
    moved_previous: list[str] = []
    published: list[str] = []
    try:
        for name in names:
            destination = output / name
            if destination.exists():
                os.replace(destination, backup / name)
                moved_previous.append(name)
        for name in names:
            os.replace(staging / name, output / name)
            published.append(name)
    except OSError:
        for name in published:
            (output / name).unlink(missing_ok=True)
        for name in moved_previous:
            os.replace(backup / name, output / name)
        raise


def _validate_detector_contract(detector: ExportedDetector) -> None:
    metadata = detector.metadata
    try:
        feature = metadata["feature_extractor"]
        audio = metadata["audio"]
        if feature["model_identifier"] != YAMNET_MODEL_HANDLE:
            raise FullMatchInferenceError("detector uses an unexpected YAMNet model")
        if feature["pooling"] != "mean":
            raise FullMatchInferenceError("detector pooling must be mean")
        if feature["embedding_dimension"] != YAMNET_EMBEDDING_SIZE:
            raise FullMatchInferenceError("detector embedding dimension must be 1024")
        planner = InferenceConfig()
        if audio != {
            "sample_rate_hz": planner.sample_rate_hz,
            "window_sec": planner.window_sec,
            "hop_sec": planner.hop_sec,
            "post_padding_sec": planner.post_padding_sec,
        }:
            raise FullMatchInferenceError("detector audio planner is not canonical")
    except (KeyError, TypeError) as exc:
        raise FullMatchInferenceError(
            f"detector metadata is incomplete: {exc}"
        ) from exc


def _validate_candidates(candidates: Sequence[CandidateWindow]) -> None:
    identities: set[tuple[int, int, float, float]] = set()
    previous: tuple[int, int] | None = None
    for candidate in candidates:
        order = (candidate.segment_index, candidate.window_index_in_segment)
        if previous is not None and order <= previous:
            raise FullMatchInferenceError(
                "candidate ordering must be segment_index then window_index"
            )
        previous = order
        identity = _candidate_identity(candidate)
        if identity in identities:
            raise FullMatchInferenceError("duplicate canonical window identity")
        identities.add(identity)
        if not 0 <= candidate.window_index_in_segment < candidate.candidate_count_in_segment:
            raise FullMatchInferenceError("candidate window index/count is invalid")
        if not math.isclose(
            candidate.end_sec - candidate.start_sec,
            3.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise FullMatchInferenceError("candidate duration must be exactly 3 seconds")


def _validate_probabilities(probabilities: NDArray[Any], count: int) -> None:
    if probabilities.shape != (count,):
        raise FullMatchInferenceError(
            f"detector probabilities must have shape ({count},)"
        )
    if not np.issubdtype(probabilities.dtype, np.floating):
        raise FullMatchInferenceError("detector probabilities must be floating point")
    if not np.isfinite(probabilities).all():
        raise FullMatchInferenceError("detector probabilities must be finite")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise FullMatchInferenceError("detector probabilities must be in [0, 1]")


def _validate_output_windows(
    windows: Sequence[FullMatchWindow], threshold: float, window_sec: float
) -> None:
    identities: set[tuple[int, int, float, float]] = set()
    previous: tuple[int, int] | None = None
    for item in windows:
        order = (item.segment_index, item.window_index_in_segment)
        if previous is not None and order <= previous:
            raise FullMatchInferenceError("output ordering is not deterministic")
        previous = order
        if item.identity in identities:
            raise FullMatchInferenceError("duplicate output window identity")
        identities.add(item.identity)
        if item.start_sec >= item.end_sec or not math.isclose(
            item.end_sec - item.start_sec,
            window_sec,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise FullMatchInferenceError("output window timestamps are invalid")
        if item.predicted_cheer != int(item.cheer_probability >= threshold):
            raise FullMatchInferenceError("threshold prediction is inconsistent")


def _validate_manifest(
    manifest: SampleManifest,
    *,
    match_id: str,
    segments_digest: str,
    planner: InferenceConfig,
    candidates: Sequence[CandidateWindow],
) -> None:
    if manifest.match_id != match_id:
        raise FullMatchInferenceError("sample manifest match_id mismatch")
    if manifest.segments_sha256 != segments_digest:
        raise FullMatchInferenceError("sample manifest segments SHA-256 mismatch")
    if manifest.planner != planner:
        raise FullMatchInferenceError("sample manifest planner mismatch")
    if manifest.candidate_window_count != len(candidates):
        raise FullMatchInferenceError("sample manifest candidate count mismatch")
    eligible = len({item.segment_index for item in candidates})
    if manifest.eligible_segment_count != eligible:
        raise FullMatchInferenceError("sample manifest eligible segment count mismatch")


def _sampled_identities(
    manifest: SampleManifest | None,
    candidates: Sequence[CandidateWindow],
) -> set[tuple[int, int, float, float]]:
    if manifest is None:
        return set()
    candidate_by_identity = {
        _candidate_identity(candidate): candidate for candidate in candidates
    }
    identities: set[tuple[int, int, float, float]] = set()
    for sample in manifest.windows:
        identity = (
            sample.segment_index,
            sample.window_index_in_segment,
            sample.start_sec,
            sample.end_sec,
        )
        candidate = candidate_by_identity.get(identity)
        if candidate is None:
            raise FullMatchInferenceError(
                f"sample rank {sample.sample_rank}: canonical identity not found"
            )
        if candidate.candidate_count_in_segment != sample.candidate_count_in_segment:
            raise FullMatchInferenceError(
                f"sample rank {sample.sample_rank}: candidate count mismatch"
            )
        if not math.isclose(
            candidate.relative_window_position,
            sample.relative_window_position,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise FullMatchInferenceError(
                f"sample rank {sample.sample_rank}: relative position mismatch"
            )
        identities.add(identity)
    if len(identities) != manifest.sample_size:
        raise FullMatchInferenceError("sample manifest contains duplicate identities")
    return identities


def _verify_external_probabilities(
    windows: Sequence[FullMatchWindow],
    manifest: SampleManifest | None,
    path: Path | None,
) -> dict[str, Any]:
    if path is None:
        return {
            "available": False,
            "matched_window_count": 0,
            "max_absolute_probability_difference": None,
            "rtol": PROBABILITY_RTOL,
            "atol": PROBABILITY_ATOL,
        }
    if manifest is None:
        raise FullMatchInferenceError("external predictions require a manifest")
    by_identity = {item.identity: item.cheer_probability for item in windows}
    external: dict[tuple[int, int, float, float], float] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            required = {
                "match_id",
                "segment_index",
                "window_index_in_segment",
                "start_sec",
                "end_sec",
                "positive_probability",
            }
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise FullMatchInferenceError(
                    "external predictions CSV is missing identity/probability fields"
                )
            for row_number, row in enumerate(reader, start=2):
                if row["match_id"] != manifest.match_id:
                    raise FullMatchInferenceError(
                        f"external predictions row {row_number}: match_id mismatch"
                    )
                identity = (
                    int(row["segment_index"]),
                    int(row["window_index_in_segment"]),
                    float(row["start_sec"]),
                    float(row["end_sec"]),
                )
                if identity in external:
                    raise FullMatchInferenceError(
                        "external predictions contain duplicate identities"
                    )
                probability = float(row["positive_probability"])
                if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                    raise FullMatchInferenceError(
                        f"external predictions row {row_number}: invalid probability"
                    )
                external[identity] = probability
    except (OSError, TypeError, ValueError) as exc:
        if isinstance(exc, FullMatchInferenceError):
            raise
        raise FullMatchInferenceError(
            f"cannot load external predictions: {exc}"
        ) from exc

    manifest_identities = {
        (
            item.segment_index,
            item.window_index_in_segment,
            item.start_sec,
            item.end_sec,
        )
        for item in manifest.windows
    }
    if set(external) != manifest_identities:
        raise FullMatchInferenceError(
            "external prediction identities do not exactly equal manifest identities"
        )
    full_values = np.asarray([by_identity[key] for key in external], dtype=np.float64)
    external_values = np.asarray(list(external.values()), dtype=np.float64)
    differences = np.abs(full_values - external_values)
    if not np.allclose(
        full_values,
        external_values,
        rtol=PROBABILITY_RTOL,
        atol=PROBABILITY_ATOL,
    ):
        raise FullMatchInferenceError(
            "full-match probabilities differ from frozen external validation"
        )
    return {
        "available": True,
        "matched_window_count": int(external_values.size),
        "max_absolute_probability_difference": float(np.max(differences)),
        "rtol": PROBABILITY_RTOL,
        "atol": PROBABILITY_ATOL,
    }


def _build_summary(
    *,
    match_id: str,
    windows: Sequence[FullMatchWindow],
    segment_count: int,
    eligible_segment_count: int,
    detector: ExportedDetector,
    model_sha256: str,
    equivalence: dict[str, Any],
    runtime_sec: float,
) -> dict[str, Any]:
    probabilities = np.asarray(
        [item.cheer_probability for item in windows], dtype=np.float64
    )
    threshold_counts = {}
    for value, name in ((0.5, "p_ge_0_5"), (0.9, "p_ge_0_9"), (0.99, "p_ge_0_99")):
        count = int(np.count_nonzero(probabilities >= value))
        threshold_counts[name] = {
            "threshold": value,
            "count": count,
            "rate": count / probabilities.size,
        }
    positive_count = int(
        np.count_nonzero(probabilities >= detector.threshold)
    )
    return {
        "match_id": match_id,
        "candidate_window_count": int(probabilities.size),
        "segment_count": segment_count,
        "eligible_segment_count": eligible_segment_count,
        "model_id": detector.metadata["model_id"],
        "model_sha256": model_sha256,
        "threshold": detector.threshold,
        "probability": _probability_statistics(probabilities),
        "predicted_positive_count": positive_count,
        "predicted_positive_rate": positive_count / probabilities.size,
        "descriptive_threshold_counts": threshold_counts,
        "sampled_external_probability_equivalence": equivalence,
        "inference_runtime_sec": runtime_sec,
    }


def _probability_statistics(values: NDArray[np.float64]) -> dict[str, float]:
    quantiles = np.quantile(values, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "min": float(np.min(values)),
        "q01": float(quantiles[0]),
        "q05": float(quantiles[1]),
        "q25": float(quantiles[2]),
        "median": float(quantiles[3]),
        "q75": float(quantiles[4]),
        "q95": float(quantiles[5]),
        "q99": float(quantiles[6]),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def _distribution_statistics(values: NDArray[np.float64]) -> dict[str, float | int]:
    quantiles = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "q05": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q75": float(quantiles[3]),
        "q95": float(quantiles[4]),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def _count_statistics(values: Any) -> dict[str, float | int]:
    counts = np.asarray(list(values), dtype=np.float64)
    if counts.size == 0:
        raise FullMatchInferenceError("segment count statistics cannot be empty")
    return {
        "segment_count": int(counts.size),
        "mean_windows_per_segment": float(np.mean(counts)),
        "median_windows_per_segment": float(np.median(counts)),
        "min_windows_per_segment": int(np.min(counts)),
        "max_windows_per_segment": int(np.max(counts)),
    }


def _bin_distribution(
    all_values: NDArray[np.float64], sampled_values: NDArray[np.float64]
) -> list[dict[str, Any]]:
    edges = np.asarray(RELATIVE_POSITION_BIN_EDGES, dtype=np.float64)
    all_counts, _ = np.histogram(all_values, bins=edges)
    sampled_counts, _ = np.histogram(sampled_values, bins=edges)
    rows = []
    for index in range(10):
        rows.append(
            {
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "upper_inclusive": index == 9,
                "all_candidate_count": int(all_counts[index]),
                "all_candidate_fraction": float(
                    all_counts[index] / all_values.size
                ),
                "sampled_count": int(sampled_counts[index]),
                "sampled_fraction": (
                    float(sampled_counts[index] / sampled_values.size)
                    if sampled_values.size
                    else None
                ),
            }
        )
    return rows


def _build_metadata(
    *,
    match_id: str,
    detector: ExportedDetector,
    model_directory: Path,
    video_path: Path,
    segments_path: Path,
    audio_cache_path: Path,
    manifest_path: Path | None,
    external_predictions_path: Path | None,
    segment_count: int,
    candidate_window_count: int,
    segment_digest: str,
    planner: InferenceConfig,
    equivalence: dict[str, Any],
    generated_at: str,
) -> dict[str, Any]:
    model_path = model_directory / "model.npz"
    model_metadata_path = model_directory / "metadata.json"
    model = detector.metadata
    inputs = {
        "model": _artifact_identity(model_path),
        "model_metadata": _artifact_identity(model_metadata_path),
        "segments": _artifact_identity(segments_path, known_sha=segment_digest),
        "audio_cache": _artifact_identity(audio_cache_path),
    }
    if manifest_path is not None:
        inputs["sample_manifest"] = _artifact_identity(manifest_path)
    if external_predictions_path is not None:
        inputs["external_validation_predictions"] = _artifact_identity(
            external_predictions_path
        )
    return {
        "experiment_type": "full_match_inference",
        "match_id": match_id,
        "model": {
            "model_id": model["model_id"],
            "model_sha256": inputs["model"]["sha256"],
            "threshold": detector.threshold,
            "training_matches": list(model["training"]["matches"]),
        },
        "feature_extractor": {
            "model_identifier": model["feature_extractor"]["model_identifier"],
            "pooling": model["feature_extractor"]["pooling"],
            "embedding_dimension": model["feature_extractor"][
                "embedding_dimension"
            ],
        },
        "audio": {
            "sample_rate_hz": YAMNET_SAMPLE_RATE_HZ,
            "channels": 1,
            "canonical_audio_cache_version": AUDIO_CACHE_FORMAT_VERSION,
            "audio_cache_sha256": inputs["audio_cache"]["sha256"],
        },
        "segmentation": {
            "segments_sha256": segment_digest,
            "segment_count": segment_count,
            "identity": "current segments array positional zero-based index",
        },
        "planner": {
            "window_sec": planner.window_sec,
            "hop_sec": planner.hop_sec,
            "post_padding_sec": planner.post_padding_sec,
            "candidate_window_count": candidate_window_count,
        },
        "sampled_external_probability_equivalence": equivalence,
        "software_provenance": {
            "numpy_version": np.__version__,
            "tensorflow_version": _package_version("tensorflow"),
            "tensorflow_hub_version": _package_version("tensorflow-hub"),
            "package_version": _package_version("badminton-audio-highlight"),
            "generated_at": generated_at,
        },
        "inputs": inputs,
        "source_media": {
            "path": _portable_path(video_path),
            "role": "source used only if canonical normalized cache requires rebuild",
        },
    }


def _write_cheer_windows(path: Path, windows: Sequence[FullMatchWindow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(FullMatchWindow.__dataclass_fields__))
        writer.writeheader()
        for item in windows:
            writer.writerow(asdict(item))


def _write_sampling_windows(path: Path, windows: Sequence[SamplingWindow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(SamplingWindow.__dataclass_fields__))
        writer.writeheader()
        for item in windows:
            row = asdict(item)
            row["is_in_labeled_sample"] = (
                "true" if item.is_in_labeled_sample else "false"
            )
            writer.writerow(row)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")


def _write_sampling_plot(path: Path, windows: Sequence[SamplingWindow]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_values = np.asarray(
        [item.relative_window_position for item in windows], dtype=np.float64
    )
    sampled_values = np.asarray(
        [item.relative_window_position for item in windows if item.is_in_labeled_sample],
        dtype=np.float64,
    )
    edges = np.asarray(RELATIVE_POSITION_BIN_EDGES, dtype=np.float64)
    all_counts, _ = np.histogram(all_values, bins=edges)
    all_fraction = all_counts / all_values.size
    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.stairs(
        all_fraction,
        edges,
        label=f"all candidate windows (N={all_values.size})",
        linewidth=2,
    )
    if sampled_values.size:
        sampled_counts, _ = np.histogram(sampled_values, bins=edges)
        axis.stairs(
            sampled_counts / sampled_values.size,
            edges,
            label=f"sampled windows (N={sampled_values.size})",
            linewidth=2,
        )
    axis.set_title("Sampling distribution by relative window position")
    axis.set_xlabel("relative window position")
    axis.set_ylabel("fraction of windows per fixed 0.1 bin")
    axis.set_xlim(0.0, 1.0)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _candidate_identity(
    candidate: CandidateWindow,
) -> tuple[int, int, float, float]:
    return (
        candidate.segment_index,
        candidate.window_index_in_segment,
        candidate.start_sec,
        candidate.end_sec,
    )


def _artifact_identity(path: Path, *, known_sha: str | None = None) -> dict[str, str]:
    return {"path": _portable_path(path), "sha256": known_sha or _sha256(path)}


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
        raise FullMatchInferenceError(f"cannot hash input {path}: {exc}") from exc
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None
