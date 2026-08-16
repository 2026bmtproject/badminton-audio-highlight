"""Build canonical YAMNet feature datasets from blind human labels."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from audio_highlight.audio import (
    AudioNormalizer,
    AudioSlice,
    FFmpegAudioNormalizer,
    YAMNET_SAMPLE_RATE_HZ,
)
from audio_highlight.labeling import (
    LABEL_SOURCE,
    SAMPLING_ALGORITHM_VERSION,
    LabelStore,
    LabelingError,
    load_manifest,
)
from audio_highlight.windows import InferenceConfig
from audio_highlight.yamnet import (
    YAMNET_EMBEDDING_SIZE,
    YAMNET_MODEL_HANDLE,
    YamNetEmbeddingExtractor,
)


class DatasetError(ValueError):
    """Raised when canonical feature data is incomplete or inconsistent."""


class FeatureEmbedder(Protocol):
    """Embedding boundary used by the dataset builder."""

    def embed(self, audio: AudioSlice) -> NDArray[np.float32]:
        ...


@dataclass(frozen=True, slots=True)
class FeatureDataset:
    embeddings: NDArray[np.float32]
    labels: NDArray[np.uint8]
    sample_ranks: NDArray[np.int64]
    segment_indices: NDArray[np.int64]
    window_indices: NDArray[np.int64]
    start_secs: NDArray[np.float64]
    end_secs: NDArray[np.float64]
    match_id: str
    segments_sha256: str
    embedding_dimension: int
    sample_rate_hz: int
    model_identifier: str
    window_sec: float
    hop_sec: float
    post_padding_sec: float
    label_source: str
    sampling_seed: int
    sampling_algorithm_version: int

    def __post_init__(self) -> None:
        if self.embeddings.ndim != 2 or self.embeddings.shape[1] != YAMNET_EMBEDDING_SIZE:
            raise DatasetError("embeddings must have shape (N, 1024)")
        row_count = self.embeddings.shape[0]
        arrays: Sequence[NDArray[np.generic]] = (
            self.labels,
            self.sample_ranks,
            self.segment_indices,
            self.window_indices,
            self.start_secs,
            self.end_secs,
        )
        if any(array.shape != (row_count,) for array in arrays):
            raise DatasetError("feature metadata must align with embeddings")
        if self.embeddings.dtype != np.float32 or not np.isfinite(self.embeddings).all():
            raise DatasetError("embeddings must be finite float32")
        if self.labels.dtype != np.uint8 or not np.isin(self.labels, [0, 1]).all():
            raise DatasetError("labels must be binary uint8")
        expected_dtypes = (
            (self.sample_ranks, np.dtype(np.int64), "sample_ranks"),
            (self.segment_indices, np.dtype(np.int64), "segment_indices"),
            (self.window_indices, np.dtype(np.int64), "window_indices"),
            (self.start_secs, np.dtype(np.float64), "start_secs"),
            (self.end_secs, np.dtype(np.float64), "end_secs"),
        )
        for array, dtype, name in expected_dtypes:
            if array.dtype != dtype:
                raise DatasetError(f"{name} must have dtype {dtype}")
        if (
            not np.isfinite(self.start_secs).all()
            or not np.isfinite(self.end_secs).all()
            or np.any(self.start_secs < 0)
            or np.any(self.end_secs <= self.start_secs)
        ):
            raise DatasetError("timestamps must be valid absolute ranges")
        if (
            np.any(self.sample_ranks <= 0)
            or np.any(self.segment_indices < 0)
            or np.any(self.window_indices < 0)
        ):
            raise DatasetError("provenance indices must be non-negative")
        if self.embedding_dimension != YAMNET_EMBEDDING_SIZE:
            raise DatasetError("embedding_dimension must be 1024")
        if self.sample_rate_hz != YAMNET_SAMPLE_RATE_HZ:
            raise DatasetError("sample_rate_hz must be 16000")
        if self.label_source != LABEL_SOURCE:
            raise DatasetError(f"label_source must be {LABEL_SOURCE!r}")
        if self.sampling_algorithm_version != SAMPLING_ALGORITHM_VERSION:
            raise DatasetError("unsupported sampling algorithm version")
        InferenceConfig(
            sample_rate_hz=self.sample_rate_hz,
            window_sec=self.window_sec,
            hop_sec=self.hop_sec,
            post_padding_sec=self.post_padding_sec,
        )
        if not self.match_id or len(self.segments_sha256) != 64 or not self.model_identifier:
            raise DatasetError("feature identity metadata is invalid")

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp.npz")
        try:
            np.savez_compressed(
                temporary,
                embeddings=self.embeddings,
                labels=self.labels,
                sample_ranks=self.sample_ranks,
                segment_indices=self.segment_indices,
                window_indices=self.window_indices,
                start_secs=self.start_secs,
                end_secs=self.end_secs,
                match_id=np.asarray(self.match_id, dtype=np.str_),
                segments_sha256=np.asarray(self.segments_sha256, dtype=np.str_),
                embedding_dimension=np.asarray(self.embedding_dimension, dtype=np.int64),
                sample_rate_hz=np.asarray(self.sample_rate_hz, dtype=np.int64),
                model_identifier=np.asarray(self.model_identifier, dtype=np.str_),
                window_sec=np.asarray(self.window_sec, dtype=np.float64),
                hop_sec=np.asarray(self.hop_sec, dtype=np.float64),
                post_padding_sec=np.asarray(self.post_padding_sec, dtype=np.float64),
                label_source=np.asarray(self.label_source, dtype=np.str_),
                sampling_seed=np.asarray(self.sampling_seed, dtype=np.int64),
                sampling_algorithm_version=np.asarray(
                    self.sampling_algorithm_version, dtype=np.int64
                ),
            )
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
        return output

    @classmethod
    def load(cls, path: str | Path) -> FeatureDataset:
        try:
            with np.load(Path(path), allow_pickle=False) as values:
                return cls(
                    embeddings=np.asarray(values["embeddings"]),
                    labels=np.asarray(values["labels"]),
                    sample_ranks=np.asarray(values["sample_ranks"]),
                    segment_indices=np.asarray(values["segment_indices"]),
                    window_indices=np.asarray(values["window_indices"]),
                    start_secs=np.asarray(values["start_secs"]),
                    end_secs=np.asarray(values["end_secs"]),
                    match_id=str(values["match_id"].item()),
                    segments_sha256=str(values["segments_sha256"].item()),
                    embedding_dimension=int(values["embedding_dimension"].item()),
                    sample_rate_hz=int(values["sample_rate_hz"].item()),
                    model_identifier=str(values["model_identifier"].item()),
                    window_sec=float(values["window_sec"].item()),
                    hop_sec=float(values["hop_sec"].item()),
                    post_padding_sec=float(values["post_padding_sec"].item()),
                    label_source=str(values["label_source"].item()),
                    sampling_seed=int(values["sampling_seed"].item()),
                    sampling_algorithm_version=int(
                        values["sampling_algorithm_version"].item()
                    ),
                )
        except (KeyError, ValueError) as exc:
            raise DatasetError(f"invalid feature dataset: {exc}") from exc


@dataclass(frozen=True, slots=True)
class FeatureBuildResult:
    dataset: FeatureDataset
    reviewed: int
    binary_included: int
    ambiguous_excluded: int
    output_path: Path


def build_feature_dataset(
    *,
    video_path: str | Path,
    labels_path: str | Path,
    manifest_path: str | Path,
    output_path: str | Path,
    audio_cache_path: str | Path | None = None,
    normalizer: AudioNormalizer | None = None,
    extractor_factory: Callable[[], FeatureEmbedder] | None = None,
    model_identifier: str = YAMNET_MODEL_HANDLE,
) -> FeatureBuildResult:
    """Embed completed binary labels directly from their absolute timestamps."""

    manifest = load_manifest(manifest_path)
    store = LabelStore(labels_path, manifest)
    decisions = store.decisions
    if len(decisions) != manifest.sample_size:
        raise LabelingError(
            f"labels are incomplete: reviewed {len(decisions)} of "
            f"{manifest.sample_size}"
        )
    binary = [
        decisions[window.sample_rank]
        for window in manifest.windows
        if not decisions[window.sample_rank].is_ambiguous
    ]
    ambiguous_count = manifest.sample_size - len(binary)
    if not binary:
        raise LabelingError("labels contain no binary samples to embed")

    output = Path(output_path)
    cache = (
        Path(audio_cache_path)
        if audio_cache_path is not None
        else output.with_name(f"{manifest.match_id}.audio.f32le")
    )
    source = (normalizer or FFmpegAudioNormalizer()).normalize(video_path, cache)
    embeddings: list[NDArray[np.float32]] = []
    try:
        if any(decision.window_end_sec > source.duration_sec for decision in binary):
            raise LabelingError("label window exceeds normalized media duration")
        extractor = (
            extractor_factory()
            if extractor_factory is not None
            else YamNetEmbeddingExtractor()
        )
        for decision in binary:
            audio = source.slice_absolute(
                decision.window_start_sec, decision.window_end_sec
            )
            embedding = np.asarray(extractor.embed(audio), dtype=np.float32)
            if embedding.shape != (YAMNET_EMBEDDING_SIZE,):
                raise DatasetError(
                    f"sample {decision.sample_rank}: embedding must have shape "
                    f"({YAMNET_EMBEDDING_SIZE},)"
                )
            if not np.isfinite(embedding).all():
                raise DatasetError(
                    f"sample {decision.sample_rank}: embedding is not finite"
                )
            embeddings.append(embedding)
    finally:
        source.close()

    planner = manifest.planner
    dataset = FeatureDataset(
        embeddings=np.stack(embeddings).astype(np.float32, copy=False),
        labels=np.asarray([decision.has_cheer for decision in binary], dtype=np.uint8),
        sample_ranks=np.asarray(
            [decision.sample_rank for decision in binary], dtype=np.int64
        ),
        segment_indices=np.asarray(
            [decision.segment_index for decision in binary], dtype=np.int64
        ),
        window_indices=np.asarray(
            [decision.window_index_in_segment for decision in binary], dtype=np.int64
        ),
        start_secs=np.asarray(
            [decision.window_start_sec for decision in binary], dtype=np.float64
        ),
        end_secs=np.asarray(
            [decision.window_end_sec for decision in binary], dtype=np.float64
        ),
        match_id=manifest.match_id,
        segments_sha256=manifest.segments_sha256,
        embedding_dimension=YAMNET_EMBEDDING_SIZE,
        sample_rate_hz=planner.sample_rate_hz,
        model_identifier=model_identifier,
        window_sec=planner.window_sec,
        hop_sec=planner.hop_sec,
        post_padding_sec=planner.post_padding_sec,
        label_source=LABEL_SOURCE,
        sampling_seed=manifest.seed,
        sampling_algorithm_version=manifest.sampling_algorithm_version,
    )
    saved = dataset.save(output)
    return FeatureBuildResult(
        dataset=dataset,
        reviewed=len(decisions),
        binary_included=len(binary),
        ambiguous_excluded=ambiguous_count,
        output_path=saved,
    )
