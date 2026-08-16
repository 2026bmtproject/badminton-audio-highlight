"""Historical label import and rebuildable YAMNet feature dataset creation."""

from __future__ import annotations

import csv
import math
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from audio_highlight.audio import (
    AudioSlice,
    FFmpegAudioNormalizer,
    YAMNET_SAMPLE_RATE_HZ,
)
from audio_highlight.yamnet import (
    YAMNET_EMBEDDING_SIZE,
    YAMNET_MODEL_HANDLE,
    YamNetEmbeddingExtractor,
)

_REQUIRED_COLUMNS = {
    "window_global_id",
    "segment_id",
    "window_start_sec",
    "window_end_sec",
    "has_cheer",
    "cheer_confidence",
    "reviewed",
}
_MISSING_INT = -1


class TrainingDataError(ValueError):
    """Raised when historical labels or generated features are invalid."""


@dataclass(frozen=True, slots=True)
class LabeledWindow:
    """One reviewed historical label keyed only by absolute match timestamps."""

    match_id: str
    window_id: int
    source_segment_id: int | None
    start_sec: float
    end_sec: float
    has_cheer: bool
    cheer_confidence: int | None
    reviewed: bool
    source_wav_path: str | None = None


@dataclass(frozen=True, slots=True)
class LabelImportSummary:
    total_rows: int
    reviewed_rows: int
    skipped_unreviewed_rows: int
    negative_rows: int
    positive_rows: int


@dataclass(frozen=True, slots=True)
class ImportedLabels:
    windows: tuple[LabeledWindow, ...]
    summary: LabelImportSummary


def _field(row: dict[str, str | None], name: str, row_number: int) -> str:
    value = row.get(name)
    if value is None or not value.strip():
        raise TrainingDataError(f"row {row_number}: missing {name}")
    return value.strip()


def _optional_int(value: str | None, name: str, row_number: int) -> int | None:
    if value is None or not value.strip():
        return None
    try:
        return int(value.strip())
    except ValueError as exc:
        raise TrainingDataError(f"row {row_number}: {name} must be an integer") from exc


def _strict_bool(value: str, name: str, row_number: int) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise TrainingDataError(f"row {row_number}: {name} must be True/False or 1/0")


def _timestamp(value: str, name: str, row_number: int) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise TrainingDataError(f"row {row_number}: {name} must be numeric") from exc
    if not math.isfinite(result):
        raise TrainingDataError(f"row {row_number}: {name} timestamp must be finite")
    return result


def import_cheer_labels(
    path: str | Path,
    *,
    match_id: str,
) -> ImportedLabels:
    """Import reviewed CSV rows without reading legacy ``wav_path`` files."""

    if not match_id:
        raise TrainingDataError("match_id must not be empty")
    csv_path = Path(path)
    windows: list[LabeledWindow] = []
    total_rows = 0
    skipped = 0

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        columns = set(reader.fieldnames or ())
        missing = sorted(_REQUIRED_COLUMNS - columns)
        if missing:
            raise TrainingDataError(f"label CSV is missing columns: {', '.join(missing)}")

        for row_number, row in enumerate(reader, start=2):
            total_rows += 1
            reviewed = _strict_bool(_field(row, "reviewed", row_number), "reviewed", row_number)
            if not reviewed:
                skipped += 1
                continue

            window_id = _optional_int(
                _field(row, "window_global_id", row_number),
                "window_global_id",
                row_number,
            )
            assert window_id is not None
            start_sec = _timestamp(
                _field(row, "window_start_sec", row_number),
                "window_start_sec",
                row_number,
            )
            end_sec = _timestamp(
                _field(row, "window_end_sec", row_number),
                "window_end_sec",
                row_number,
            )
            if start_sec < 0 or end_sec <= start_sec:
                raise TrainingDataError(
                    f"row {row_number}, window {window_id}: invalid absolute timestamp "
                    f"range [{start_sec}, {end_sec})"
                )
            label_text = (row.get("has_cheer") or "").strip()
            if label_text not in {"0", "1"}:
                raise TrainingDataError(
                    f"row {row_number}, window {window_id}: has_cheer must be 0 or 1"
                )
            legacy_wav_path = row.get("wav_path")
            windows.append(
                LabeledWindow(
                    match_id=match_id,
                    window_id=window_id,
                    source_segment_id=_optional_int(
                        row.get("segment_id"),
                        "segment_id",
                        row_number,
                    ),
                    start_sec=start_sec,
                    end_sec=end_sec,
                    has_cheer=label_text == "1",
                    cheer_confidence=_optional_int(
                        row.get("cheer_confidence"),
                        "cheer_confidence",
                        row_number,
                    ),
                    reviewed=True,
                    source_wav_path=(
                        legacy_wav_path
                        if legacy_wav_path is not None and legacy_wav_path != ""
                        else None
                    ),
                )
            )

    positives = sum(window.has_cheer for window in windows)
    summary = LabelImportSummary(
        total_rows=total_rows,
        reviewed_rows=len(windows),
        skipped_unreviewed_rows=skipped,
        negative_rows=len(windows) - positives,
        positive_rows=positives,
    )
    return ImportedLabels(tuple(windows), summary)


@dataclass(frozen=True, slots=True)
class FeatureDataset:
    """Safe, non-pickled feature matrix and aligned training metadata."""

    embeddings: NDArray[np.float32]
    labels: NDArray[np.uint8]
    window_ids: NDArray[np.int64]
    start_secs: NDArray[np.float64]
    end_secs: NDArray[np.float64]
    source_segment_ids: NDArray[np.int64]
    cheer_confidences: NDArray[np.int64]
    match_id: str
    embedding_dimension: int = YAMNET_EMBEDDING_SIZE
    sample_rate_hz: int = YAMNET_SAMPLE_RATE_HZ
    model_identifier: str = YAMNET_MODEL_HANDLE

    def __post_init__(self) -> None:
        if self.embeddings.ndim != 2 or self.embeddings.shape[1] != self.embedding_dimension:
            raise TrainingDataError(
                f"embeddings must have shape (N, {self.embedding_dimension}), "
                f"got {self.embeddings.shape}"
            )
        row_count = self.embeddings.shape[0]
        arrays: Sequence[NDArray[np.generic]] = (
            self.labels,
            self.window_ids,
            self.start_secs,
            self.end_secs,
            self.source_segment_ids,
            self.cheer_confidences,
        )
        if any(array.shape != (row_count,) for array in arrays):
            raise TrainingDataError("feature metadata arrays must align with embeddings")
        if self.embeddings.dtype != np.float32 or not np.isfinite(self.embeddings).all():
            raise TrainingDataError("embeddings must be finite float32 values")
        if self.labels.dtype != np.uint8 or not np.isin(self.labels, [0, 1]).all():
            raise TrainingDataError("labels must be uint8 binary values")

    def save(self, path: str | Path) -> Path:
        """Atomically save a compressed NPZ containing no pickled objects."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp.npz")
        try:
            np.savez_compressed(
                temporary,
                embeddings=self.embeddings,
                labels=self.labels,
                window_ids=self.window_ids,
                start_secs=self.start_secs,
                end_secs=self.end_secs,
                source_segment_ids=self.source_segment_ids,
                cheer_confidences=self.cheer_confidences,
                match_id=np.asarray(self.match_id, dtype=np.str_),
                embedding_dimension=np.asarray(self.embedding_dimension, dtype=np.int64),
                sample_rate_hz=np.asarray(self.sample_rate_hz, dtype=np.int64),
                model_identifier=np.asarray(self.model_identifier, dtype=np.str_),
            )
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
        return output

    @classmethod
    def load(cls, path: str | Path) -> FeatureDataset:
        with np.load(Path(path), allow_pickle=False) as values:
            return cls(
                embeddings=np.asarray(values["embeddings"], dtype=np.float32),
                labels=np.asarray(values["labels"], dtype=np.uint8),
                window_ids=np.asarray(values["window_ids"], dtype=np.int64),
                start_secs=np.asarray(values["start_secs"], dtype=np.float64),
                end_secs=np.asarray(values["end_secs"], dtype=np.float64),
                source_segment_ids=np.asarray(values["source_segment_ids"], dtype=np.int64),
                cheer_confidences=np.asarray(values["cheer_confidences"], dtype=np.int64),
                match_id=str(values["match_id"].item()),
                embedding_dimension=int(values["embedding_dimension"].item()),
                sample_rate_hz=int(values["sample_rate_hz"].item()),
                model_identifier=str(values["model_identifier"].item()),
            )


class FeatureAudioSource(Protocol):
    duration_sec: float

    def slice_absolute(self, start_sec: float, end_sec: float) -> AudioSlice:
        ...

    def close(self) -> None:
        ...


class FeatureAudioNormalizer(Protocol):
    def normalize(
        self,
        media_path: str | Path,
        cache_path: str | Path,
        *,
        rebuild: bool = False,
    ) -> FeatureAudioSource:
        ...


class FeatureEmbedder(Protocol):
    def embed(self, window: AudioSlice) -> NDArray[np.float32]:
        ...


@dataclass(frozen=True, slots=True)
class FeatureBuildResult:
    dataset: FeatureDataset
    labels: LabelImportSummary
    output_path: Path


def _optional_values(values: Sequence[int | None]) -> NDArray[np.int64]:
    return np.asarray(
        [value if value is not None else _MISSING_INT for value in values],
        dtype=np.int64,
    )


def build_feature_dataset(
    *,
    match_id: str,
    video_path: str | Path,
    labels_path: str | Path,
    output_path: str | Path,
    audio_cache_path: str | Path | None = None,
    normalizer: FeatureAudioNormalizer | None = None,
    extractor_factory: Callable[[], FeatureEmbedder] | None = None,
    model_identifier: str = YAMNET_MODEL_HANDLE,
) -> FeatureBuildResult:
    """Build one match's features without segments.json or legacy WAV access."""

    imported = import_cheer_labels(labels_path, match_id=match_id)
    windows = imported.windows
    output = Path(output_path)
    if not windows:
        raise TrainingDataError("no reviewed labeled windows to build")

    cache = (
        Path(audio_cache_path)
        if audio_cache_path is not None
        else output.with_name(f"{output.stem}.audio.f32le")
    )
    audio_normalizer = normalizer or FFmpegAudioNormalizer()
    source = audio_normalizer.normalize(video_path, cache)
    embeddings: list[NDArray[np.float32]] = []
    try:
        for window in windows:
            if window.end_sec > source.duration_sec:
                raise TrainingDataError(
                    f"window {window.window_id} [{window.start_sec}, {window.end_sec}) "
                    f"exceeds normalized media duration {source.duration_sec}"
                )

        embedder = (
            extractor_factory()
            if extractor_factory is not None
            else YamNetEmbeddingExtractor()
        )
        for window in windows:
            audio = source.slice_absolute(window.start_sec, window.end_sec)
            embedding = np.asarray(embedder.embed(audio), dtype=np.float32)
            if embedding.shape != (YAMNET_EMBEDDING_SIZE,):
                raise TrainingDataError(
                    f"window {window.window_id}: embedding must have shape "
                    f"({YAMNET_EMBEDDING_SIZE},), got {embedding.shape}"
                )
            if not np.isfinite(embedding).all():
                raise TrainingDataError(
                    f"window {window.window_id}: embedding contains NaN or Inf"
                )
            embeddings.append(embedding)
    finally:
        source.close()

    dataset = FeatureDataset(
        embeddings=np.stack(embeddings).astype(np.float32, copy=False),
        labels=np.asarray([window.has_cheer for window in windows], dtype=np.uint8),
        window_ids=np.asarray([window.window_id for window in windows], dtype=np.int64),
        start_secs=np.asarray([window.start_sec for window in windows], dtype=np.float64),
        end_secs=np.asarray([window.end_sec for window in windows], dtype=np.float64),
        source_segment_ids=_optional_values(
            [window.source_segment_id for window in windows]
        ),
        cheer_confidences=_optional_values(
            [window.cheer_confidence for window in windows]
        ),
        match_id=match_id,
        model_identifier=model_identifier,
    )
    saved_path = dataset.save(output)
    return FeatureBuildResult(dataset, imported.summary, saved_path)
