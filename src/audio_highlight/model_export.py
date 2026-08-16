"""Train and export the frozen baseline as numeric deployable artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.exceptions import ConvergenceWarning

from audio_highlight.baseline import baseline_metadata
from audio_highlight.classifier import (
    BASELINE_ID,
    ExportedCheerDetector,
    build_baseline_classifier,
)
from audio_highlight.dataset import FeatureDataset
from audio_highlight.yamnet import YAMNET_EMBEDDING_SIZE, YAMNET_MODEL_HANDLE


class ModelTrainingError(ValueError):
    """Raised when final training data or export state violates the baseline."""


@dataclass(frozen=True, slots=True)
class TrainingDatasetRecord:
    match_id: str
    path: str
    sha256: str
    samples: int


@dataclass(frozen=True, slots=True)
class FinalTrainingData:
    embeddings: NDArray[np.float32]
    labels: NDArray[np.uint8]
    datasets: tuple[FeatureDataset, ...]
    records: tuple[TrainingDatasetRecord, ...]


@dataclass(frozen=True, slots=True)
class ModelTrainingResult:
    baseline_id: str
    training_matches: tuple[str, ...]
    sample_count: int
    positive_count: int
    negative_count: int
    embedding_dimension: int
    converged: bool
    iterations: int
    model_path: Path
    metadata_path: Path
    model_sha256: str
    model_size_bytes: int
    max_probability_difference: float
    binary_predictions_equal: bool


def load_final_training_data(
    feature_paths: Sequence[str | Path],
) -> FinalTrainingData:
    """Load compatible canonical NPZ datasets using their safe dataset loader."""

    if not feature_paths:
        raise ModelTrainingError("at least one feature dataset is required")
    paths = tuple(Path(path) for path in feature_paths)
    datasets = tuple(FeatureDataset.load(path) for path in paths)
    match_ids = [dataset.match_id for dataset in datasets]
    if len(set(match_ids)) != len(match_ids):
        raise ModelTrainingError("training feature datasets must have unique match_id")
    reference = datasets[0]
    frozen = baseline_metadata()
    expected = {
        "embedding_dimension": YAMNET_EMBEDDING_SIZE,
        "model_identifier": YAMNET_MODEL_HANDLE,
        "sample_rate_hz": frozen["audio"]["sample_rate_hz"],
        "window_sec": frozen["audio"]["window_sec"],
        "hop_sec": frozen["audio"]["hop_sec"],
        "post_padding_sec": frozen["audio"]["post_padding_sec"],
    }
    frozen_mismatches = [
        name for name, value in expected.items() if getattr(reference, name) != value
    ]
    if frozen_mismatches:
        raise ModelTrainingError(
            "feature dataset does not match frozen baseline: "
            + ", ".join(frozen_mismatches)
        )
    compatible_fields = (
        "embedding_dimension",
        "model_identifier",
        "sample_rate_hz",
        "window_sec",
        "hop_sec",
        "post_padding_sec",
        "label_source",
        "sampling_algorithm_version",
    )
    for dataset in datasets[1:]:
        mismatches = [
            name
            for name in compatible_fields
            if getattr(dataset, name) != getattr(reference, name)
        ]
        if mismatches:
            raise ModelTrainingError(
                f"incompatible feature datasets {reference.match_id!r} and "
                f"{dataset.match_id!r}: {', '.join(mismatches)} differ"
            )
    embeddings = np.concatenate(
        [dataset.embeddings for dataset in datasets], axis=0
    ).astype(np.float32, copy=False)
    labels = np.concatenate([dataset.labels for dataset in datasets], axis=0).astype(
        np.uint8, copy=False
    )
    if embeddings.shape[0] == 0 or np.unique(labels).size != 2:
        raise ModelTrainingError("final training data must contain both binary classes")
    records = tuple(
        TrainingDatasetRecord(
            match_id=dataset.match_id,
            path=_portable_path(path),
            sha256=_sha256(path),
            samples=int(dataset.embeddings.shape[0]),
        )
        for path, dataset in zip(paths, datasets, strict=True)
    )
    return FinalTrainingData(embeddings, labels, datasets, records)


def train_and_export_model(
    feature_paths: Sequence[str | Path],
    *,
    baseline_id: str = BASELINE_ID,
    output_dir: str | Path | None = None,
    trained_at: str | None = None,
) -> ModelTrainingResult:
    """Fit all supplied clean samples once and export a pure-numeric detector."""

    if baseline_id != BASELINE_ID:
        raise ModelTrainingError(
            f"unsupported baseline_id {baseline_id!r}; expected {BASELINE_ID!r}"
        )
    training = load_final_training_data(feature_paths)
    output = Path(output_dir or Path("artifacts/models") / baseline_id)
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "model.npz"
    metadata_path = output / "metadata.json"

    pipeline = build_baseline_classifier()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline.fit(training.embeddings, training.labels)
    converged = not any(
        issubclass(item.category, ConvergenceWarning) for item in caught
    )
    if not converged:
        raise ModelTrainingError("frozen Logistic Regression did not converge")

    scaler = pipeline.named_steps["standard_scaler"]
    classifier = pipeline.named_steps["logistic_regression"]
    if not np.array_equal(classifier.classes_, [0, 1]):
        raise ModelTrainingError("fitted classifier classes must equal [0, 1]")
    scaler_samples = int(np.asarray(scaler.n_samples_seen_).item())
    if scaler_samples != training.embeddings.shape[0]:
        raise ModelTrainingError("StandardScaler was not fit on all final samples")

    _write_model_npz(
        model_path,
        scaler_mean=np.asarray(scaler.mean_, dtype=np.float64),
        scaler_scale=np.asarray(scaler.scale_, dtype=np.float64),
        lr_coef=np.asarray(classifier.coef_[0], dtype=np.float64),
        lr_intercept=float(classifier.intercept_[0]),
        classes=np.asarray(classifier.classes_, dtype=np.int64),
    )
    model_sha256 = _sha256(model_path)
    metadata = _build_metadata(
        training,
        model_sha256=model_sha256,
        iterations=int(classifier.n_iter_[0]),
        scaler_fit_sample_count=scaler_samples,
        trained_at=trained_at or datetime.now(UTC).isoformat(),
    )
    _write_metadata(metadata_path, metadata)

    detector = ExportedCheerDetector.load(output)
    verification_embeddings = training.embeddings.astype(np.float64)
    sklearn_probabilities = np.asarray(
        pipeline.predict_proba(verification_embeddings)[:, 1], dtype=np.float64
    )
    manual_probabilities = detector.positive_probabilities(training.embeddings)
    max_difference = float(
        np.max(np.abs(sklearn_probabilities - manual_probabilities))
    )
    try:
        np.testing.assert_allclose(
            manual_probabilities,
            sklearn_probabilities,
            rtol=1e-10,
            atol=1e-12,
        )
    except AssertionError as exc:
        raise ModelTrainingError(
            "exported probabilities are not equivalent to sklearn"
        ) from exc
    sklearn_predictions = np.asarray(pipeline.predict(verification_embeddings))
    manual_predictions = detector.predict_embeddings(training.embeddings)
    predictions_equal = bool(
        np.array_equal(manual_predictions, sklearn_predictions)
    )
    if not predictions_equal:
        raise ModelTrainingError(
            "exported binary predictions are not equivalent to sklearn"
        )

    return ModelTrainingResult(
        baseline_id=baseline_id,
        training_matches=tuple(dataset.match_id for dataset in training.datasets),
        sample_count=int(training.labels.size),
        positive_count=int(np.count_nonzero(training.labels == 1)),
        negative_count=int(np.count_nonzero(training.labels == 0)),
        embedding_dimension=int(training.embeddings.shape[1]),
        converged=True,
        iterations=int(classifier.n_iter_[0]),
        model_path=model_path,
        metadata_path=metadata_path,
        model_sha256=model_sha256,
        model_size_bytes=model_path.stat().st_size,
        max_probability_difference=max_difference,
        binary_predictions_equal=predictions_equal,
    )


def _write_model_npz(
    path: Path,
    *,
    scaler_mean: NDArray[np.float64],
    scaler_scale: NDArray[np.float64],
    lr_coef: NDArray[np.float64],
    lr_intercept: float,
    classes: NDArray[np.int64],
) -> None:
    temporary = path.with_name(f".{path.stem}.tmp.npz")
    try:
        np.savez_compressed(
            temporary,
            scaler_mean=scaler_mean,
            scaler_scale=scaler_scale,
            lr_coef=lr_coef,
            lr_intercept=np.asarray(lr_intercept, dtype=np.float64),
            classes=classes,
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _build_metadata(
    training: FinalTrainingData,
    *,
    model_sha256: str,
    iterations: int,
    scaler_fit_sample_count: int,
    trained_at: str,
) -> dict[str, Any]:
    frozen = baseline_metadata()
    sample_count = int(training.labels.size)
    return {
        "model_id": BASELINE_ID,
        "model_type": "logistic_regression",
        "model_sha256": model_sha256,
        "feature_extractor": frozen["feature_extractor"],
        "audio": frozen["audio"],
        "classifier": {
            key: value
            for key, value in frozen["classifier"].items()
            if key != "type"
        },
        "training": {
            "matches": [dataset.match_id for dataset in training.datasets],
            "sample_count": sample_count,
            "positive_count": int(np.count_nonzero(training.labels == 1)),
            "negative_count": int(np.count_nonzero(training.labels == 0)),
            "converged": True,
            "iterations": iterations,
            "scaler_fit_sample_count": scaler_fit_sample_count,
            "sklearn_version": version("scikit-learn"),
            "numpy_version": np.__version__,
            "trained_at": trained_at,
            "datasets": [
                {
                    "match_id": record.match_id,
                    "path": record.path,
                    "sha256": record.sha256,
                    "samples": record.samples,
                }
                for record in training.records
            ],
        },
    }


def _write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)
            file.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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
        raise ModelTrainingError(f"cannot read artifact {path}: {exc}") from exc
    return digest.hexdigest()
