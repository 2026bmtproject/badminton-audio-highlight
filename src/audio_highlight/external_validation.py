"""Untouched-match evaluation using only a frozen exported detector."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from audio_highlight.calibration import (
    CALIBRATION_BIN_COUNT,
    PredictionRecord,
    ProbabilitySummary,
    diagnose_calibration,
)
from audio_highlight.classifier import ExportedCheerDetector
from audio_highlight.dataset import FeatureDataset
from audio_highlight.evaluation import (
    BinaryClassificationMetrics,
    binary_metrics_dict,
    compute_binary_metrics,
    threshold_probabilities,
)


class ExternalValidationError(ValueError):
    """Raised when frozen-model external validation would be invalid."""


@dataclass(frozen=True, slots=True)
class ExternalPrediction:
    """One frozen-detector prediction retaining feature-row identity."""

    match_id: str
    sample_rank: int
    segment_index: int
    window_index_in_segment: int
    start_sec: float
    end_sec: float
    true_label: int
    predicted_label: int
    positive_probability: float


@dataclass(frozen=True, slots=True)
class ExternalValidationResult:
    """Metrics and provenance for one untouched external match."""

    validation_type: str
    baseline_id: str
    model_sha256: str
    threshold: float
    training_matches: tuple[str, ...]
    match_id: str
    feature_sha256: str
    sample_count: int
    positive_count: int
    negative_count: int
    metrics: BinaryClassificationMetrics
    brier_score: float
    log_loss: float
    ece: float
    observed_prevalence: float
    predicted_positive_rate: float
    positive_probability_summary: ProbabilitySummary
    negative_probability_summary: ProbabilitySummary
    predictions: tuple[ExternalPrediction, ...]


@dataclass(frozen=True, slots=True)
class ExternalValidationArtifactPaths:
    predictions_csv: Path
    metrics_json: Path


def evaluate_external_match(
    feature_path: str | Path,
    model_dir: str | Path,
    *,
    expected_match_id: str | None = None,
) -> ExternalValidationResult:
    """Evaluate one test-only feature dataset without fitting any state."""

    source = Path(feature_path)
    model_directory = Path(model_dir)
    detector = ExportedCheerDetector.load(model_directory)
    dataset = FeatureDataset.load(source)
    metadata = detector.metadata
    training_matches = tuple(metadata["training"]["matches"])
    if dataset.match_id in training_matches:
        raise ExternalValidationError(
            f"test match {dataset.match_id!r} appears in model training matches"
        )
    if expected_match_id is not None and dataset.match_id != expected_match_id:
        raise ExternalValidationError(
            f"feature match_id {dataset.match_id!r} does not match "
            f"requested {expected_match_id!r}"
        )
    _validate_feature_compatibility(dataset, metadata)

    probabilities = detector.positive_probabilities(dataset.embeddings)
    predicted = threshold_probabilities(probabilities, threshold=detector.threshold)
    metrics = compute_binary_metrics(dataset.labels, predicted, probabilities)
    records = tuple(
        PredictionRecord(
            match_id=dataset.match_id,
            true_label=int(dataset.labels[index]),
            predicted_label=int(predicted[index]),
            positive_probability=float(probabilities[index]),
        )
        for index in range(dataset.labels.size)
    )
    calibration = diagnose_calibration(records, n_bins=CALIBRATION_BIN_COUNT)
    descriptive = calibration.matches[0]
    predictions = tuple(
        ExternalPrediction(
            match_id=dataset.match_id,
            sample_rank=int(dataset.sample_ranks[index]),
            segment_index=int(dataset.segment_indices[index]),
            window_index_in_segment=int(dataset.window_indices[index]),
            start_sec=float(dataset.start_secs[index]),
            end_sec=float(dataset.end_secs[index]),
            true_label=int(dataset.labels[index]),
            predicted_label=int(predicted[index]),
            positive_probability=float(probabilities[index]),
        )
        for index in range(dataset.labels.size)
    )
    return ExternalValidationResult(
        validation_type="untouched_external_match",
        baseline_id=str(metadata["model_id"]),
        model_sha256=_sha256(model_directory / "model.npz"),
        threshold=detector.threshold,
        training_matches=training_matches,
        match_id=dataset.match_id,
        feature_sha256=_sha256(source),
        sample_count=int(dataset.labels.size),
        positive_count=int(np.count_nonzero(dataset.labels == 1)),
        negative_count=int(np.count_nonzero(dataset.labels == 0)),
        metrics=metrics,
        brier_score=descriptive.brier_score,
        log_loss=descriptive.log_loss,
        ece=descriptive.ece,
        observed_prevalence=descriptive.prevalence,
        predicted_positive_rate=descriptive.predicted_positive_rate,
        positive_probability_summary=descriptive.positive_probability_summary,
        negative_probability_summary=descriptive.negative_probability_summary,
        predictions=predictions,
    )


def write_external_validation_artifacts(
    result: ExternalValidationResult,
    output_dir: str | Path,
) -> ExternalValidationArtifactPaths:
    """Atomically write external predictions and metrics separately from LOMO."""

    output = Path(output_dir)
    if _is_cross_match_path(output):
        raise ExternalValidationError(
            "external validation output must not use artifacts/cross_match"
        )
    output.mkdir(parents=True, exist_ok=True)
    predictions_path = output / "predictions.csv"
    metrics_path = output / "metrics.json"
    temporary_predictions = output / f".{predictions_path.name}.tmp"
    temporary_metrics = output / f".{metrics_path.name}.tmp"
    try:
        with temporary_predictions.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file, fieldnames=list(ExternalPrediction.__dataclass_fields__)
            )
            writer.writeheader()
            for prediction in result.predictions:
                writer.writerow(asdict(prediction))
        with temporary_metrics.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(_result_dict(result), file, ensure_ascii=False, indent=2)
            file.write("\n")
        os.replace(temporary_predictions, predictions_path)
        os.replace(temporary_metrics, metrics_path)
    finally:
        temporary_predictions.unlink(missing_ok=True)
        temporary_metrics.unlink(missing_ok=True)
    return ExternalValidationArtifactPaths(predictions_path, metrics_path)


def _validate_feature_compatibility(
    dataset: FeatureDataset, metadata: dict[str, Any]
) -> None:
    feature = metadata["feature_extractor"]
    audio = metadata["audio"]
    checks = (
        ("embedding_dimension", dataset.embedding_dimension, feature["embedding_dimension"]),
        ("model_identifier", dataset.model_identifier, feature["model_identifier"]),
        ("sample_rate_hz", dataset.sample_rate_hz, audio["sample_rate_hz"]),
        ("window_sec", dataset.window_sec, audio["window_sec"]),
        ("hop_sec", dataset.hop_sec, audio["hop_sec"]),
        ("post_padding_sec", dataset.post_padding_sec, audio["post_padding_sec"]),
    )
    mismatches = [name for name, actual, expected in checks if actual != expected]
    if mismatches:
        raise ExternalValidationError(
            "feature dataset is incompatible with frozen model: "
            + ", ".join(mismatches)
        )


def _result_dict(result: ExternalValidationResult) -> dict[str, Any]:
    metrics = binary_metrics_dict(result.metrics)
    metrics.update(
        {
            "brier_score": result.brier_score,
            "log_loss": result.log_loss,
            "ece": result.ece,
        }
    )
    return {
        "validation_type": result.validation_type,
        "baseline_id": result.baseline_id,
        "model": {
            "model_sha256": result.model_sha256,
            "threshold": result.threshold,
            "training_matches": list(result.training_matches),
        },
        "test": {
            "match_id": result.match_id,
            "sample_count": result.sample_count,
            "positive_count": result.positive_count,
            "negative_count": result.negative_count,
            "feature_sha256": result.feature_sha256,
        },
        "metrics": metrics,
        "descriptive": {
            "observed_prevalence": result.observed_prevalence,
            "predicted_positive_rate": result.predicted_positive_rate,
            "positive_probability": asdict(result.positive_probability_summary),
            "negative_probability": asdict(result.negative_probability_summary),
            "ece_bin_count": CALIBRATION_BIN_COUNT,
            "ece_bin_strategy": "uniform",
        },
    }


def _is_cross_match_path(path: Path) -> bool:
    resolved_parts = path.resolve(strict=False).parts
    return any(
        first.lower() == "artifacts" and second.lower() == "cross_match"
        for first, second in zip(resolved_parts, resolved_parts[1:])
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ExternalValidationError(f"cannot hash artifact {path}: {exc}") from exc
    return digest.hexdigest()
