"""Leakage-safe leave-one-match-out classifier evaluation."""

from __future__ import annotations

import csv
import json
import os
import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from audio_highlight.baseline import BASELINE_ID
from audio_highlight.classifier import (
    LOGISTIC_REGRESSION_C,
    LOGISTIC_REGRESSION_MAX_ITER,
    PREDICTION_THRESHOLD,
    build_baseline_classifier,
)
from audio_highlight.dataset import FeatureDataset
_METRIC_NAMES = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
)


class EvaluationError(ValueError):
    """Raised when cross-match evaluation inputs or state are invalid."""


@dataclass(frozen=True, slots=True)
class BinaryClassificationMetrics:
    """Binary metrics for one held-out match, with positive class ``1``."""

    sample_count: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    average_precision: float | None
    tn: int
    fp: int
    fn: int
    tp: int


@dataclass(frozen=True, slots=True)
class EvaluationFold:
    """Training membership and results for one held-out match."""

    train_matches: tuple[str, ...]
    test_match: str
    train_samples: int
    test_samples: int
    metrics: BinaryClassificationMetrics
    converged: bool
    iterations: int


@dataclass(frozen=True, slots=True)
class WindowPrediction:
    """One out-of-fold prediction retaining canonical window provenance."""

    test_match_id: str
    sample_rank: int
    segment_index: int
    window_index_in_segment: int
    start_sec: float
    end_sec: float
    true_label: int
    predicted_label: int
    positive_probability: float


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """All leave-one-match-out folds and their equally weighted macro mean."""

    folds: tuple[EvaluationFold, ...]
    macro_mean: dict[str, float | None]
    predictions: tuple[WindowPrediction, ...]


@dataclass(frozen=True, slots=True)
class EvaluationArtifactPaths:
    predictions_csv: Path
    metrics_json: Path


def threshold_probabilities(
    probabilities: NDArray[np.floating[Any]],
    *,
    threshold: float = PREDICTION_THRESHOLD,
) -> NDArray[np.uint8]:
    """Map probabilities greater than or equal to the fixed threshold to class 1."""

    values = np.asarray(probabilities)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise EvaluationError("probabilities must be a finite one-dimensional array")
    if not 0.0 <= threshold <= 1.0:
        raise EvaluationError("threshold must be between 0 and 1")
    return (values >= threshold).astype(np.uint8)


def compute_binary_metrics(
    true_labels: NDArray[np.integer[Any]],
    predicted_labels: NDArray[np.integer[Any]],
    positive_probabilities: NDArray[np.floating[Any]],
) -> BinaryClassificationMetrics:
    """Compute deterministic positive-class metrics and TN/FP/FN/TP counts."""

    true = np.asarray(true_labels)
    predicted = np.asarray(predicted_labels)
    probabilities = np.asarray(positive_probabilities)
    if true.ndim != 1 or predicted.shape != true.shape or probabilities.shape != true.shape:
        raise EvaluationError("labels and probabilities must be aligned 1-D arrays")
    if true.size == 0:
        raise EvaluationError("cannot compute metrics for an empty test match")
    if not np.isin(true, [0, 1]).all() or not np.isin(predicted, [0, 1]).all():
        raise EvaluationError("true and predicted labels must be binary")
    if not np.isfinite(probabilities).all():
        raise EvaluationError("positive probabilities must be finite")

    tn, fp, fn, tp = confusion_matrix(true, predicted, labels=[0, 1]).ravel()
    has_both_classes = np.unique(true).size == 2
    return BinaryClassificationMetrics(
        sample_count=int(true.size),
        accuracy=float(accuracy_score(true, predicted)),
        precision=float(precision_score(true, predicted, pos_label=1, zero_division=0)),
        recall=float(recall_score(true, predicted, pos_label=1, zero_division=0)),
        f1=float(f1_score(true, predicted, pos_label=1, zero_division=0)),
        roc_auc=float(roc_auc_score(true, probabilities)) if has_both_classes else None,
        average_precision=(
            float(average_precision_score(true, probabilities))
            if has_both_classes
            else None
        ),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )


def macro_mean_metrics(
    metrics: Sequence[BinaryClassificationMetrics],
) -> dict[str, float | None]:
    """Average fold metrics with one equal vote per held-out match."""

    if not metrics:
        raise EvaluationError("at least one fold is required for macro metrics")
    result: dict[str, float | None] = {}
    for name in _METRIC_NAMES:
        values = [getattr(item, name) for item in metrics]
        result[name] = (
            None if any(value is None for value in values) else float(np.mean(values))
        )
    return result


def _validate_datasets(datasets: Sequence[FeatureDataset]) -> None:
    if len(datasets) < 2:
        raise EvaluationError("at least two distinct match feature datasets are required")
    match_ids = [dataset.match_id for dataset in datasets]
    if len(set(match_ids)) != len(match_ids):
        raise EvaluationError("feature datasets must have distinct match_id values")
    if any(dataset.embeddings.shape[0] == 0 for dataset in datasets):
        raise EvaluationError("feature datasets must not be empty")

    reference = datasets[0]
    for dataset in datasets[1:]:
        mismatches: list[str] = []
        if dataset.embedding_dimension != reference.embedding_dimension:
            mismatches.append("embedding_dimension")
        if dataset.sample_rate_hz != reference.sample_rate_hz:
            mismatches.append("sample_rate_hz")
        if dataset.model_identifier != reference.model_identifier:
            mismatches.append("model_identifier")
        if mismatches:
            fields = ", ".join(mismatches)
            raise EvaluationError(
                f"incompatible feature datasets {reference.match_id!r} and "
                f"{dataset.match_id!r}: {fields} differ"
            )


def evaluate_cross_match(
    feature_paths: Sequence[str | Path],
) -> EvaluationResult:
    """Run N-fold leave-one-match-out evaluation without window-level splitting."""

    datasets = tuple(FeatureDataset.load(path) for path in feature_paths)
    _validate_datasets(datasets)

    folds: list[EvaluationFold] = []
    predictions: list[WindowPrediction] = []
    for held_out_index, test_dataset in enumerate(datasets):
        training = tuple(
            dataset for index, dataset in enumerate(datasets) if index != held_out_index
        )
        train_embeddings = np.concatenate(
            [dataset.embeddings for dataset in training], axis=0
        )
        train_labels = np.concatenate([dataset.labels for dataset in training], axis=0)
        if np.unique(train_labels).size != 2:
            raise EvaluationError(
                f"training matches for held-out {test_dataset.match_id!r} "
                "must contain both binary classes"
            )

        classifier = build_baseline_classifier()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            classifier.fit(train_embeddings, train_labels)
        converged = not any(
            issubclass(warning.category, ConvergenceWarning) for warning in caught
        )

        logistic_regression = classifier.named_steps["logistic_regression"]
        positive_class_index = int(np.flatnonzero(logistic_regression.classes_ == 1)[0])
        probabilities = classifier.predict_proba(test_dataset.embeddings)[
            :, positive_class_index
        ]
        predicted = threshold_probabilities(probabilities)
        metrics = compute_binary_metrics(test_dataset.labels, predicted, probabilities)
        folds.append(
            EvaluationFold(
                train_matches=tuple(dataset.match_id for dataset in training),
                test_match=test_dataset.match_id,
                train_samples=int(train_embeddings.shape[0]),
                test_samples=int(test_dataset.embeddings.shape[0]),
                metrics=metrics,
                converged=converged,
                iterations=int(logistic_regression.n_iter_[0]),
            )
        )

        for index in range(test_dataset.embeddings.shape[0]):
            predictions.append(
                WindowPrediction(
                    test_match_id=test_dataset.match_id,
                    sample_rank=int(test_dataset.sample_ranks[index]),
                    segment_index=int(test_dataset.segment_indices[index]),
                    window_index_in_segment=int(test_dataset.window_indices[index]),
                    start_sec=float(test_dataset.start_secs[index]),
                    end_sec=float(test_dataset.end_secs[index]),
                    true_label=int(test_dataset.labels[index]),
                    predicted_label=int(predicted[index]),
                    positive_probability=float(probabilities[index]),
                )
            )

    return EvaluationResult(
        folds=tuple(folds),
        macro_mean=macro_mean_metrics([fold.metrics for fold in folds]),
        predictions=tuple(predictions),
    )


def binary_metrics_dict(metrics: BinaryClassificationMetrics) -> dict[str, Any]:
    """Serialize binary metrics with confusion order ``[[TN, FP], [FN, TP]]``."""

    values = asdict(metrics)
    values["confusion_matrix"] = [
        [values.pop("tn"), values.pop("fp")],
        [values.pop("fn"), values.pop("tp")],
    ]
    return values


def _summary_dict(result: EvaluationResult) -> dict[str, Any]:
    return {
        "model": {
            "baseline_id": BASELINE_ID,
            "type": "logistic_regression",
            "C": LOGISTIC_REGRESSION_C,
            "max_iter": LOGISTIC_REGRESSION_MAX_ITER,
            "solver": "lbfgs",
            "class_weight": None,
            "threshold": PREDICTION_THRESHOLD,
            "preprocessing": "training-fold StandardScaler",
        },
        "folds": [
            {
                "train_matches": list(fold.train_matches),
                "test_match": fold.test_match,
                "train_samples": fold.train_samples,
                "test_samples": fold.test_samples,
                "converged": fold.converged,
                "iterations": fold.iterations,
                "metrics": binary_metrics_dict(fold.metrics),
            }
            for fold in result.folds
        ],
        "macro_mean": result.macro_mean,
    }


def write_evaluation_artifacts(
    result: EvaluationResult,
    output_dir: str | Path,
) -> EvaluationArtifactPaths:
    """Atomically write out-of-fold predictions and the JSON metric summary."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions_path = output / "cross_match_predictions.csv"
    metrics_path = output / "cross_match_metrics.json"
    temporary_predictions = output / f".{predictions_path.name}.tmp"
    temporary_metrics = output / f".{metrics_path.name}.tmp"

    fieldnames = list(WindowPrediction.__dataclass_fields__)
    try:
        with temporary_predictions.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for prediction in result.predictions:
                writer.writerow(asdict(prediction))
        with temporary_metrics.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(_summary_dict(result), file, ensure_ascii=False, indent=2)
            file.write("\n")
        os.replace(temporary_predictions, predictions_path)
        os.replace(temporary_metrics, metrics_path)
    finally:
        temporary_predictions.unlink(missing_ok=True)
        temporary_metrics.unlink(missing_ok=True)

    return EvaluationArtifactPaths(predictions_path, metrics_path)
