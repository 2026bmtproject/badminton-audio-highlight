"""Descriptive calibration diagnostics over existing OOF predictions only."""

from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from audio_highlight.baseline import BASELINE_ID
from audio_highlight.classifier import PREDICTION_THRESHOLD

DEFAULT_PREDICTIONS_PATH = Path(
    "artifacts/cross_match/evaluation/cross_match_predictions.csv"
)
DEFAULT_CALIBRATION_OUTPUT_DIR = Path(
    "artifacts/cross_match/evaluation/calibration"
)
CALIBRATION_BIN_COUNT = 10
CALIBRATION_BIN_STRATEGY = "uniform"
_REQUIRED_FIELDS = {
    "test_match_id",
    "true_label",
    "predicted_label",
    "positive_probability",
}
_SUMMARY_FIELDS = (
    "match_id",
    "sample_count",
    "prevalence",
    "predicted_positive_rate",
    "brier_score",
    "log_loss",
    "positive_prob_mean",
    "positive_prob_median",
    "negative_prob_mean",
    "negative_prob_median",
    "roc_auc",
    "average_precision",
    "ece",
)


class CalibrationError(ValueError):
    """Raised when OOF predictions or diagnostic settings are invalid."""


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    match_id: str
    true_label: int
    predicted_label: int
    positive_probability: float


@dataclass(frozen=True, slots=True)
class ProbabilitySummary:
    mean: float
    median: float
    std: float
    q1: float
    q3: float
    min: float
    max: float


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    bin_index: int
    lower_bound: float
    upper_bound: float
    sample_count: int
    mean_predicted_probability: float | None
    fraction_positive: float | None


@dataclass(frozen=True, slots=True)
class MatchCalibrationMetrics:
    match_id: str
    sample_count: int
    positive_count: int
    negative_count: int
    prevalence: float
    predicted_positive_count: int
    predicted_positive_rate: float
    brier_score: float
    log_loss: float
    roc_auc: float | None
    average_precision: float | None
    ece: float
    positive_probability_summary: ProbabilitySummary
    negative_probability_summary: ProbabilitySummary
    calibration_bins: tuple[CalibrationBin, ...]


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    baseline_id: str
    threshold: float
    n_bins: int
    strategy: str
    matches: tuple[MatchCalibrationMetrics, ...]
    records: tuple[PredictionRecord, ...]


@dataclass(frozen=True, slots=True)
class CalibrationArtifactPaths:
    metrics_json: Path
    summary_csv: Path
    probability_distribution_plots: tuple[Path, ...]
    reliability_plots: tuple[Path, ...]
    combined_reliability_plot: Path


def load_predictions(path: str | Path) -> tuple[PredictionRecord, ...]:
    """Parse canonical OOF predictions without loading features or a model."""

    source = Path(path)
    records: list[PredictionRecord] = []
    try:
        with source.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            missing = _REQUIRED_FIELDS - set(reader.fieldnames or ())
            if missing:
                raise CalibrationError(
                    "prediction CSV is missing fields: "
                    + ", ".join(sorted(missing))
                )
            for row_number, row in enumerate(reader, start=2):
                try:
                    record = PredictionRecord(
                        match_id=row["test_match_id"].strip(),
                        true_label=int(row["true_label"]),
                        predicted_label=int(row["predicted_label"]),
                        positive_probability=float(row["positive_probability"]),
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise CalibrationError(
                        f"prediction CSV row {row_number}: invalid value: {exc}"
                    ) from exc
                _validate_record(record, row_number=row_number)
                records.append(record)
    except OSError:
        raise
    if not records:
        raise CalibrationError("prediction CSV contains no rows")
    return tuple(records)


def _validate_record(record: PredictionRecord, *, row_number: int) -> None:
    if not record.match_id:
        raise CalibrationError(f"prediction CSV row {row_number}: empty match ID")
    if record.true_label not in {0, 1} or record.predicted_label not in {0, 1}:
        raise CalibrationError(
            f"prediction CSV row {row_number}: labels must be binary"
        )
    if (
        not math.isfinite(record.positive_probability)
        or not 0.0 <= record.positive_probability <= 1.0
    ):
        raise CalibrationError(
            f"prediction CSV row {row_number}: probability must be in [0, 1]"
        )
    expected = int(record.positive_probability >= PREDICTION_THRESHOLD)
    if record.predicted_label != expected:
        raise CalibrationError(
            f"prediction CSV row {row_number}: predicted_label does not match "
            f"the frozen threshold {PREDICTION_THRESHOLD}"
        )


def summarize_probabilities(
    probabilities: NDArray[np.floating[Any]],
) -> ProbabilitySummary:
    """Compute deterministic population statistics for one non-empty class."""

    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise CalibrationError("probability summary requires a non-empty finite vector")
    return ProbabilitySummary(
        mean=float(np.mean(values)),
        median=float(np.median(values)),
        std=float(np.std(values, ddof=0)),
        q1=float(np.quantile(values, 0.25)),
        q3=float(np.quantile(values, 0.75)),
        min=float(np.min(values)),
        max=float(np.max(values)),
    )


def compute_calibration_bins(
    labels: NDArray[np.integer[Any]],
    probabilities: NDArray[np.floating[Any]],
    *,
    n_bins: int = CALIBRATION_BIN_COUNT,
) -> tuple[CalibrationBin, ...]:
    """Compute fixed uniform bins, retaining empty bins with null statistics.

    Bins are ``[lower, upper)`` except the final bin, which includes probability
    ``1.0``. Empty bins have ``sample_count=0`` and both statistics set to null.
    """

    true = np.asarray(labels)
    scores = np.asarray(probabilities, dtype=np.float64)
    if n_bins <= 0:
        raise CalibrationError("n_bins must be positive")
    if true.ndim != 1 or scores.shape != true.shape or true.size == 0:
        raise CalibrationError("calibration bins require aligned non-empty vectors")
    if not np.isin(true, [0, 1]).all():
        raise CalibrationError("calibration labels must be binary")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise CalibrationError("calibration probabilities must be in [0, 1]")

    bin_indices = np.minimum((scores * n_bins).astype(np.int64), n_bins - 1)
    bins: list[CalibrationBin] = []
    for index in range(n_bins):
        selected = bin_indices == index
        count = int(np.count_nonzero(selected))
        bins.append(
            CalibrationBin(
                bin_index=index,
                lower_bound=index / n_bins,
                upper_bound=(index + 1) / n_bins,
                sample_count=count,
                mean_predicted_probability=(
                    float(np.mean(scores[selected])) if count else None
                ),
                fraction_positive=float(np.mean(true[selected])) if count else None,
            )
        )
    return tuple(bins)


def expected_calibration_error(
    bins: Sequence[CalibrationBin], *, sample_count: int
) -> float:
    """Compute descriptive ECE as Σ(n_b/N)*|observed_b-mean_score_b|.

    ECE is reported alongside Brier score, log loss, and reliability diagrams;
    it is not treated as a sufficient calibration assessment by itself.
    """

    if sample_count <= 0 or sum(item.sample_count for item in bins) != sample_count:
        raise CalibrationError("ECE bins must account for every sample exactly once")
    return float(
        sum(
            (item.sample_count / sample_count)
            * abs(item.fraction_positive - item.mean_predicted_probability)
            for item in bins
            if item.sample_count
            and item.fraction_positive is not None
            and item.mean_predicted_probability is not None
        )
    )


def diagnose_calibration(
    records: Sequence[PredictionRecord],
    *,
    n_bins: int = CALIBRATION_BIN_COUNT,
) -> CalibrationResult:
    """Describe fixed OOF scores; never fit a model, calibrator, or threshold."""

    if not records:
        raise CalibrationError("at least one prediction record is required")
    grouped: dict[str, list[PredictionRecord]] = defaultdict(list)
    for record in records:
        _validate_record(record, row_number=0)
        grouped[record.match_id].append(record)

    matches: list[MatchCalibrationMetrics] = []
    for match_id in sorted(grouped):
        match_records = grouped[match_id]
        labels = np.asarray(
            [record.true_label for record in match_records], dtype=np.uint8
        )
        probabilities = np.asarray(
            [record.positive_probability for record in match_records],
            dtype=np.float64,
        )
        predicted = np.asarray(
            [record.predicted_label for record in match_records], dtype=np.uint8
        )
        positive_probabilities = probabilities[labels == 1]
        negative_probabilities = probabilities[labels == 0]
        if positive_probabilities.size == 0 or negative_probabilities.size == 0:
            raise CalibrationError(
                f"match {match_id!r} must contain both binary classes"
            )
        bins = compute_calibration_bins(labels, probabilities, n_bins=n_bins)
        sample_count = int(labels.size)
        positive_count = int(np.count_nonzero(labels == 1))
        predicted_positive_count = int(np.count_nonzero(predicted == 1))
        matches.append(
            MatchCalibrationMetrics(
                match_id=match_id,
                sample_count=sample_count,
                positive_count=positive_count,
                negative_count=sample_count - positive_count,
                prevalence=positive_count / sample_count,
                predicted_positive_count=predicted_positive_count,
                predicted_positive_rate=predicted_positive_count / sample_count,
                brier_score=float(
                    brier_score_loss(labels, probabilities, pos_label=1)
                ),
                log_loss=float(log_loss(labels, probabilities, labels=[0, 1])),
                roc_auc=float(roc_auc_score(labels, probabilities)),
                average_precision=float(
                    average_precision_score(labels, probabilities)
                ),
                ece=expected_calibration_error(bins, sample_count=sample_count),
                positive_probability_summary=summarize_probabilities(
                    positive_probabilities
                ),
                negative_probability_summary=summarize_probabilities(
                    negative_probabilities
                ),
                calibration_bins=bins,
            )
        )
    return CalibrationResult(
        baseline_id=BASELINE_ID,
        threshold=PREDICTION_THRESHOLD,
        n_bins=n_bins,
        strategy=CALIBRATION_BIN_STRATEGY,
        matches=tuple(matches),
        records=tuple(records),
    )


def diagnose_calibration_file(
    predictions_path: str | Path = DEFAULT_PREDICTIONS_PATH,
) -> CalibrationResult:
    return diagnose_calibration(load_predictions(predictions_path))


def compare_matches(
    result: CalibrationResult, first_match_id: str, second_match_id: str
) -> dict[str, Any]:
    """Return a data-derived comparison without claiming a causal domain shift."""

    indexed = {item.match_id: item for item in result.matches}
    try:
        first = indexed[first_match_id]
        second = indexed[second_match_id]
    except KeyError as exc:
        raise CalibrationError(f"comparison match not found: {exc.args[0]}") from exc
    return {
        "first_match_id": first_match_id,
        "second_match_id": second_match_id,
        "same_observed_prevalence": math.isclose(
            first.prevalence, second.prevalence, rel_tol=0.0, abs_tol=1e-12
        ),
        "prevalence_difference": second.prevalence - first.prevalence,
        "predicted_positive_rate_difference": (
            second.predicted_positive_rate - first.predicted_positive_rate
        ),
        "positive_probability_median_difference": (
            second.positive_probability_summary.median
            - first.positive_probability_summary.median
        ),
        "negative_probability_median_difference": (
            second.negative_probability_summary.median
            - first.negative_probability_summary.median
        ),
        "brier_score_difference": second.brier_score - first.brier_score,
        "log_loss_difference": second.log_loss - first.log_loss,
    }


def _result_dict(result: CalibrationResult) -> dict[str, Any]:
    return {
        "baseline_id": result.baseline_id,
        "threshold": result.threshold,
        "n_bins": result.n_bins,
        "strategy": result.strategy,
        "matches": [asdict(item) for item in result.matches],
    }


def write_calibration_artifacts(
    result: CalibrationResult,
    output_dir: str | Path = DEFAULT_CALIBRATION_OUTPUT_DIR,
) -> CalibrationArtifactPaths:
    """Write JSON, CSV, histograms, and reliability diagrams atomically."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "calibration_metrics.json"
    summary_path = output / "calibration_summary.csv"
    _write_json_atomic(_result_dict(result), metrics_path)
    _write_summary_csv(result, summary_path)
    distribution_paths, reliability_paths, combined_path = _write_plots(
        result, output
    )
    return CalibrationArtifactPaths(
        metrics_json=metrics_path,
        summary_csv=summary_path,
        probability_distribution_plots=distribution_paths,
        reliability_plots=reliability_paths,
        combined_reliability_plot=combined_path,
    )


def _write_json_atomic(value: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(value, file, ensure_ascii=False, indent=2)
            file.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_summary_csv(result: CalibrationResult, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=_SUMMARY_FIELDS)
            writer.writeheader()
            for item in result.matches:
                writer.writerow(
                    {
                        "match_id": item.match_id,
                        "sample_count": item.sample_count,
                        "prevalence": item.prevalence,
                        "predicted_positive_rate": item.predicted_positive_rate,
                        "brier_score": item.brier_score,
                        "log_loss": item.log_loss,
                        "positive_prob_mean": (
                            item.positive_probability_summary.mean
                        ),
                        "positive_prob_median": (
                            item.positive_probability_summary.median
                        ),
                        "negative_prob_mean": (
                            item.negative_probability_summary.mean
                        ),
                        "negative_prob_median": (
                            item.negative_probability_summary.median
                        ),
                        "roc_auc": item.roc_auc,
                        "average_precision": item.average_precision,
                        "ece": item.ece,
                    }
                )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_plots(
    result: CalibrationResult, output: Path
) -> tuple[tuple[Path, ...], tuple[Path, ...], Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    distributions: list[Path] = []
    reliability: list[Path] = []
    records_by_match = _records_for_plotting(result.records)
    for item in result.matches:
        labels, probabilities = records_by_match[item.match_id]
        distribution_path = output / f"{item.match_id}_probability_distribution.png"
        figure, axis = plt.subplots(figsize=(7, 4.5))
        histogram_bins = np.linspace(0.0, 1.0, 21)
        axis.hist(
            probabilities[labels == 0],
            bins=histogram_bins,
            alpha=0.65,
            label="true label 0",
        )
        axis.hist(
            probabilities[labels == 1],
            bins=histogram_bins,
            alpha=0.65,
            label="true label 1",
        )
        axis.axvline(result.threshold, color="black", linestyle="--", label="threshold 0.5")
        axis.set(xlim=(0, 1), xlabel="Positive probability", ylabel="Count")
        axis.set_title(
            f"{item.match_id} | N={item.sample_count} | prevalence={item.prevalence:.0%}"
        )
        axis.legend()
        figure.tight_layout()
        _save_figure_atomic(figure, distribution_path)
        plt.close(figure)
        distributions.append(distribution_path)

        reliability_path = output / f"{item.match_id}_reliability.png"
        figure, axis = plt.subplots(figsize=(5, 5))
        _plot_reliability(axis, item, label=item.match_id)
        axis.legend()
        figure.tight_layout()
        _save_figure_atomic(figure, reliability_path)
        plt.close(figure)
        reliability.append(reliability_path)

    combined_path = output / "all_matches_reliability.png"
    figure, axis = plt.subplots(figsize=(6, 6))
    for item in result.matches:
        _plot_reliability(axis, item, label=item.match_id, add_diagonal=False)
    axis.plot([0, 1], [0, 1], color="black", linestyle="--", label="perfect")
    axis.set_title("Reliability | all matches")
    axis.legend()
    figure.tight_layout()
    _save_figure_atomic(figure, combined_path)
    plt.close(figure)
    return tuple(distributions), tuple(reliability), combined_path


def _records_for_plotting(
    records: Sequence[PredictionRecord],
) -> dict[str, tuple[NDArray[np.uint8], NDArray[np.float64]]]:
    grouped: dict[str, list[PredictionRecord]] = defaultdict(list)
    for record in records:
        grouped[record.match_id].append(record)
    return {
        match_id: (
            np.asarray([item.true_label for item in values], dtype=np.uint8),
            np.asarray(
                [item.positive_probability for item in values], dtype=np.float64
            ),
        )
        for match_id, values in grouped.items()
    }


def _plot_reliability(
    axis: Any,
    item: MatchCalibrationMetrics,
    *,
    label: str,
    add_diagonal: bool = True,
) -> None:
    nonempty = [value for value in item.calibration_bins if value.sample_count]
    x = [value.mean_predicted_probability for value in nonempty]
    y = [value.fraction_positive for value in nonempty]
    axis.plot(x, y, marker="o", label=label)
    if add_diagonal:
        axis.plot([0, 1], [0, 1], color="black", linestyle="--", label="perfect")
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Mean predicted probability",
        ylabel="Observed positive fraction",
        title=f"Reliability | {item.match_id}",
    )


def _save_figure_atomic(figure: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    try:
        figure.savefig(temporary, dpi=150, format="png")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
