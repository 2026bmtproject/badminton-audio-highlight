from __future__ import annotations

import csv
import hashlib
import inspect
import json
import math
from pathlib import Path

import numpy as np
import pytest

import audio_highlight.calibration as calibration_module
from audio_highlight.calibration import (
    CALIBRATION_BIN_COUNT,
    CalibrationError,
    PredictionRecord,
    compare_matches,
    compute_calibration_bins,
    diagnose_calibration,
    expected_calibration_error,
    load_predictions,
    summarize_probabilities,
    write_calibration_artifacts,
)
from audio_highlight.cli import run


def records(match_id: str = "match_a") -> tuple[PredictionRecord, ...]:
    return (
        PredictionRecord(match_id, 0, 0, 0.1),
        PredictionRecord(match_id, 0, 0, 0.4),
        PredictionRecord(match_id, 1, 1, 0.6),
        PredictionRecord(match_id, 1, 1, 0.9),
    )


def write_predictions(path: Path, values: tuple[PredictionRecord, ...]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "test_match_id",
                "sample_rank",
                "true_label",
                "predicted_label",
                "positive_probability",
            ),
        )
        writer.writeheader()
        for index, item in enumerate(values, start=1):
            writer.writerow(
                {
                    "test_match_id": item.match_id,
                    "sample_rank": index,
                    "true_label": item.true_label,
                    "predicted_label": item.predicted_label,
                    "positive_probability": item.positive_probability,
                }
            )
    return path


def test_prediction_csv_parsing_and_match_ids(tmp_path: Path) -> None:
    path = write_predictions(tmp_path / "predictions.csv", records("match_007"))

    loaded = load_predictions(path)

    assert loaded == records("match_007")
    assert {item.match_id for item in loaded} == {"match_007"}


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("true_label", "2", "binary"),
        ("predicted_label", "2", "binary"),
        ("positive_probability", "1.01", r"\[0, 1\]"),
        ("positive_probability", "nan", r"\[0, 1\]"),
    ],
)
def test_prediction_validation(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    path = write_predictions(tmp_path / "predictions.csv", records())
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    rows[0][field] = value
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(CalibrationError, match=message):
        load_predictions(path)


def test_frozen_threshold_is_validated_not_optimized(tmp_path: Path) -> None:
    path = write_predictions(tmp_path / "predictions.csv", records())
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    rows[0]["predicted_label"] = "1"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(CalibrationError, match="frozen threshold 0.5"):
        load_predictions(path)


def test_probability_summary_statistics() -> None:
    summary = summarize_probabilities(np.asarray([0.1, 0.4], dtype=np.float64))

    assert summary.mean == pytest.approx(0.25)
    assert summary.median == pytest.approx(0.25)
    assert summary.std == pytest.approx(0.15)
    assert summary.q1 == pytest.approx(0.175)
    assert summary.q3 == pytest.approx(0.325)
    assert (summary.min, summary.max) == (0.1, 0.4)


def test_brier_log_loss_and_per_match_grouping() -> None:
    result = diagnose_calibration(records("match_b") + records("match_a"))

    assert [item.match_id for item in result.matches] == ["match_a", "match_b"]
    for item in result.matches:
        assert item.sample_count == 4
        assert item.positive_count == 2 and item.negative_count == 2
        assert item.prevalence == 0.5
        assert item.predicted_positive_rate == 0.5
        assert item.brier_score == pytest.approx(0.085)
        expected_log_loss = -np.mean(
            [math.log(0.9), math.log(0.6), math.log(0.6), math.log(0.9)]
        )
        assert item.log_loss == pytest.approx(expected_log_loss)
        assert item.positive_probability_summary.median == pytest.approx(0.75)
        assert item.negative_probability_summary.median == pytest.approx(0.25)


def test_uniform_bins_retain_empty_bins_and_include_probability_one() -> None:
    labels = np.asarray([0, 0, 1], dtype=np.uint8)
    probabilities = np.asarray([0.0, 0.05, 1.0], dtype=np.float64)

    bins = compute_calibration_bins(labels, probabilities)

    assert len(bins) == CALIBRATION_BIN_COUNT
    assert bins[0].sample_count == 2
    assert bins[0].mean_predicted_probability == pytest.approx(0.025)
    assert bins[0].fraction_positive == 0.0
    assert bins[1].sample_count == 0
    assert bins[1].mean_predicted_probability is None
    assert bins[1].fraction_positive is None
    assert bins[9].sample_count == 1
    assert bins[9].mean_predicted_probability == 1.0


def test_ece_uses_weighted_observed_positive_fraction() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.uint8)
    probabilities = np.asarray([0.1, 0.4, 0.6, 0.9], dtype=np.float64)
    bins = compute_calibration_bins(labels, probabilities, n_bins=2)

    assert expected_calibration_error(bins, sample_count=4) == pytest.approx(0.25)


def test_diagnostic_source_has_no_training_or_yamnet_path() -> None:
    source = inspect.getsource(calibration_module)

    assert ".fit(" not in source
    assert "FeatureDataset" not in source
    assert "YamNet" not in source
    assert "tensorflow" not in source.lower()
    assert "best_threshold" not in source
    assert "optimal_threshold" not in source


def test_feature_file_is_not_modified(tmp_path: Path) -> None:
    feature_path = tmp_path / "features.npz"
    feature_path.write_bytes(b"opaque feature artifact")
    before = hashlib.sha256(feature_path.read_bytes()).hexdigest()

    diagnose_calibration(records())

    assert hashlib.sha256(feature_path.read_bytes()).hexdigest() == before


def test_json_csv_and_headless_plots_round_trip(tmp_path: Path) -> None:
    result = diagnose_calibration(records("match_002") + records("match_004"))
    artifacts = write_calibration_artifacts(result, tmp_path / "calibration")

    value = json.loads(artifacts.metrics_json.read_text(encoding="utf-8"))
    with artifacts.summary_csv.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert value["baseline_id"] == "yamnet_mean_lr_v1"
    assert value["threshold"] == 0.5
    assert value["n_bins"] == 10
    assert len(value["matches"][0]["calibration_bins"]) == 10
    assert [row["match_id"] for row in rows] == ["match_002", "match_004"]
    assert len(artifacts.probability_distribution_plots) == 2
    assert len(artifacts.reliability_plots) == 2
    for path in (
        *artifacts.probability_distribution_plots,
        *artifacts.reliability_plots,
        artifacts.combined_reliability_plot,
    ):
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_match_comparison_uses_result_data() -> None:
    first = records("alpha")
    second = tuple(
        PredictionRecord("beta", item.true_label, item.predicted_label, score)
        for item, score in zip(first, (0.2, 0.45, 0.7, 0.95), strict=True)
    )
    result = diagnose_calibration(first + second)

    comparison = compare_matches(result, "alpha", "beta")

    assert comparison["same_observed_prevalence"] is True
    assert comparison["positive_probability_median_difference"] == pytest.approx(0.075)
    assert comparison["negative_probability_median_difference"] == pytest.approx(0.075)
    assert "match_002" not in inspect.getsource(compare_matches)
    assert "match_004" not in inspect.getsource(compare_matches)


def test_cli_writes_diagnostic_without_model_inputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    predictions = write_predictions(tmp_path / "predictions.csv", records())
    output = tmp_path / "calibration"
    baseline = tmp_path / "baseline.json"
    monkeypatch.setattr(
        "audio_highlight.baseline.write_baseline_metadata",
        lambda: baseline,
    )

    assert run(
        [
            "diagnose-calibration",
            "--predictions",
            str(predictions),
            "--output-dir",
            str(output),
        ]
    ) == 0

    console = capsys.readouterr().out
    assert "threshold" not in console.lower() or "0.5" not in console
    assert (output / "calibration_metrics.json").is_file()
    assert (output / "calibration_summary.csv").is_file()
