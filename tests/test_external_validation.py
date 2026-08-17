from __future__ import annotations

import csv
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import audio_highlight.evaluation as evaluation_module
import audio_highlight.external_validation as external_module
from audio_highlight.classifier import (
    BASELINE_ID,
    ExportedCheerDetector,
    ModelArtifactError,
)
from audio_highlight.cli import run
from audio_highlight.dataset import DatasetError, FeatureDataset
from audio_highlight.external_validation import (
    ExternalValidationError,
    evaluate_external_match,
    write_external_validation_artifacts,
)
from audio_highlight.labeling import LABEL_SOURCE, SAMPLING_ALGORITHM_VERSION
from audio_highlight.yamnet import YAMNET_EMBEDDING_SIZE, YAMNET_MODEL_HANDLE


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_dataset(match_id: str, *, model_identifier: str = YAMNET_MODEL_HANDLE):
    labels = np.asarray([0, 0, 1, 1, 1, 0], dtype=np.uint8)
    embeddings = np.zeros((6, YAMNET_EMBEDDING_SIZE), dtype=np.float32)
    embeddings[:, 0] = np.asarray([-2, -1, 0, 1, 2, 3], dtype=np.float32)
    starts = np.arange(6, dtype=np.float64) * 10.0
    return FeatureDataset(
        embeddings=embeddings,
        labels=labels,
        sample_ranks=np.arange(1, 7, dtype=np.int64),
        segment_indices=np.arange(10, 16, dtype=np.int64),
        window_indices=np.arange(6, dtype=np.int64),
        start_secs=starts,
        end_secs=starts + 3.0,
        match_id=match_id,
        segments_sha256="a" * 64,
        embedding_dimension=YAMNET_EMBEDDING_SIZE,
        sample_rate_hz=16_000,
        model_identifier=model_identifier,
        window_sec=3.0,
        hop_sec=1.0,
        post_padding_sec=3.0,
        label_source=LABEL_SOURCE,
        sampling_seed=42,
        sampling_algorithm_version=SAMPLING_ALGORITHM_VERSION,
    )


def write_model(
    directory: Path,
    *,
    training_matches: tuple[str, ...] = ("train_a", "train_b"),
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / "model.npz"
    coefficient = np.zeros(YAMNET_EMBEDDING_SIZE, dtype=np.float64)
    coefficient[0] = 1.0
    np.savez_compressed(
        model_path,
        scaler_mean=np.zeros(YAMNET_EMBEDDING_SIZE, dtype=np.float64),
        scaler_scale=np.ones(YAMNET_EMBEDDING_SIZE, dtype=np.float64),
        lr_coef=coefficient,
        lr_intercept=np.asarray(0.0, dtype=np.float64),
        classes=np.asarray([0, 1], dtype=np.int64),
    )
    per_dataset = 2
    total = per_dataset * len(training_matches)
    metadata = {
        "model_id": BASELINE_ID,
        "model_type": "logistic_regression",
        "model_sha256": sha256(model_path),
        "feature_extractor": {
            "model_identifier": YAMNET_MODEL_HANDLE,
            "pooling": "mean",
            "embedding_dimension": YAMNET_EMBEDDING_SIZE,
        },
        "audio": {
            "sample_rate_hz": 16_000,
            "window_sec": 3.0,
            "hop_sec": 1.0,
            "post_padding_sec": 3.0,
        },
        "classifier": {
            "preprocessing": "StandardScaler",
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 2000,
            "class_weight": None,
            "threshold": 0.5,
        },
        "training": {
            "matches": list(training_matches),
            "sample_count": total,
            "positive_count": len(training_matches),
            "negative_count": len(training_matches),
            "converged": True,
            "iterations": 1,
            "scaler_fit_sample_count": total,
            "sklearn_version": "test",
            "numpy_version": np.__version__,
            "trained_at": "2026-08-17T00:00:00+00:00",
            "datasets": [
                {
                    "match_id": match_id,
                    "path": f"artifacts/{match_id}/features/features.npz",
                    "sha256": str(index + 1) * 64,
                    "samples": per_dataset,
                }
                for index, match_id in enumerate(training_matches)
            ],
        },
    }
    (directory / "metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    return directory


def make_artifacts(tmp_path: Path, *, match_id: str = "match_005"):
    feature_path = tmp_path / "features.npz"
    make_dataset(match_id).save(feature_path)
    model_dir = write_model(tmp_path / "model")
    return feature_path, model_dir


def rewrite_npz(path: Path, **replacements: object) -> None:
    with np.load(path, allow_pickle=False) as values:
        arrays = {name: np.asarray(values[name]) for name in values.files}
    arrays.update(replacements)
    np.savez_compressed(path, **arrays)


def test_external_evaluator_loads_exported_model_and_uses_frozen_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    feature_path, model_dir = make_artifacts(tmp_path)
    original = ExportedCheerDetector.load.__func__
    calls: list[Path] = []

    def recording_load(cls, directory):
        calls.append(Path(directory))
        return original(cls, directory)

    monkeypatch.setattr(ExportedCheerDetector, "load", classmethod(recording_load))
    result = evaluate_external_match(feature_path, model_dir)

    assert calls == [model_dir]
    assert result.threshold == 0.5
    assert result.training_matches == ("train_a", "train_b")


def test_evaluator_never_calls_classifier_or_scaler_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    feature_path, model_dir = make_artifacts(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("fit path must not be called")

    monkeypatch.setattr(evaluation_module, "build_baseline_classifier", forbidden)
    monkeypatch.setattr(StandardScaler, "fit", forbidden)

    assert evaluate_external_match(feature_path, model_dir).sample_count == 6


def test_external_module_does_not_import_training_implementation() -> None:
    source = inspect.getsource(external_module)

    assert "model_export" not in source
    assert "train_and_export_model" not in source
    assert ".fit(" not in source
    assert "partial_fit" not in source


def test_test_match_in_training_provenance_fails_before_metrics(tmp_path: Path) -> None:
    feature_path = tmp_path / "features.npz"
    make_dataset("match_005").save(feature_path)
    model_dir = write_model(
        tmp_path / "model", training_matches=("train_a", "match_005")
    )

    with pytest.raises(ExternalValidationError, match="appears in model training"):
        evaluate_external_match(feature_path, model_dir)


def test_incompatible_yamnet_identifier_is_rejected(tmp_path: Path) -> None:
    feature_path = tmp_path / "features.npz"
    make_dataset("future_match", model_identifier="other-yamnet").save(feature_path)
    model_dir = write_model(tmp_path / "model")

    with pytest.raises(ExternalValidationError, match="model_identifier"):
        evaluate_external_match(feature_path, model_dir)


def test_wrong_embedding_dimension_is_rejected(tmp_path: Path) -> None:
    feature_path, model_dir = make_artifacts(tmp_path)
    rewrite_npz(
        feature_path,
        embeddings=np.zeros((6, 100), dtype=np.float32),
        embedding_dimension=np.asarray(100, dtype=np.int64),
    )

    with pytest.raises(DatasetError, match=r"shape \(N, 1024\)"):
        evaluate_external_match(feature_path, model_dir)


def test_invalid_labels_and_nonfinite_embeddings_are_rejected(tmp_path: Path) -> None:
    feature_path, model_dir = make_artifacts(tmp_path / "labels")
    rewrite_npz(
        feature_path,
        labels=np.asarray([0, 0, 1, 1, 2, 0], dtype=np.uint8),
    )
    with pytest.raises(DatasetError, match="binary uint8"):
        evaluate_external_match(feature_path, model_dir)

    feature_path, model_dir = make_artifacts(tmp_path / "embeddings")
    embeddings = make_dataset("match_005").embeddings.copy()
    embeddings[0, 0] = np.nan
    rewrite_npz(feature_path, embeddings=embeddings)
    with pytest.raises(DatasetError, match="finite float32"):
        evaluate_external_match(feature_path, model_dir)


def test_predictions_and_binary_metrics_are_correct(tmp_path: Path) -> None:
    feature_path, model_dir = make_artifacts(tmp_path)
    result = evaluate_external_match(feature_path, model_dir)
    expected_probabilities = 1.0 / (
        1.0 + np.exp(-np.asarray([-2, -1, 0, 1, 2, 3], dtype=np.float64))
    )

    assert len(result.predictions) == 6
    np.testing.assert_allclose(
        [item.positive_probability for item in result.predictions],
        expected_probabilities,
    )
    assert all(0.0 <= item.positive_probability <= 1.0 for item in result.predictions)
    assert [item.predicted_label for item in result.predictions] == [0, 0, 1, 1, 1, 1]
    metrics = result.metrics
    assert metrics.accuracy == pytest.approx(5 / 6)
    assert metrics.precision == pytest.approx(3 / 4)
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.f1 == pytest.approx(6 / 7)
    assert (metrics.tn, metrics.fp, metrics.fn, metrics.tp) == (2, 1, 0, 3)
    assert metrics.roc_auc == pytest.approx(
        roc_auc_score(make_dataset("x").labels, expected_probabilities)
    )
    assert metrics.average_precision == pytest.approx(
        average_precision_score(make_dataset("x").labels, expected_probabilities)
    )


def test_calibration_sensitive_and_descriptive_metrics_are_correct(
    tmp_path: Path,
) -> None:
    feature_path, model_dir = make_artifacts(tmp_path)
    result = evaluate_external_match(feature_path, model_dir)
    labels = make_dataset("x").labels.astype(np.float64)
    probabilities = np.asarray(
        [item.positive_probability for item in result.predictions], dtype=np.float64
    )

    assert result.brier_score == pytest.approx(np.mean((labels - probabilities) ** 2))
    expected_log_loss = -np.mean(
        labels * np.log(probabilities) + (1 - labels) * np.log(1 - probabilities)
    )
    assert result.log_loss == pytest.approx(expected_log_loss)
    assert result.observed_prevalence == pytest.approx(0.5)
    assert result.predicted_positive_rate == pytest.approx(4 / 6)
    assert result.positive_probability_summary.median == pytest.approx(
        np.median(probabilities[labels == 1])
    )
    assert result.negative_probability_summary.median == pytest.approx(
        np.median(probabilities[labels == 0])
    )
    assert 0.0 <= result.ece <= 1.0


def test_prediction_csv_and_metrics_json_round_trip(tmp_path: Path) -> None:
    feature_path, model_dir = make_artifacts(tmp_path)
    result = evaluate_external_match(feature_path, model_dir)
    artifacts = write_external_validation_artifacts(result, tmp_path / "external")

    with artifacts.predictions_csv.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    metrics = json.loads(artifacts.metrics_json.read_text(encoding="utf-8"))

    assert len(rows) == 6
    assert rows[0] == {
        "match_id": "match_005",
        "sample_rank": "1",
        "segment_index": "10",
        "window_index_in_segment": "0",
        "start_sec": "0.0",
        "end_sec": "3.0",
        "true_label": "0",
        "predicted_label": "0",
        "positive_probability": str(result.predictions[0].positive_probability),
    }
    assert metrics["validation_type"] == "untouched_external_match"
    assert metrics["model"]["threshold"] == 0.5
    assert metrics["metrics"]["confusion_matrix"] == [[2, 1], [0, 3]]
    assert metrics["test"]["match_id"] == "match_005"


def test_model_and_feature_hashes_are_recorded_and_inputs_stay_identical(
    tmp_path: Path,
) -> None:
    feature_path, model_dir = make_artifacts(tmp_path)
    model_path = model_dir / "model.npz"
    metadata_path = model_dir / "metadata.json"
    before = (sha256(feature_path), sha256(model_path), sha256(metadata_path))

    result = evaluate_external_match(feature_path, model_dir)
    write_external_validation_artifacts(result, tmp_path / "external")

    assert result.feature_sha256 == before[0]
    assert result.model_sha256 == before[1]
    assert (
        sha256(feature_path),
        sha256(model_path),
        sha256(metadata_path),
    ) == before


def test_external_output_cannot_use_cross_match_tree(tmp_path: Path) -> None:
    feature_path, model_dir = make_artifacts(tmp_path)
    result = evaluate_external_match(feature_path, model_dir)

    with pytest.raises(ExternalValidationError, match="must not use"):
        write_external_validation_artifacts(
            result, tmp_path / "artifacts" / "cross_match" / "evaluation"
        )


def test_generic_future_match_id_and_requested_identity(tmp_path: Path) -> None:
    feature_path, model_dir = make_artifacts(tmp_path, match_id="match_007")

    result = evaluate_external_match(
        feature_path, model_dir, expected_match_id="match_007"
    )

    assert result.match_id == "match_007"
    assert all(item.match_id == "match_007" for item in result.predictions)
    with pytest.raises(ExternalValidationError, match="does not match"):
        evaluate_external_match(
            feature_path, model_dir, expected_match_id="match_006"
        )


def test_corrupt_model_metadata_is_rejected(tmp_path: Path) -> None:
    feature_path, model_dir = make_artifacts(tmp_path)
    metadata_path = model_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["model_id"] = "wrong-model"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ModelArtifactError, match="model_id"):
        evaluate_external_match(feature_path, model_dir)


def test_cli_uses_canonical_paths_and_separate_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    feature_path = Path("artifacts/future_match/features/features.npz")
    make_dataset("future_match").save(feature_path)
    write_model(Path("artifacts/models/yamnet_mean_lr_v1"))

    assert run(["evaluate-external", "--match-id", "future_match"]) == 0

    output = Path("artifacts/future_match/external_validation")
    assert (output / "predictions.csv").is_file()
    assert (output / "metrics.json").is_file()
    assert not Path("artifacts/cross_match").exists()
    console = capsys.readouterr().out
    assert "validation_type=untouched_external_match" in console
    assert "threshold=0.5" in console
