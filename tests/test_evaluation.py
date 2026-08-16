from __future__ import annotations

import csv
import inspect
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import audio_highlight.dataset as dataset_module
from audio_highlight.cli import run
from audio_highlight.evaluation import (
    BinaryClassificationMetrics,
    EvaluationError,
    compute_binary_metrics,
    evaluate_cross_match,
    macro_mean_metrics,
    threshold_probabilities,
    write_evaluation_artifacts,
)
from audio_highlight.dataset import DatasetError, FeatureDataset


def make_dataset(
    match_id: str,
    *,
    offset: float = 0.0,
    model_identifier: str = "test-yamnet",
) -> FeatureDataset:
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.uint8)
    embeddings = np.zeros((6, 1024), dtype=np.float32)
    embeddings[:, 0] = np.asarray([-3, -2, -1, 1, 2, 3], dtype=np.float32)
    embeddings[:, 1] = offset
    starts = np.arange(6, dtype=np.float64) + offset
    return FeatureDataset(
        embeddings=embeddings,
        labels=labels,
        sample_ranks=np.arange(10, 16, dtype=np.int64),
        segment_indices=np.arange(100, 106, dtype=np.int64),
        window_indices=np.arange(6, dtype=np.int64),
        start_secs=starts,
        end_secs=starts + 3.0,
        match_id=match_id,
        segments_sha256="a" * 64,
        embedding_dimension=1024,
        sample_rate_hz=16_000,
        model_identifier=model_identifier,
        window_sec=3.0,
        hop_sec=1.0,
        post_padding_sec=3.0,
        label_source="current_segments_blind_human",
        sampling_seed=42,
        sampling_algorithm_version=1,
    )


def save_matches(tmp_path: Path, count: int = 2) -> list[Path]:
    paths: list[Path] = []
    for index in range(count):
        path = tmp_path / f"match_{index}.npz"
        make_dataset(f"match_{index}", offset=index * 10.0).save(path)
        paths.append(path)
    return paths


def test_npz_loading_uses_allow_pickle_false(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = save_matches(tmp_path, 1)[0]
    original_load = np.load
    calls: list[bool | None] = []

    def recording_load(*args: object, **kwargs: object):
        calls.append(kwargs.get("allow_pickle"))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(dataset_module.np, "load", recording_load)
    loaded = FeatureDataset.load(path)

    assert loaded.match_id == "match_0"
    assert loaded.embeddings.shape == (6, 1024)
    assert calls == [False]


def test_invalid_embedding_dimension_is_rejected() -> None:
    valid = make_dataset("match")

    with pytest.raises(DatasetError, match=r"shape \(N, 1024\)"):
        replace(
            valid,
            embeddings=np.zeros((6, 100), dtype=np.float32),
            embedding_dimension=100,
        )


@pytest.mark.parametrize(
    "labels",
    [
        np.asarray([0, 0, 0, 1, 1, 2], dtype=np.uint8),
        np.asarray([0, 0, 0, 1, 1, 0.5], dtype=np.float32),
    ],
)
def test_invalid_binary_labels_are_rejected(labels: np.ndarray) -> None:
    valid = make_dataset("match")

    with pytest.raises(DatasetError, match="binary uint8"):
        replace(valid, labels=labels)


def test_metadata_length_mismatch_is_rejected() -> None:
    valid = make_dataset("match")

    with pytest.raises(DatasetError, match="feature metadata must align"):
        replace(valid, sample_ranks=valid.sample_ranks[:-1])


def test_incompatible_model_identifiers_are_rejected(tmp_path: Path) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    make_dataset("first", model_identifier="yamnet-a").save(first)
    make_dataset("second", model_identifier="yamnet-b").save(second)

    with pytest.raises(EvaluationError, match="model_identifier"):
        evaluate_cross_match([first, second])


def test_two_matches_produce_two_strict_leave_one_match_out_folds(
    tmp_path: Path,
) -> None:
    result = evaluate_cross_match(save_matches(tmp_path))

    assert len(result.folds) == 2
    assert [fold.test_match for fold in result.folds] == ["match_0", "match_1"]
    assert all(fold.test_match not in fold.train_matches for fold in result.folds)
    assert all(len(fold.train_matches) == 1 for fold in result.folds)
    assert all(fold.train_samples == 6 and fold.test_samples == 6 for fold in result.folds)


def test_n_matches_produce_n_folds_and_each_match_is_test_once(tmp_path: Path) -> None:
    result = evaluate_cross_match(save_matches(tmp_path, count=3))

    assert len(result.folds) == 3
    assert sorted(fold.test_match for fold in result.folds) == [
        "match_0",
        "match_1",
        "match_2",
    ]
    assert all(len(fold.train_matches) == 2 for fold in result.folds)
    assert all(fold.test_match not in fold.train_matches for fold in result.folds)


def test_evaluator_has_no_random_window_split_path() -> None:
    source = inspect.getsource(evaluate_cross_match)

    assert "train_test_split" not in source
    assert "held_out_index" in source


def test_predictions_preserve_provenance_and_probability_range(tmp_path: Path) -> None:
    result = evaluate_cross_match(save_matches(tmp_path))
    first = result.predictions[0]
    last_first_match = result.predictions[5]

    assert first.test_match_id == "match_0"
    assert first.sample_rank == 10
    assert first.segment_index == 100
    assert first.window_index_in_segment == 0
    assert (first.start_sec, first.end_sec) == (0.0, 3.0)
    assert last_first_match.segment_index == 105
    assert all(0.0 <= item.positive_probability <= 1.0 for item in result.predictions)


def test_fixed_threshold_includes_probability_equal_to_point_five() -> None:
    probabilities = np.asarray([0.49, 0.5, 0.51], dtype=np.float64)

    assert threshold_probabilities(probabilities).tolist() == [0, 1, 1]


def test_confusion_matrix_order_and_per_fold_metrics() -> None:
    true = np.asarray([0, 0, 1, 1], dtype=np.uint8)
    predicted = np.asarray([0, 1, 0, 1], dtype=np.uint8)
    probabilities = np.asarray([0.1, 0.8, 0.4, 0.9], dtype=np.float64)

    metrics = compute_binary_metrics(true, predicted, probabilities)

    assert (metrics.tn, metrics.fp, metrics.fn, metrics.tp) == (1, 1, 1, 1)
    assert metrics.sample_count == 4
    assert metrics.accuracy == pytest.approx(0.5)
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.f1 == pytest.approx(0.5)
    assert metrics.roc_auc == pytest.approx(0.75)
    assert metrics.average_precision == pytest.approx(5 / 6)


def test_macro_mean_weights_matches_equally_not_by_sample_count() -> None:
    small = BinaryClassificationMetrics(2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0, 1, 1, 0)
    large = BinaryClassificationMetrics(
        2_000, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 1_000, 0, 0, 1_000
    )

    macro = macro_mean_metrics([small, large])

    assert macro == {
        "accuracy": 0.5,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "roc_auc": 0.5,
        "average_precision": 0.5,
    }


def test_prediction_csv_and_metrics_json_round_trip(tmp_path: Path) -> None:
    result = evaluate_cross_match(save_matches(tmp_path / "features"))
    artifacts = write_evaluation_artifacts(result, tmp_path / "evaluation")

    with artifacts.predictions_csv.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    summary = json.loads(artifacts.metrics_json.read_text(encoding="utf-8"))

    assert len(rows) == 12
    assert rows[0]["test_match_id"] == "match_0"
    assert rows[0]["sample_rank"] == "10"
    assert rows[0]["segment_index"] == "100"
    assert 0.0 <= float(rows[0]["positive_probability"]) <= 1.0
    assert summary["model"] == {
        "baseline_id": "yamnet_mean_lr_v1",
        "type": "logistic_regression",
        "C": 1.0,
        "max_iter": 2000,
        "solver": "lbfgs",
        "class_weight": None,
        "threshold": 0.5,
        "preprocessing": "training-fold StandardScaler",
    }
    assert len(summary["folds"]) == 2
    assert summary["folds"][0]["metrics"]["confusion_matrix"] == [[3, 0], [0, 3]]
    assert summary["macro_mean"] == result.macro_mean


def test_cli_evaluate_writes_artifacts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    paths = save_matches(tmp_path / "features")
    output = tmp_path / "evaluation"

    assert run(["evaluate", *(str(path) for path in paths), "--output-dir", str(output)]) == 0

    console = capsys.readouterr().out
    assert "Fold: match_1 -> match_0" in console
    assert "Fold: match_0 -> match_1" in console
    assert "Macro mean" in console
    assert (output / "cross_match_predictions.csv").is_file()
    assert (output / "cross_match_metrics.json").is_file()
