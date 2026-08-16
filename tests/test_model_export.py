from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import audio_highlight.classifier as classifier_module
from audio_highlight.classifier import (
    BASELINE_ID,
    ExportedCheerDetector,
    ModelArtifactError,
    stable_sigmoid,
)
from audio_highlight.cli import run
from audio_highlight.dataset import DatasetError, FeatureDataset
from audio_highlight.labeling import LABEL_SOURCE, SAMPLING_ALGORITHM_VERSION
from audio_highlight.model_export import (
    ModelTrainingError,
    load_final_training_data,
    train_and_export_model,
)
from audio_highlight.yamnet import YAMNET_EMBEDDING_SIZE, YAMNET_MODEL_HANDLE


def make_dataset(
    match_id: str,
    *,
    offset: float = 0.0,
    model_identifier: str = YAMNET_MODEL_HANDLE,
) -> FeatureDataset:
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.uint8)
    embeddings = np.zeros((6, YAMNET_EMBEDDING_SIZE), dtype=np.float32)
    embeddings[:, 0] = np.asarray([-3, -2, -1, 1, 2, 3], dtype=np.float32)
    embeddings[:, 1] = np.float32(offset / 10.0)
    starts = np.arange(6, dtype=np.float64) + offset
    return FeatureDataset(
        embeddings=embeddings,
        labels=labels,
        sample_ranks=np.arange(1, 7, dtype=np.int64),
        segment_indices=np.arange(6, dtype=np.int64),
        window_indices=np.zeros(6, dtype=np.int64),
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


def save_datasets(tmp_path: Path, count: int = 4) -> list[Path]:
    paths: list[Path] = []
    for index in range(count):
        path = tmp_path / f"match_{index:03d}" / "features.npz"
        make_dataset(f"match_{index:03d}", offset=index * 10.0).save(path)
        paths.append(path)
    return paths


def export_detector(tmp_path: Path):
    paths = save_datasets(tmp_path / "features")
    output = tmp_path / "model"
    result = train_and_export_model(
        paths,
        output_dir=output,
        trained_at="2026-08-17T00:00:00+00:00",
    )
    return paths, result, ExportedCheerDetector.load(output)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rewrite_model(output: Path, **replacements: object) -> None:
    model_path = output / "model.npz"
    with np.load(model_path, allow_pickle=False) as values:
        arrays = {name: np.asarray(values[name]) for name in values.files}
    arrays.update(replacements)
    np.savez_compressed(model_path, **arrays)
    metadata_path = output / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["model_sha256"] = sha256(model_path)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")


def test_all_datasets_are_concatenated_without_resampling(tmp_path: Path) -> None:
    paths = save_datasets(tmp_path)

    training = load_final_training_data(paths)

    assert training.embeddings.shape == (24, YAMNET_EMBEDDING_SIZE)
    assert training.labels.tolist() == [0, 0, 0, 1, 1, 1] * 4
    assert [item.match_id for item in training.datasets] == [
        "match_000",
        "match_001",
        "match_002",
        "match_003",
    ]


def test_incompatible_feature_identity_is_rejected(tmp_path: Path) -> None:
    paths = save_datasets(tmp_path, 2)
    incompatible = replace(make_dataset("match_001"), model_identifier="other-yamnet")
    incompatible.save(paths[1])

    with pytest.raises(ModelTrainingError, match="model_identifier"):
        load_final_training_data(paths)


def test_duplicate_match_id_is_rejected(tmp_path: Path) -> None:
    paths = save_datasets(tmp_path, 2)
    make_dataset("match_000").save(paths[1])

    with pytest.raises(ModelTrainingError, match="unique match_id"):
        load_final_training_data(paths)


def test_invalid_labels_and_nonfinite_embeddings_are_rejected() -> None:
    valid = make_dataset("match")
    with pytest.raises(DatasetError, match="binary uint8"):
        replace(valid, labels=np.asarray([0, 0, 0, 1, 1, 2], dtype=np.uint8))
    embeddings = valid.embeddings.copy()
    embeddings[0, 0] = np.nan
    with pytest.raises(DatasetError, match="finite float32"):
        replace(valid, embeddings=embeddings)


def test_incompatible_embedding_dimension_is_rejected() -> None:
    valid = make_dataset("match")

    with pytest.raises(DatasetError, match=r"shape \(N, 1024\)"):
        replace(
            valid,
            embeddings=np.zeros((6, 100), dtype=np.float32),
            embedding_dimension=100,
        )


def test_final_fit_uses_all_samples_and_frozen_classifier(tmp_path: Path) -> None:
    _, result, detector = export_detector(tmp_path)
    metadata = detector.metadata

    assert result.sample_count == 24
    assert (result.positive_count, result.negative_count) == (12, 12)
    assert result.converged is True
    assert result.iterations > 0
    assert metadata["training"]["scaler_fit_sample_count"] == 24
    assert metadata["classifier"] == {
        "preprocessing": "StandardScaler",
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 2000,
        "class_weight": None,
        "threshold": 0.5,
    }


def test_exported_npz_has_exact_numeric_array_contract(tmp_path: Path) -> None:
    _, result, _ = export_detector(tmp_path)

    with np.load(result.model_path, allow_pickle=False) as values:
        assert set(values.files) == {
            "scaler_mean",
            "scaler_scale",
            "lr_coef",
            "lr_intercept",
            "classes",
        }
        assert values["scaler_mean"].shape == (YAMNET_EMBEDDING_SIZE,)
        assert values["scaler_scale"].shape == (YAMNET_EMBEDDING_SIZE,)
        assert values["lr_coef"].shape == (YAMNET_EMBEDDING_SIZE,)
        assert values["lr_intercept"].shape == ()
        assert values["classes"].shape == (2,)
        assert values["scaler_mean"].dtype == np.float64
        assert values["scaler_scale"].dtype == np.float64
        assert values["lr_coef"].dtype == np.float64
        assert values["lr_intercept"].dtype == np.float64
        assert np.issubdtype(values["classes"].dtype, np.number)


def test_model_loader_always_disables_pickle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, result, _ = export_detector(tmp_path)
    original_load = np.load
    allow_pickle_values: list[object] = []

    def recording_load(*args: object, **kwargs: object):
        allow_pickle_values.append(kwargs.get("allow_pickle"))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(classifier_module.np, "load", recording_load)
    ExportedCheerDetector.load(result.model_path.parent)

    assert allow_pickle_values == [False]


def test_single_and_batch_manual_inference_agree(tmp_path: Path) -> None:
    paths, _, detector = export_detector(tmp_path)
    embeddings = FeatureDataset.load(paths[0]).embeddings

    batch = detector.positive_probabilities(embeddings)
    singles = np.asarray(
        [detector.positive_probability(embedding) for embedding in embeddings]
    )

    np.testing.assert_allclose(batch, singles, rtol=0, atol=0)
    assert detector.predict_embeddings(embeddings).tolist() == [0, 0, 0, 1, 1, 1]
    assert detector.predict_embedding(embeddings[0]) == 0


def test_stable_sigmoid_handles_extreme_logits() -> None:
    probabilities = stable_sigmoid(np.asarray([-1000.0, 0.0, 1000.0]))

    assert probabilities.tolist() == [0.0, 0.5, 1.0]


def test_manual_inference_is_equivalent_to_sklearn(tmp_path: Path) -> None:
    _, result, _ = export_detector(tmp_path)

    assert result.max_probability_difference <= 1e-12
    assert result.binary_predictions_equal is True


def test_threshold_point_five_is_inclusive() -> None:
    detector = ExportedCheerDetector(
        scaler_mean=np.zeros(YAMNET_EMBEDDING_SIZE, dtype=np.float64),
        scaler_scale=np.ones(YAMNET_EMBEDDING_SIZE, dtype=np.float64),
        lr_coef=np.zeros(YAMNET_EMBEDDING_SIZE, dtype=np.float64),
        lr_intercept=0.0,
        classes=np.asarray([0, 1], dtype=np.int64),
        metadata=_minimal_metadata(),
    )

    assert detector.positive_probability(
        np.zeros(YAMNET_EMBEDDING_SIZE, dtype=np.float32)
    ) == 0.5
    assert detector.predict_embedding(
        np.zeros(YAMNET_EMBEDDING_SIZE, dtype=np.float32)
    ) == 1


def test_corrupt_scale_and_wrong_input_dimension_are_rejected(tmp_path: Path) -> None:
    _, result, detector = export_detector(tmp_path)
    invalid_scale = np.ones(YAMNET_EMBEDDING_SIZE, dtype=np.float64)
    invalid_scale[10] = 0.0
    rewrite_model(result.model_path.parent, scaler_scale=invalid_scale)

    with pytest.raises(ModelArtifactError, match="greater than zero"):
        ExportedCheerDetector.load(result.model_path.parent)
    with pytest.raises(ModelArtifactError, match=r"\(1024,\)"):
        detector.positive_probability(np.zeros(10, dtype=np.float32))
    with pytest.raises(ModelArtifactError, match=r"\(N, 1024\)"):
        detector.positive_probabilities(np.zeros((2, 10), dtype=np.float32))


def test_model_hash_and_metadata_round_trip(tmp_path: Path) -> None:
    paths, result, detector = export_detector(tmp_path)
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))

    assert result.model_sha256 == sha256(result.model_path)
    assert metadata == detector.metadata
    assert metadata["model_id"] == BASELINE_ID
    assert metadata["model_sha256"] == result.model_sha256
    assert metadata["training"]["matches"] == [
        "match_000",
        "match_001",
        "match_002",
        "match_003",
    ]
    for path, record in zip(paths, metadata["training"]["datasets"], strict=True):
        assert record["sha256"] == sha256(path)
        assert not Path(record["path"]).is_absolute()


def test_reloaded_detector_contains_no_fitted_sklearn_object(tmp_path: Path) -> None:
    _, _, detector = export_detector(tmp_path)

    assert set(detector.__slots__) == {
        "classes",
        "lr_coef",
        "lr_intercept",
        "metadata",
        "scaler_mean",
        "scaler_scale",
        "threshold",
    }
    assert not hasattr(detector, "pipeline")
    assert not hasattr(detector, "estimator")


def test_tampered_model_and_metadata_are_rejected(tmp_path: Path) -> None:
    _, result, _ = export_detector(tmp_path)
    result.model_path.write_bytes(result.model_path.read_bytes() + b"tampered")
    with pytest.raises(ModelArtifactError, match="SHA-256"):
        ExportedCheerDetector.load(result.model_path.parent)

    _, second_result, _ = export_detector(tmp_path / "second")
    metadata = json.loads(second_result.metadata_path.read_text(encoding="utf-8"))
    metadata["model_id"] = "not-the-baseline"
    second_result.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ModelArtifactError, match="model_id"):
        ExportedCheerDetector.load(second_result.model_path.parent)


def test_cli_supports_explicit_paths_and_match_discovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = save_datasets(tmp_path / "explicit", 2)
    explicit_output = tmp_path / "explicit-model"
    assert run(
        [
            "train-model",
            *(str(path) for path in paths),
            "--output-dir",
            str(explicit_output),
        ]
    ) == 0
    assert "sample_count=12" in capsys.readouterr().out

    monkeypatch.chdir(tmp_path)
    canonical_paths = [
        Path("artifacts") / match_id / "features" / "features.npz"
        for match_id in ("match_a", "match_b")
    ]
    make_dataset("match_a").save(canonical_paths[0])
    make_dataset("match_b", offset=10.0).save(canonical_paths[1])
    assert run(["train-model", "--matches", "match_a", "match_b"]) == 0
    assert (
        tmp_path / "artifacts" / "models" / BASELINE_ID / "model.npz"
    ).is_file()


def test_cli_requires_exactly_one_input_mode(capsys: pytest.CaptureFixture[str]) -> None:
    assert run(["train-model"]) == 2
    assert "either explicit feature paths or --matches" in capsys.readouterr().out


def _minimal_metadata() -> dict[str, object]:
    return {
        "model_id": BASELINE_ID,
        "model_type": "logistic_regression",
        "model_sha256": "0" * 64,
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
            "matches": ["synthetic"],
            "sample_count": 2,
            "positive_count": 1,
            "negative_count": 1,
            "converged": True,
            "iterations": 1,
            "scaler_fit_sample_count": 2,
            "sklearn_version": "test",
            "numpy_version": np.__version__,
            "trained_at": "2026-08-17T00:00:00+00:00",
            "datasets": [
                {
                    "match_id": "synthetic",
                    "path": "features.npz",
                    "sha256": "0" * 64,
                    "samples": 2,
                }
            ],
        },
    }
