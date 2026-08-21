from __future__ import annotations

import csv
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

import audio_highlight.classifier as classifier_module
from audio_highlight.dataset import FeatureDataset
from audio_highlight.labeling import LABEL_SOURCE, SAMPLING_ALGORITHM_VERSION
from audio_highlight.yamnet import YAMNET_CLASS_COUNT, YAMNET_MODEL_HANDLE, YamNetError
from audio_highlight.zero_shot import (
    METHODS,
    AudioSetClass,
    NativeClassScores,
    ZeroShotComparisonError,
    aggregate_mean_class_scores,
    compare_zero_shot_baselines,
    load_supervised_references,
    resolve_audioset_classes,
    waveform_log_rms_db,
    write_baseline_comparison_artifacts,
)


def write_class_map(path: Path, names: tuple[str, ...] = ("Cheering", "Applause", "Crowd")):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("index", "mid", "display_name"))
        writer.writeheader()
        for index, name in zip((17, 31, 204), names, strict=True):
            writer.writerow({"index": index, "mid": f"/m/{index}", "display_name": name})
    return path


class FakeScoreProvider:
    model_identifier = YAMNET_MODEL_HANDLE

    def __init__(self, class_map_path: Path) -> None:
        self.class_map_path = class_map_path
        self.classes = resolve_audioset_classes(class_map_path)
        self.call_count = 0

    def score(self, audio) -> NativeClassScores:
        self.call_count += 1
        level = float(np.mean(np.abs(audio.samples)))
        return NativeClassScores(level, level / 2.0, level / 4.0)


def make_dataset(match_id: str) -> FeatureDataset:
    count = 4
    starts = np.arange(count, dtype=np.float64) * 3.0
    return FeatureDataset(
        embeddings=np.zeros((count, 1024), dtype=np.float32),
        labels=np.asarray([0, 0, 1, 1], dtype=np.uint8),
        sample_ranks=np.arange(1, count + 1, dtype=np.int64),
        segment_indices=np.arange(10, 10 + count, dtype=np.int64),
        window_indices=np.arange(count, dtype=np.int64),
        start_secs=starts,
        end_secs=starts + 3.0,
        match_id=match_id,
        segments_sha256="a" * 64,
        embedding_dimension=1024,
        sample_rate_hz=16_000,
        model_identifier=YAMNET_MODEL_HANDLE,
        window_sec=3.0,
        hop_sec=1.0,
        post_padding_sec=3.0,
        label_source=LABEL_SOURCE,
        sampling_seed=42,
        sampling_algorithm_version=SAMPLING_ALGORITHM_VERSION,
    )


def reference_rows(match_id: str, probabilities: list[float]) -> list[dict[str, object]]:
    dataset = make_dataset(match_id)
    rows = []
    for index in range(dataset.labels.size):
        rows.append(
            {
                "match_id": match_id,
                "sample_rank": int(dataset.sample_ranks[index]),
                "segment_index": int(dataset.segment_indices[index]),
                "window_index_in_segment": int(dataset.window_indices[index]),
                "start_sec": float(dataset.start_secs[index]),
                "end_sec": float(dataset.end_secs[index]),
                "true_label": int(dataset.labels[index]),
                "positive_probability": probabilities[index],
            }
        )
    return rows


def write_references(path: Path, rows: list[dict[str, object]], *, match_field: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        match_field,
        "sample_rank",
        "segment_index",
        "window_index_in_segment",
        "start_sec",
        "end_sec",
        "true_label",
        "positive_probability",
    )
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in reversed(rows):
            value = dict(row)
            value[match_field] = value.pop("match_id")
            writer.writerow(value)
    return path


def make_comparison_inputs(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    amplitudes = (0.01, 0.03, 0.3, 0.6)
    matches = ("dev_a", "dev_b", "external_c")
    for match_id in matches:
        root = artifacts / match_id
        make_dataset(match_id).save(root / "features" / "features.npz")
        labeling = root / "labeling"
        labeling.mkdir(parents=True, exist_ok=True)
        (labeling / "labels.csv").write_text("synthetic-labels\n", encoding="utf-8")
        (labeling / "sample_manifest.json").write_text("{}\n", encoding="utf-8")
        audio = root / "audio" / "audio.f32le"
        audio.parent.mkdir(parents=True, exist_ok=True)
        samples = np.concatenate(
            [np.full(48_000, amplitude, dtype="<f4") for amplitude in amplitudes]
        )
        samples.tofile(audio)
    dev_rows = reference_rows("dev_a", [0.1, 0.2, 0.8, 0.9]) + reference_rows(
        "dev_b", [0.15, 0.25, 0.75, 0.85]
    )
    external_rows = reference_rows("external_c", [0.9, 0.8, 0.2, 0.1])
    dev_predictions = write_references(
        tmp_path / "development.csv", dev_rows, match_field="test_match_id"
    )
    external_predictions = write_references(
        tmp_path / "external.csv", external_rows, match_field="match_id"
    )
    model = tmp_path / "model.npz"
    model.write_bytes(b"frozen-model")
    model.with_name("metadata.json").write_text(
        json.dumps(
            {
                "model_id": "yamnet_mean_lr_v1",
                "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                "training": {"matches": ["dev_a", "dev_b"]},
            }
        ),
        encoding="utf-8",
    )
    class_map = write_class_map(tmp_path / "class_map.csv")
    provider = FakeScoreProvider(class_map)
    return artifacts, dev_predictions, external_predictions, model, provider


def run_comparison(tmp_path: Path):
    artifacts, dev_predictions, external_predictions, model, provider = (
        make_comparison_inputs(tmp_path)
    )
    result = compare_zero_shot_baselines(
        ("dev_a", "dev_b"),
        "external_c",
        artifact_root=artifacts,
        development_predictions_path=dev_predictions,
        external_predictions_path=external_predictions,
        model_path=model,
        score_provider=provider,
    )
    return result, provider, (dev_predictions, external_predictions, model)


def test_exact_audioset_classes_are_resolved_from_csv_not_fixed_indices(
    tmp_path: Path,
) -> None:
    path = write_class_map(tmp_path / "class_map.csv")

    classes = resolve_audioset_classes(path)

    assert classes == (
        AudioSetClass("Cheering", 17),
        AudioSetClass("Applause", 31),
        AudioSetClass("Crowd", 204),
    )
    source = inspect.getsource(resolve_audioset_classes)
    assert "Cheering" not in source
    assert "Applause" not in source
    assert "Crowd" not in source


@pytest.mark.parametrize("missing_name", ["Cheering", "Applause", "Crowd"])
def test_missing_required_exact_class_fails(tmp_path: Path, missing_name: str) -> None:
    names = tuple(name for name in ("Cheering", "Applause", "Crowd") if name != missing_name)
    path = tmp_path / "class_map.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("index", "display_name"))
        writer.writeheader()
        for index, name in enumerate(names):
            writer.writerow({"index": index, "display_name": name})

    with pytest.raises(ZeroShotComparisonError, match=missing_name):
        resolve_audioset_classes(path)


def test_patch_shape_mean_aggregation_and_crowd_max_are_fixed() -> None:
    scores = np.zeros((2, YAMNET_CLASS_COUNT), dtype=np.float32)
    scores[:, 17] = [0.2, 0.6]
    scores[:, 31] = [0.7, 0.9]
    scores[:, 204] = [0.1, 0.3]
    classes = (
        AudioSetClass("Cheering", 17),
        AudioSetClass("Applause", 31),
        AudioSetClass("Crowd", 204),
    )

    result = aggregate_mean_class_scores(scores, classes)

    assert result.cheering == pytest.approx(0.4)
    assert result.applause == pytest.approx(0.8)
    assert result.crowd == pytest.approx(0.2)
    assert result.crowd_combo == pytest.approx(0.8)
    with pytest.raises(YamNetError, match="rank 2"):
        aggregate_mean_class_scores(np.zeros(YAMNET_CLASS_COUNT), classes)


def test_rms_and_silent_log_rms_are_correct_and_finite() -> None:
    waveform = np.asarray([3.0, 4.0], dtype=np.float32)
    expected_rms = np.sqrt((9.0 + 16.0) / 2.0)

    assert waveform_log_rms_db(waveform) == pytest.approx(
        20 * np.log10(expected_rms + 1e-12)
    )
    silence = waveform_log_rms_db(np.zeros(48_000, dtype=np.float32))
    assert silence == pytest.approx(-240.0)
    assert np.isfinite(silence)


def test_reference_loader_rejects_invalid_labels(tmp_path: Path) -> None:
    rows = reference_rows("dev", [0.1, 0.2, 0.8, 0.9])
    rows[0]["true_label"] = 2
    path = write_references(tmp_path / "predictions.csv", rows, match_field="test_match_id")

    with pytest.raises(ZeroShotComparisonError, match="labels must be binary"):
        load_supervised_references(path, match_field="test_match_id")


def test_identity_join_does_not_depend_on_reference_row_position(tmp_path: Path) -> None:
    result, provider, _ = run_comparison(tmp_path)

    assert len(result.predictions) == 12
    assert provider.call_count == 12
    first = result.predictions[0]
    assert (first.match_id, first.sample_rank, first.supervised_lr_probability) == (
        "dev_a",
        1,
        0.1,
    )


def test_mismatched_sample_identity_fails(tmp_path: Path) -> None:
    artifacts, dev_predictions, external_predictions, model, provider = (
        make_comparison_inputs(tmp_path)
    )
    rows = list(csv.DictReader(dev_predictions.open(encoding="utf-8", newline="")))
    rows[0]["sample_rank"] = "999"
    with dev_predictions.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ZeroShotComparisonError, match="identity or label mismatch"):
        compare_zero_shot_baselines(
            ("dev_a", "dev_b"),
            "external_c",
            artifact_root=artifacts,
            development_predictions_path=dev_predictions,
            external_predictions_path=external_predictions,
            model_path=model,
            score_provider=provider,
        )


def test_frozen_reference_origins_and_external_exclusion_from_macro(tmp_path: Path) -> None:
    result, _, _ = run_comparison(tmp_path)
    development = result.metrics["development"]
    external = result.metrics["external"]["external_c"]

    assert result.metadata["supervised_reference"] == {
        "development": "existing_leave_one_match_out_oof_predictions",
        "external": "existing_frozen_final_detector_predictions",
    }
    assert development["dev_a"]["methods"]["embedding_lr"]["roc_auc"] == 1.0
    assert development["dev_b"]["methods"]["embedding_lr"]["roc_auc"] == 1.0
    assert development["macro_mean"]["embedding_lr"]["roc_auc"] == 1.0
    assert external["methods"]["embedding_lr"]["roc_auc"] == 0.0


def test_comparison_never_fits_classifier_or_scaler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("training is forbidden")

    monkeypatch.setattr(classifier_module, "build_baseline_classifier", forbidden)
    monkeypatch.setattr(StandardScaler, "fit", forbidden)

    assert run_comparison(tmp_path)[0].experiment_id == "zero_shot_v1"


def test_auc_ap_and_development_macro_are_correct(tmp_path: Path) -> None:
    result, _, _ = run_comparison(tmp_path)
    development = result.metrics["development"]

    for match_id in ("dev_a", "dev_b"):
        for method in ("rms", "yamnet_cheering", "yamnet_crowd_combo"):
            assert development[match_id]["methods"][method] == {
                "roc_auc": 1.0,
                "average_precision": 1.0,
            }
    assert set(development["macro_mean"]) == set(METHODS)
    assert "external_c" not in development


def test_serialization_is_deterministic_and_round_trips(tmp_path: Path) -> None:
    result, _, _ = run_comparison(tmp_path)
    first = write_baseline_comparison_artifacts(result, tmp_path / "output-a")
    second = write_baseline_comparison_artifacts(result, tmp_path / "output-b")

    for first_path, second_path in zip(
        (first.predictions_csv, first.metrics_json, first.summary_csv, first.metadata_json),
        (second.predictions_csv, second.metrics_json, second.summary_csv, second.metadata_json),
        strict=True,
    ):
        assert first_path.read_bytes() == second_path.read_bytes()
    with first.predictions_csv.open(encoding="utf-8", newline="") as file:
        assert len(list(csv.DictReader(file))) == 12
    metrics = json.loads(first.metrics_json.read_text(encoding="utf-8"))
    assert metrics == result.metrics


def test_input_artifacts_remain_byte_identical(tmp_path: Path) -> None:
    artifacts, dev_predictions, external_predictions, model, provider = (
        make_comparison_inputs(tmp_path)
    )
    inputs = [dev_predictions, external_predictions, model, model.with_name("metadata.json")]
    inputs.extend(artifacts.glob("**/*"))
    files = [path for path in inputs if path.is_file()]
    before = {path: path.read_bytes() for path in files}

    result = compare_zero_shot_baselines(
        ("dev_a", "dev_b"),
        "external_c",
        artifact_root=artifacts,
        development_predictions_path=dev_predictions,
        external_predictions_path=external_predictions,
        model_path=model,
        score_provider=provider,
    )
    write_baseline_comparison_artifacts(result, tmp_path / "output")

    assert {path: path.read_bytes() for path in files} == before
