from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import audio_highlight.full_inference as full_inference
from audio_highlight.audio import AudioSlice
from audio_highlight.full_inference import (
    FullMatchInferenceError,
    FullMatchWindow,
    SamplingWindow,
    build_sampling_distribution,
    infer_full_match,
    write_full_match_inference_artifacts,
)
from audio_highlight.labeling import (
    CandidateWindow,
    SampleManifest,
    SampleWindow,
    build_candidate_windows,
    segments_sha256,
    write_manifest,
)
from audio_highlight.contracts import load_segments_artifact
from audio_highlight.windows import InferenceConfig


class FakeSource:
    def __init__(self) -> None:
        self.duration_sec = 10.0
        self.slice_calls: list[tuple[float, float]] = []
        self.close_calls = 0

    def slice_absolute(self, start_sec: float, end_sec: float) -> AudioSlice:
        self.slice_calls.append((start_sec, end_sec))
        samples = np.full(48_000, start_sec / 10.0, dtype=np.float32)
        samples.setflags(write=False)
        return AudioSlice(start_sec, end_sec, 16_000, 1, samples)

    def close(self) -> None:
        self.close_calls += 1


class FakeNormalizer:
    def __init__(self, source: FakeSource) -> None:
        self.source = source
        self.calls = 0

    def normalize(
        self,
        media_path: str | Path,
        cache_path: str | Path,
        *,
        rebuild: bool = False,
    ) -> FakeSource:
        self.calls += 1
        return self.source


class FakeExtractor:
    def __init__(self) -> None:
        self.calls = 0

    def embed(self, audio: AudioSlice) -> np.ndarray:
        self.calls += 1
        return np.full(1024, audio.start_sec / 10.0, dtype=np.float32)


class FakeDetector:
    def __init__(self, *, mode: str = "valid") -> None:
        self.threshold = 0.5
        self.calls = 0
        self.mode = mode
        self.metadata = {
            "model_id": "yamnet_mean_lr_v1",
            "feature_extractor": {
                "model_identifier": "https://tfhub.dev/google/yamnet/1",
                "pooling": "mean",
                "embedding_dimension": 1024,
            },
            "audio": {
                "sample_rate_hz": 16_000,
                "window_sec": 3.0,
                "hop_sec": 1.0,
                "post_padding_sec": 3.0,
            },
            "training": {"matches": ["match_train"]},
        }

    def positive_probabilities(self, embeddings: object) -> np.ndarray:
        self.calls += 1
        values = np.asarray(embeddings, dtype=np.float64)[:, 0]
        if self.mode == "nan":
            values[0] = np.nan
        elif self.mode == "range":
            values[0] = 1.1
        elif self.mode == "shape":
            return values[:, None]
        return values


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_segments(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "fps": 30.0,
                "segments": [
                    {
                        "start_frame": 30,
                        "end_frame": 120,
                        "start_sec": 1.0,
                        "end_sec": 4.0,
                        "duration_sec": 3.0,
                    },
                    {
                        "start_frame": 150,
                        "end_frame": 180,
                        "start_sec": 5.0,
                        "end_sec": 6.0,
                        "duration_sec": 1.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_external_predictions(
    path: Path,
    candidates: tuple[CandidateWindow, ...],
    selected: tuple[int, ...] = (0, 5),
    *,
    offset: float = 0.0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        fields = (
            "match_id",
            "sample_rank",
            "segment_index",
            "window_index_in_segment",
            "start_sec",
            "end_sec",
            "true_label",
            "predicted_label",
            "positive_probability",
        )
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for rank, index in enumerate(selected, start=1):
            candidate = candidates[index]
            probability = float(np.float32(candidate.start_sec / 10.0)) + offset
            writer.writerow(
                {
                    "match_id": "future_match",
                    "sample_rank": rank,
                    "segment_index": candidate.segment_index,
                    "window_index_in_segment": candidate.window_index_in_segment,
                    "start_sec": candidate.start_sec,
                    "end_sec": candidate.end_sec,
                    "true_label": rank % 2,
                    "predicted_label": int(probability >= 0.5),
                    "positive_probability": probability,
                }
            )


def _fixture(tmp_path: Path, *, with_sampling: bool = True):
    video = tmp_path / "local_data" / "future_match" / "match.mp4"
    segments = video.with_name("segments.json")
    audio = tmp_path / "artifacts" / "future_match" / "audio" / "audio.f32le"
    model_dir = tmp_path / "artifacts" / "models" / "yamnet_mean_lr_v1"
    video.parent.mkdir(parents=True)
    audio.parent.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    video.write_bytes(b"video")
    audio.write_bytes(np.asarray([0.0], dtype="<f4").tobytes())
    _write_segments(segments)
    (model_dir / "model.npz").write_bytes(b"numeric model")
    (model_dir / "metadata.json").write_text("{}", encoding="utf-8")
    artifact = load_segments_artifact(segments)
    candidates = build_candidate_windows(
        artifact, media_duration_sec=10.0
    )
    manifest_path = None
    predictions_path = None
    if with_sampling:
        selected = (candidates[0], candidates[-1])
        manifest = SampleManifest(
            match_id="future_match",
            sample_size=2,
            seed=42,
            sampling_algorithm_version=1,
            segments_sha256=segments_sha256(segments),
            planner=InferenceConfig(),
            candidate_window_count=len(candidates),
            eligible_segment_count=2,
            windows=tuple(
                SampleWindow(
                    sample_rank=rank,
                    segment_index=candidate.segment_index,
                    window_index_in_segment=candidate.window_index_in_segment,
                    candidate_count_in_segment=candidate.candidate_count_in_segment,
                    relative_window_position=candidate.relative_window_position,
                    start_sec=candidate.start_sec,
                    end_sec=candidate.end_sec,
                )
                for rank, candidate in enumerate(selected, start=1)
            ),
        )
        manifest_path = tmp_path / "manifest.json"
        write_manifest(manifest, manifest_path)
        predictions_path = tmp_path / "predictions.csv"
        _write_external_predictions(predictions_path, candidates)
    return video, segments, audio, model_dir, manifest_path, predictions_path, candidates


def _run(tmp_path: Path, *, detector: FakeDetector | None = None, with_sampling=True):
    paths = _fixture(tmp_path, with_sampling=with_sampling)
    source = FakeSource()
    normalizer = FakeNormalizer(source)
    extractor = FakeExtractor()
    actual_detector = detector or FakeDetector()
    detector_loads = []
    extractor_loads = []

    def load_detector(path: str | Path) -> FakeDetector:
        detector_loads.append(Path(path))
        return actual_detector

    def load_extractor() -> FakeExtractor:
        extractor_loads.append(extractor)
        return extractor

    result = infer_full_match(
        match_id="future_match",
        video_path=paths[0],
        segments_path=paths[1],
        audio_cache_path=paths[2],
        model_dir=paths[3],
        manifest_path=paths[4],
        external_predictions_path=paths[5],
        normalizer=normalizer,
        extractor_factory=load_extractor,
        detector_loader=load_detector,
        generated_at="2026-01-01T00:00:00+00:00",
    )
    return result, source, normalizer, extractor, actual_detector, detector_loads, extractor_loads, paths


def test_full_inference_reuses_canonical_components_and_preserves_identity(tmp_path):
    result, source, normalizer, extractor, detector, detector_loads, extractor_loads, paths = _run(tmp_path)

    assert len(result.windows) == 6
    assert [(item.segment_index, item.window_index_in_segment) for item in result.windows] == [
        (0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)
    ]
    assert [item.candidate_count_in_segment for item in result.windows] == [4, 4, 4, 4, 2, 2]
    assert [item.relative_window_position for item in result.windows] == pytest.approx(
        [0.0, 1 / 3, 2 / 3, 1.0, 0.0, 1.0]
    )
    assert [(item.start_sec, item.end_sec) for item in result.windows] == [
        (1.0, 4.0), (2.0, 5.0), (3.0, 6.0), (4.0, 7.0), (5.0, 8.0), (6.0, 9.0)
    ]
    assert normalizer.calls == 1
    assert len(source.slice_calls) == 6
    assert source.close_calls == 1
    assert extractor.calls == 6 and len(extractor_loads) == 1
    assert detector.calls == 1 and len(detector_loads) == 1
    assert result.summary["sampled_external_probability_equivalence"] == {
        "available": True,
        "matched_window_count": 2,
        "max_absolute_probability_difference": 0.0,
        "rtol": 1e-9,
        "atol": 1e-12,
    }


def test_inference_does_not_modify_model_audio_segments_or_sampling_inputs(tmp_path):
    paths = _fixture(tmp_path)
    protected = (paths[1], paths[2], paths[3] / "model.npz", paths[3] / "metadata.json", paths[4], paths[5])
    before = {path: _sha256(path) for path in protected}
    infer_full_match(
        match_id="future_match",
        video_path=paths[0],
        segments_path=paths[1],
        audio_cache_path=paths[2],
        model_dir=paths[3],
        manifest_path=paths[4],
        external_predictions_path=paths[5],
        normalizer=FakeNormalizer(FakeSource()),
        extractor_factory=FakeExtractor,
        detector_loader=lambda _: FakeDetector(),
    )
    assert {path: _sha256(path) for path in protected} == before


def test_probability_validation_and_frozen_threshold(tmp_path):
    result, *_ = _run(tmp_path)
    probabilities = np.asarray([item.cheer_probability for item in result.windows])
    assert probabilities.shape == (6,)
    assert np.isfinite(probabilities).all()
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert [item.predicted_cheer for item in result.windows] == [0, 0, 0, 0, 1, 1]
    assert result.summary["threshold"] == 0.5


@pytest.mark.parametrize("mode", ["nan", "range", "shape"])
def test_invalid_detector_probabilities_are_rejected(tmp_path, mode):
    with pytest.raises(FullMatchInferenceError, match="probabilities"):
        _run(tmp_path, detector=FakeDetector(mode=mode))


def test_external_prediction_difference_fails(tmp_path):
    paths = _fixture(tmp_path)
    _write_external_predictions(paths[5], paths[6], offset=1e-4)
    with pytest.raises(FullMatchInferenceError, match="differ"):
        infer_full_match(
            match_id="future_match",
            video_path=paths[0],
            segments_path=paths[1],
            audio_cache_path=paths[2],
            model_dir=paths[3],
            manifest_path=paths[4],
            external_predictions_path=paths[5],
            normalizer=FakeNormalizer(FakeSource()),
            extractor_factory=FakeExtractor,
            detector_loader=lambda _: FakeDetector(),
        )


def test_sampling_distribution_bins_statistics_and_segment_weighting(tmp_path):
    result, *_ = _run(tmp_path)
    distribution = result.sampling_distribution
    relative = distribution["relative_window_position"]
    assert relative["all_candidate_windows"]["count"] == 6
    assert relative["sampled_windows"]["count"] == 2
    assert relative["all_candidate_windows"]["median"] == pytest.approx(0.5)
    assert relative["sampled_windows"]["mean"] == pytest.approx(0.5)
    assert sum(item["all_candidate_count"] for item in relative["fixed_bins"]) == 6
    assert sum(item["sampled_count"] for item in relative["fixed_bins"]) == 2
    assert relative["fixed_bins"][-1]["upper_inclusive"] is True
    assert relative["fixed_bins"][-1]["sampled_count"] == 1
    assert relative["descriptive_distances"]["kolmogorov_smirnov_statistic"] >= 0
    weighting = distribution["segment_weighting"]
    assert weighting["all_candidate_windows"] == {
        "segment_count": 2,
        "mean_windows_per_segment": 3.0,
        "median_windows_per_segment": 3.0,
        "min_windows_per_segment": 2,
        "max_windows_per_segment": 4,
    }
    assert weighting["sampled_windows"]["max_windows_per_segment"] == 1


def test_sampling_membership_is_exact_and_has_no_labels_or_probabilities(tmp_path):
    result, *_ = _run(tmp_path)
    assert [item.is_in_labeled_sample for item in result.sampling_windows] == [
        True, False, False, False, False, True
    ]
    output = write_full_match_inference_artifacts(result, tmp_path / "output")
    with output.sampling_windows_csv.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert sum(row["is_in_labeled_sample"] == "true" for row in rows) == 2
    assert "true_label" not in rows[0]
    assert "cheer_probability" not in rows[0]
    with output.cheer_windows_csv.open(encoding="utf-8", newline="") as file:
        cheer_rows = list(csv.DictReader(file))
    assert "true_label" not in cheer_rows[0]
    assert [int(row["segment_index"]) for row in cheer_rows] == [0, 0, 0, 0, 1, 1]


def test_summary_quantiles_and_hash_provenance(tmp_path):
    result, *rest = _run(tmp_path)
    paths = rest[-1]
    probability = result.summary["probability"]
    assert probability["min"] == pytest.approx(0.1)
    assert probability["median"] == pytest.approx(0.35)
    assert probability["max"] == pytest.approx(0.6)
    assert result.summary["predicted_positive_count"] == 2
    assert result.metadata["model"]["model_sha256"] == _sha256(paths[3] / "model.npz")
    assert result.metadata["segmentation"]["segments_sha256"] == _sha256(paths[1])
    assert result.metadata["audio"]["audio_cache_sha256"] == _sha256(paths[2])
    assert all(not Path(value["path"]).is_absolute() for value in result.metadata["inputs"].values())


def test_generic_match_without_labeling_artifacts_works(tmp_path):
    result, *_ = _run(tmp_path, with_sampling=False)
    assert len(result.windows) == 6
    assert not any(item.is_in_labeled_sample for item in result.sampling_windows)
    assert result.sampling_distribution["relative_window_position"]["sampled_windows"] is None
    assert result.summary["sampled_external_probability_equivalence"]["available"] is False


def test_duplicate_window_identity_is_rejected():
    item = FullMatchWindow("m", 0, 0, 1, 0.5, 1.0, 4.0, 0.5, 1)
    with pytest.raises(FullMatchInferenceError, match="ordering|duplicate"):
        full_inference._validate_output_windows((item, item), 0.5, 3.0)


def test_atomic_writer_does_not_publish_partial_artifacts(tmp_path, monkeypatch):
    result, *_ = _run(tmp_path)
    output = tmp_path / "failed-output"

    def fail_plot(path, windows):
        raise RuntimeError("plot failed")

    monkeypatch.setattr(full_inference, "_write_sampling_plot", fail_plot)
    with pytest.raises(RuntimeError, match="plot failed"):
        write_full_match_inference_artifacts(result, output)
    assert not output.exists()


def test_atomic_writer_restores_previous_set_if_publish_fails(tmp_path, monkeypatch):
    result, *_ = _run(tmp_path)
    output_dir = tmp_path / "existing-output"
    artifacts = write_full_match_inference_artifacts(result, output_dir)
    paths = tuple(Path(getattr(artifacts, name)) for name in artifacts.__dataclass_fields__)
    before = {path: path.read_bytes() for path in paths}
    real_replace = full_inference.os.replace
    failed = False

    def fail_once(source, destination):
        nonlocal failed
        destination_path = Path(destination)
        source_path = Path(source)
        if (
            not failed
            and destination_path.parent == output_dir
            and source_path.name == "metadata.json"
        ):
            failed = True
            raise OSError("publish failed")
        return real_replace(source, destination)

    monkeypatch.setattr(full_inference.os, "replace", fail_once)
    with pytest.raises(OSError, match="publish failed"):
        write_full_match_inference_artifacts(result, output_dir)
    assert {path: path.read_bytes() for path in paths} == before


def test_artifact_writes_are_deterministic_for_same_result(tmp_path):
    result, *_ = _run(tmp_path)
    first = write_full_match_inference_artifacts(result, tmp_path / "first")
    second = write_full_match_inference_artifacts(result, tmp_path / "second")
    for name in (
        "cheer_windows_csv",
        "metadata_json",
        "summary_json",
        "sampling_distribution_json",
        "sampling_windows_csv",
    ):
        assert getattr(first, name).read_bytes() == getattr(second, name).read_bytes()


def test_exactly_100_manifest_identities_can_align():
    candidates = tuple(
        CandidateWindow(index, 0, 1, float(index), float(index + 3))
        for index in range(100)
    )
    manifest = SampleManifest(
        match_id="match_005",
        sample_size=100,
        seed=42,
        sampling_algorithm_version=1,
        segments_sha256="a" * 64,
        planner=InferenceConfig(),
        candidate_window_count=100,
        eligible_segment_count=100,
        windows=tuple(
            SampleWindow(
                sample_rank=index + 1,
                segment_index=index,
                window_index_in_segment=0,
                candidate_count_in_segment=1,
                relative_window_position=0.5,
                start_sec=float(index),
                end_sec=float(index + 3),
            )
            for index in range(100)
        ),
    )
    assert len(full_inference._sampled_identities(manifest, candidates)) == 100


def test_sampling_distribution_public_function_rejects_empty_input():
    with pytest.raises(FullMatchInferenceError, match="candidate windows"):
        build_sampling_distribution(())
