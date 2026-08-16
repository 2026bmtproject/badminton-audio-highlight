from __future__ import annotations

import json
import inspect
from pathlib import Path

import numpy as np
import pytest

import audio_highlight.dataset as dataset_module
from audio_highlight.audio import AudioSlice
from audio_highlight.labeling import (
    LabelStore,
    LabelingError,
    create_or_load_manifest,
)
from audio_highlight.dataset import (
    FeatureDataset,
    build_feature_dataset,
)


def write_segments(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {
        "fps": 30.0,
        "segments": [
            {
                "start_frame": index * 300,
                "end_frame": index * 300 + 180,
                "start_sec": index * 10.0,
                "end_sec": index * 10.0 + 6.0,
                "duration_sec": 6.0,
            }
            for index in range(4)
        ],
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


class FakeAudioSource:
    duration_sec = 100.0

    def __init__(self) -> None:
        self.requests: list[tuple[float, float]] = []
        self.close_count = 0

    def slice_absolute(self, start_sec: float, end_sec: float) -> AudioSlice:
        self.requests.append((start_sec, end_sec))
        samples = np.zeros(round((end_sec - start_sec) * 16_000), dtype=np.float32)
        samples.setflags(write=False)
        return AudioSlice(start_sec, end_sec, 16_000, 1, samples)

    def close(self) -> None:
        self.close_count += 1


class FakeNormalizer:
    def __init__(self, source: FakeAudioSource) -> None:
        self.source = source
        self.calls = 0

    def normalize(self, media_path, cache_path, *, rebuild=False):
        self.calls += 1
        return self.source


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[AudioSlice] = []

    def embed(self, audio: AudioSlice) -> np.ndarray:
        self.calls.append(audio)
        return np.full(1024, audio.start_sec, dtype=np.float32)


def prepare_completed_labels(tmp_path: Path, *, ambiguous_rank: int = 2):
    manifest_path = tmp_path / "manifest.json"
    manifest = create_or_load_manifest(
        match_id="match_a",
        segments_path=write_segments(tmp_path / "segments.json"),
        manifest_path=manifest_path,
        sample_size=4,
        seed=42,
    ).manifest
    labels_path = tmp_path / "labels.csv"
    store = LabelStore(labels_path, manifest)
    for window in manifest.windows:
        ambiguous = window.sample_rank == ambiguous_rank
        store.record_decision(
            window.sample_rank,
            has_cheer=None if ambiguous else window.sample_rank % 2,
            is_ambiguous=ambiguous,
        )
    return manifest, manifest_path, labels_path


def test_builder_uses_absolute_timestamps_one_source_and_one_embedder(
    tmp_path: Path,
) -> None:
    manifest, manifest_path, labels_path = prepare_completed_labels(tmp_path)
    source = FakeAudioSource()
    normalizer = FakeNormalizer(source)
    embedder = FakeEmbedder()
    factory_calls = 0

    def factory() -> FakeEmbedder:
        nonlocal factory_calls
        factory_calls += 1
        return embedder

    result = build_feature_dataset(
        video_path=tmp_path / "match.mp4",
        labels_path=labels_path,
        manifest_path=manifest_path,
        output_path=tmp_path / "features" / "features.npz",
        normalizer=normalizer,
        extractor_factory=factory,
        model_identifier="fake-yamnet",
    )
    binary_windows = [window for window in manifest.windows if window.sample_rank != 2]

    assert normalizer.calls == 1 and factory_calls == 1
    assert source.close_count == 1
    assert source.requests == [(item.start_sec, item.end_sec) for item in binary_windows]
    assert len(embedder.calls) == 3
    assert result.reviewed == 4
    assert result.binary_included == 3
    assert result.ambiguous_excluded == 1
    assert result.dataset.embeddings.shape == (3, 1024)
    assert result.dataset.segment_indices.tolist() == [
        item.segment_index for item in binary_windows
    ]
    assert not hasattr(result.dataset, "source_segment_ids")


def test_incomplete_labels_fail_before_audio_or_yamnet(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest = create_or_load_manifest(
        match_id="match_a",
        segments_path=write_segments(tmp_path / "segments.json"),
        manifest_path=manifest_path,
        sample_size=4,
    ).manifest
    labels_path = tmp_path / "labels.csv"
    LabelStore(labels_path, manifest).record_decision(
        1, has_cheer=0, is_ambiguous=False
    )
    normalizer = FakeNormalizer(FakeAudioSource())
    factory_called = False

    def factory():
        nonlocal factory_called
        factory_called = True
        return FakeEmbedder()

    with pytest.raises(LabelingError, match="incomplete"):
        build_feature_dataset(
            video_path=tmp_path / "match.mp4",
            labels_path=labels_path,
            manifest_path=manifest_path,
            output_path=tmp_path / "features.npz",
            normalizer=normalizer,
            extractor_factory=factory,
        )

    assert normalizer.calls == 0
    assert factory_called is False


def test_npz_round_trip_without_pickle_or_historical_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, manifest_path, labels_path = prepare_completed_labels(tmp_path)
    output = tmp_path / "features.npz"
    built = build_feature_dataset(
        video_path=tmp_path / "match.mp4",
        labels_path=labels_path,
        manifest_path=manifest_path,
        output_path=output,
        normalizer=FakeNormalizer(FakeAudioSource()),
        extractor_factory=FakeEmbedder,
        model_identifier="fake-yamnet",
    )
    original_load = np.load
    allow_pickle_values: list[bool | None] = []

    def recording_load(*args, **kwargs):
        allow_pickle_values.append(kwargs.get("allow_pickle"))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(dataset_module.np, "load", recording_load)
    loaded = FeatureDataset.load(output)

    np.testing.assert_array_equal(loaded.embeddings, built.dataset.embeddings)
    np.testing.assert_array_equal(loaded.labels, built.dataset.labels)
    np.testing.assert_array_equal(loaded.start_secs, built.dataset.start_secs)
    assert loaded.segments_sha256 == manifest.segments_sha256
    assert loaded.label_source == "current_segments_blind_human"
    assert loaded.sampling_seed == 42
    assert loaded.sampling_algorithm_version == 1
    assert allow_pickle_values == [False]
    assert not hasattr(loaded, "window_ids")
    assert not hasattr(loaded, "source_segment_ids")
    source = inspect.getsource(build_feature_dataset)
    assert "FeatureDataset.load" not in source
    assert "source_segment_id" not in source
