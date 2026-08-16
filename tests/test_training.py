from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from audio_highlight.audio import AudioSlice
from audio_highlight.training import (
    FeatureDataset,
    TrainingDataError,
    build_feature_dataset,
    import_cheer_labels,
)

CSV_FIELDS = [
    "window_global_id",
    "segment_id",
    "window_id_in_segment",
    "window_start_sec",
    "window_end_sec",
    "window_duration_sec",
    "source_segment_start_sec",
    "source_segment_end_sec",
    "source_segment_duration_sec",
    "label",
    "notes",
    "reviewed",
    "padded_from_short_segment",
    "wav_path",
    "start_label",
    "end_label",
    "has_cheer",
    "cheer_confidence",
]


def historical_rows() -> list[dict[str, str]]:
    return [
        {
            "window_global_id": "10",
            "segment_id": "101",
            "window_id_in_segment": "1",
            "window_start_sec": "1.25",
            "window_end_sec": "4.25",
            "window_duration_sec": "3.0",
            "source_segment_start_sec": "1.25",
            "source_segment_end_sec": "5.0",
            "source_segment_duration_sec": "3.75",
            "label": "",
            "notes": "",
            "reviewed": "True",
            "padded_from_short_segment": "False",
            "wav_path": r"D:\cheer-labeling\missing\seg0101_win001.wav",
            "start_label": "00:01.25",
            "end_label": "00:04.25",
            "has_cheer": "0",
            "cheer_confidence": "1",
        },
        {
            "window_global_id": "11",
            "segment_id": "102",
            "window_id_in_segment": "1",
            "window_start_sec": "4.25",
            "window_end_sec": "7.25",
            "window_duration_sec": "3.0",
            "source_segment_start_sec": "4.25",
            "source_segment_end_sec": "8.0",
            "source_segment_duration_sec": "3.75",
            "label": "",
            "notes": "not reviewed",
            "reviewed": "False",
            "padded_from_short_segment": "False",
            "wav_path": r"D:\cheer-labeling\missing\seg0102_win001.wav",
            "start_label": "00:04.25",
            "end_label": "00:07.25",
            "has_cheer": "",
            "cheer_confidence": "",
        },
        {
            "window_global_id": "12",
            "segment_id": "205",
            "window_id_in_segment": "2",
            "window_start_sec": "2.5",
            "window_end_sec": "5.5",
            "window_duration_sec": "3.0",
            "source_segment_start_sec": "2.0",
            "source_segment_end_sec": "6.0",
            "source_segment_duration_sec": "4.0",
            "label": "",
            "notes": "",
            "reviewed": "1",
            "padded_from_short_segment": "False",
            "wav_path": r"Z:\legacy\does-not-exist\seg0205_win002.wav",
            "start_label": "00:02.50",
            "end_label": "00:05.50",
            "has_cheer": "1",
            "cheer_confidence": "3",
        },
    ]


def write_labels(path: Path, rows: list[dict[str, str]] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows or historical_rows())
    return path


class FakeAudioSource:
    def __init__(self, duration_sec: float = 8.0) -> None:
        self.duration_sec = duration_sec
        self.requests: list[tuple[float, float]] = []
        self.close_count = 0

    def slice_absolute(self, start_sec: float, end_sec: float) -> AudioSlice:
        self.requests.append((start_sec, end_sec))
        sample_count = round((end_sec - start_sec) * 16_000)
        samples = np.zeros(sample_count, dtype=np.float32)
        samples.setflags(write=False)
        return AudioSlice(start_sec, end_sec, 16_000, 1, samples)

    def close(self) -> None:
        self.close_count += 1


class FakeNormalizer:
    def __init__(self, source: FakeAudioSource) -> None:
        self.source = source
        self.call_count = 0

    def normalize(
        self,
        media_path: str | Path,
        cache_path: str | Path,
        *,
        rebuild: bool = False,
    ) -> FakeAudioSource:
        self.call_count += 1
        return self.source


class FakeEmbedder:
    def __init__(self) -> None:
        self.windows: list[AudioSlice] = []

    def embed(self, window: AudioSlice) -> np.ndarray:
        self.windows.append(window)
        return np.full(1024, window.start_sec, dtype=np.float32)


def test_historical_csv_parsing_and_training_identity(tmp_path: Path) -> None:
    imported = import_cheer_labels(
        write_labels(tmp_path / "cheer_labels.csv"),
        match_id="match_002",
    )

    assert imported.summary.total_rows == 3
    assert imported.summary.reviewed_rows == 2
    assert imported.summary.skipped_unreviewed_rows == 1
    assert imported.summary.negative_rows == 1
    assert imported.summary.positive_rows == 1
    assert [window.window_id for window in imported.windows] == [10, 12]
    assert [window.source_segment_id for window in imported.windows] == [101, 205]
    assert all(not hasattr(window, "segment_index") for window in imported.windows)
    assert [(window.start_sec, window.end_sec) for window in imported.windows] == [
        (1.25, 4.25),
        (2.5, 5.5),
    ]
    assert [window.has_cheer for window in imported.windows] == [False, True]
    assert [window.cheer_confidence for window in imported.windows] == [1, 3]


def test_legacy_wav_path_is_preserved_but_not_required(tmp_path: Path) -> None:
    imported = import_cheer_labels(
        write_labels(tmp_path / "cheer_labels.csv"),
        match_id="match_002",
    )

    assert imported.windows[0].source_wav_path == historical_rows()[0]["wav_path"]
    assert not Path(imported.windows[0].source_wav_path or "").exists()


@pytest.mark.parametrize(
    ("start", "end"),
    [("-1", "2"), ("3", "3"), ("4", "3"), ("nan", "3")],
)
def test_invalid_absolute_timestamp_fails(
    tmp_path: Path,
    start: str,
    end: str,
) -> None:
    rows = historical_rows()[:1]
    rows[0]["window_start_sec"] = start
    rows[0]["window_end_sec"] = end

    with pytest.raises(TrainingDataError, match="timestamp"):
        import_cheer_labels(
            write_labels(tmp_path / "cheer_labels.csv", rows),
            match_id="match_002",
        )


@pytest.mark.parametrize("label", ["", "yes", "2", "-1"])
def test_has_cheer_must_be_explicit_binary(tmp_path: Path, label: str) -> None:
    rows = historical_rows()[:1]
    rows[0]["has_cheer"] = label

    with pytest.raises(TrainingDataError, match="has_cheer must be 0 or 1"):
        import_cheer_labels(
            write_labels(tmp_path / "cheer_labels.csv", rows),
            match_id="match_002",
        )


def test_out_of_media_window_reports_identity_and_timeline(tmp_path: Path) -> None:
    source = FakeAudioSource(duration_sec=5.0)

    with pytest.raises(
        TrainingDataError,
        match=r"window 12 \[2.5, 5.5\).*duration 5.0",
    ):
        build_feature_dataset(
            match_id="match_002",
            video_path=tmp_path / "match.mp4",
            labels_path=write_labels(tmp_path / "cheer_labels.csv"),
            output_path=tmp_path / "features.npz",
            normalizer=FakeNormalizer(source),
            extractor_factory=FakeEmbedder,
        )
    assert source.close_count == 1


def test_builder_reuses_source_and_extractor_and_preserves_order(tmp_path: Path) -> None:
    source = FakeAudioSource()
    normalizer = FakeNormalizer(source)
    extractor = FakeEmbedder()
    factory_calls = 0

    def factory() -> FakeEmbedder:
        nonlocal factory_calls
        factory_calls += 1
        return extractor

    result = build_feature_dataset(
        match_id="match_002",
        video_path=tmp_path / "match.mp4",
        labels_path=write_labels(tmp_path / "cheer_labels.csv"),
        output_path=tmp_path / "match_002.npz",
        normalizer=normalizer,
        extractor_factory=factory,
        model_identifier="fake-yamnet",
    )

    assert normalizer.call_count == 1
    assert factory_calls == 1
    assert source.requests == [(1.25, 4.25), (2.5, 5.5)]
    assert len(extractor.windows) == 2
    assert all(not hasattr(window, "segment_index") for window in extractor.windows)
    assert source.close_count == 1
    assert result.dataset.embeddings.shape == (2, 1024)
    assert result.dataset.labels.tolist() == [0, 1]
    assert result.dataset.window_ids.tolist() == [10, 12]
    np.testing.assert_array_equal(result.dataset.embeddings[:, 0], [1.25, 2.5])


def test_builder_never_opens_legacy_wav_or_segments_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels_path = write_labels(tmp_path / "labels" / "cheer_labels.csv")
    original_open = Path.open

    def guarded_open(path: Path, *args: object, **kwargs: object):
        if path.suffix.lower() == ".wav" or path.name == "segments.json":
            raise AssertionError(f"forbidden training input opened: {path}")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    source = FakeAudioSource()

    result = build_feature_dataset(
        match_id="match_002",
        video_path=tmp_path / "match.mp4",
        labels_path=labels_path,
        output_path=tmp_path / "features" / "match_002.npz",
        normalizer=FakeNormalizer(source),
        extractor_factory=FakeEmbedder,
    )

    assert result.dataset.embeddings.shape == (2, 1024)
    assert not (tmp_path / "segments.json").exists()


def test_npz_round_trip_without_pickle(tmp_path: Path) -> None:
    output = tmp_path / "match_002.npz"
    built = build_feature_dataset(
        match_id="match_002",
        video_path=tmp_path / "match.mp4",
        labels_path=write_labels(tmp_path / "cheer_labels.csv"),
        output_path=output,
        normalizer=FakeNormalizer(FakeAudioSource()),
        extractor_factory=FakeEmbedder,
        model_identifier="fake-yamnet",
    )

    loaded = FeatureDataset.load(output)

    np.testing.assert_array_equal(loaded.embeddings, built.dataset.embeddings)
    np.testing.assert_array_equal(loaded.labels, built.dataset.labels)
    np.testing.assert_array_equal(loaded.window_ids, built.dataset.window_ids)
    np.testing.assert_array_equal(loaded.start_secs, [1.25, 2.5])
    np.testing.assert_array_equal(loaded.end_secs, [4.25, 5.5])
    np.testing.assert_array_equal(loaded.source_segment_ids, [101, 205])
    np.testing.assert_array_equal(loaded.cheer_confidences, [1, 3])
    assert loaded.match_id == "match_002"
    assert loaded.embedding_dimension == 1024
    assert loaded.sample_rate_hz == 16_000
    assert loaded.model_identifier == "fake-yamnet"
