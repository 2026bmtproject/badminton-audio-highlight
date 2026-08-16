from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

from audio_highlight.label_app import parse_app_args
from audio_highlight.labeling import (
    SAMPLING_ALGORITHM_VERSION,
    LabelStore,
    LabelingError,
    build_candidate_windows,
    label_statistics,
    create_or_load_manifest,
    default_segments_path,
    load_manifest,
    sample_segment_diverse_windows,
)
from audio_highlight.contracts import load_segments_artifact
from audio_highlight.windows import InferenceConfig, build_analysis_windows


def write_segments(path: Path, count: int = 8, *, duration: float = 6.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    segments = []
    for index in range(count):
        start = index * 20.0 + 1.0
        end = start + duration
        segments.append(
            {
                "start_frame": round(start * 30),
                "end_frame": round(end * 30),
                "start_sec": start,
                "end_sec": end,
                "duration_sec": duration,
            }
        )
    path.write_text(json.dumps({"fps": 30.0, "segments": segments}), encoding="utf-8")
    return path


def test_candidates_reuse_current_parser_and_exact_inference_planner(tmp_path: Path) -> None:
    path = write_segments(tmp_path / "segments.json", count=3)
    artifact = load_segments_artifact(path)

    candidates = build_candidate_windows(artifact)
    planned = build_analysis_windows(artifact)

    assert [(item.segment_index, item.start_sec, item.end_sec) for item in candidates] == [
        (item.segment_index, item.start_sec, item.end_sec) for item in planned
    ]
    assert candidates[0].window_index_in_segment == 0
    assert candidates[0].candidate_count_in_segment == 7


def test_planner_defaults_and_post_padding_semantics_are_unchanged(tmp_path: Path) -> None:
    artifact = load_segments_artifact(write_segments(tmp_path / "segments.json", count=1, duration=1.0))
    config = InferenceConfig()

    candidates = build_candidate_windows(artifact, config)

    assert config == InferenceConfig(
        sample_rate_hz=16_000,
        window_sec=3.0,
        hop_sec=1.0,
        post_padding_sec=3.0,
    )
    assert [(item.start_sec, item.end_sec) for item in candidates] == [
        (1.0, 4.0),
        (2.0, 5.0),
    ]


def test_sampling_is_deterministic_seeded_and_label_independent(tmp_path: Path) -> None:
    artifact = load_segments_artifact(write_segments(tmp_path / "segments.json", count=20))
    candidates = build_candidate_windows(artifact)

    first = sample_segment_diverse_windows(candidates, sample_size=15, seed=42)
    repeated = sample_segment_diverse_windows(candidates, sample_size=15, seed=42)
    changed = sample_segment_diverse_windows(candidates, sample_size=15, seed=43)

    assert first == repeated
    assert first != changed
    assert all(not hasattr(item, "has_cheer") for item in first)
    assert all(not hasattr(item, "source_segment_id") for item in first)


def test_one_hundred_eligible_segments_give_one_hundred_distinct_indices(
    tmp_path: Path,
) -> None:
    artifact = load_segments_artifact(write_segments(tmp_path / "segments.json", count=120))

    sample = sample_segment_diverse_windows(
        build_candidate_windows(artifact), sample_size=100, seed=42
    )

    assert len(sample) == 100
    assert len({item.segment_index for item in sample}) == 100
    assert max(
        sum(other.segment_index == item.segment_index for other in sample)
        for item in sample
    ) == 1


def test_second_pass_balances_segments_and_maximizes_first_extra_distance(
    tmp_path: Path,
) -> None:
    artifact = load_segments_artifact(write_segments(tmp_path / "segments.json", count=2, duration=10.0))
    candidates = build_candidate_windows(artifact)

    sample = sample_segment_diverse_windows(candidates, sample_size=4, seed=11)

    for segment_index in (0, 1):
        selected = [
            item.window_index_in_segment
            for item in sample
            if item.segment_index == segment_index
        ]
        count = next(
            item.candidate_count_in_segment
            for item in sample
            if item.segment_index == segment_index
        )
        assert len(selected) == 2
        first, second = selected
        assert abs(second - first) == max(first, count - 1 - first)


def test_manifest_round_trip_resume_and_segments_sha_mismatch(tmp_path: Path) -> None:
    segments = write_segments(tmp_path / "match" / "segments.json", count=12)
    manifest_path = tmp_path / "artifacts" / "labeling" / "sample_manifest.json"
    original_hash = hashlib.sha256(segments.read_bytes()).hexdigest()

    created = create_or_load_manifest(
        match_id="match_a",
        segments_path=segments,
        manifest_path=manifest_path,
        sample_size=10,
        seed=42,
    )
    resumed = create_or_load_manifest(
        match_id="match_a",
        segments_path=segments,
        manifest_path=manifest_path,
        sample_size=10,
        seed=42,
    )

    assert created.created is True
    assert resumed.created is False
    assert resumed.manifest.windows == created.manifest.windows
    assert load_manifest(manifest_path) == created.manifest
    assert created.manifest.segments_sha256 == original_hash
    assert created.manifest.sampling_algorithm_version == SAMPLING_ALGORITHM_VERSION

    write_segments(segments, count=13)
    with pytest.raises(LabelingError, match="segments SHA-256"):
        create_or_load_manifest(
            match_id="match_a",
            segments_path=segments,
            manifest_path=manifest_path,
            sample_size=10,
            seed=42,
        )


def test_label_round_trip_ambiguous_and_resume(tmp_path: Path) -> None:
    manifest = create_or_load_manifest(
        match_id="match_a",
        segments_path=write_segments(tmp_path / "segments.json", count=5),
        manifest_path=tmp_path / "manifest.json",
        sample_size=5,
        seed=42,
    ).manifest
    labels_path = tmp_path / "labels.csv"
    store = LabelStore(labels_path, manifest)
    store.record_decision(
        1,
        has_cheer=1,
        is_ambiguous=False,
        reviewed_at="2026-01-01T00:00:00+00:00",
    )
    store.record_decision(
        2,
        has_cheer=None,
        is_ambiguous=True,
        reviewed_at="2026-01-01T00:00:01+00:00",
    )

    resumed = LabelStore(labels_path, manifest)
    resumed.record_decision(1, has_cheer=0, is_ambiguous=False)
    decisions = LabelStore(labels_path, manifest).decisions
    stats = label_statistics(manifest, decisions)

    assert set(decisions) == {1, 2}
    assert decisions[1].has_cheer == 0
    assert decisions[2].has_cheer is None and decisions[2].is_ambiguous
    assert (stats.reviewed, stats.remaining, stats.ambiguous_count) == (2, 3, 1)


def test_app_uses_canonical_paths_and_defaults() -> None:
    parsed = parse_app_args(["--match-id", "match_a"])

    assert parsed.video == Path("local_data/match_a/match.mp4")
    assert parsed.segments == Path("local_data/match_a/segments.json")
    assert parsed.segments == default_segments_path(parsed.video)
    assert parsed.manifest == Path("artifacts/match_a/labeling/sample_manifest.json")
    assert parsed.labels == Path("artifacts/match_a/labeling/labels.csv")
    assert parsed.audio_cache == Path("artifacts/match_a/audio/audio.f32le")
    assert parsed.sample_size == 100 and parsed.seed == 42


def test_segments_json_is_read_only_during_manifest_creation(tmp_path: Path) -> None:
    segments = write_segments(tmp_path / "segments.json", count=6)
    before = segments.read_bytes()

    create_or_load_manifest(
        match_id="match_a",
        segments_path=segments,
        manifest_path=tmp_path / "manifest.json",
        sample_size=5,
    )

    assert segments.read_bytes() == before


def test_sampling_is_label_independent() -> None:
    assert "has_cheer" not in inspect.getsource(sample_segment_diverse_windows)
