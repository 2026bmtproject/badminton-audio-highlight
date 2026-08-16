from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

from audio_highlight.contracts import Segment, SegmentsArtifact, load_segments_artifact
from audio_highlight.windows import (
    AnalysisWindow,
    InferenceConfig,
    build_analysis_spans,
    build_analysis_windows,
)

FIXTURES = Path(__file__).parent / "fixtures"


def artifact(
    *,
    start_sec: float = 1.0,
    end_sec: float = 4.0,
) -> SegmentsArtifact:
    return SegmentsArtifact(
        segments=(
            Segment(
                start_frame=30,
                end_frame=120,
                start_sec=start_sec,
                end_sec=end_sec,
                duration_sec=end_sec - start_sec,
            ),
        ),
        fps=30.0,
    )


def test_defaults_match_audio_architecture() -> None:
    config = InferenceConfig()

    assert config.sample_rate_hz == 16_000
    assert config.window_sec == 3.0
    assert config.hop_sec == 1.0
    assert config.post_padding_sec == 3.0


def test_padding_extends_analysis_only() -> None:
    segments = artifact()

    span = build_analysis_spans(segments)[0]

    assert span.segment_start_sec == 1.0
    assert span.segment_end_sec == 4.0
    assert span.analysis_start_sec == 1.0
    assert span.analysis_end_sec == 7.0
    assert segments.segments[0].end_sec == 4.0


def test_padding_is_clamped_to_media_duration() -> None:
    span = build_analysis_spans(artifact(), media_duration_sec=5.5)[0]

    assert span.analysis_end_sec == 5.5


def test_media_duration_always_clamps_analysis_end() -> None:
    span = build_analysis_spans(artifact(), media_duration_sec=3.5)[0]

    assert span.analysis_end_sec == 3.5


def test_general_segment_produces_expected_windows() -> None:
    windows = build_analysis_windows(artifact(start_sec=10.0, end_sec=15.0))

    assert windows == tuple(
        AnalysisWindow(0, start, start + 3.0)
        for start in (10.0, 11.0, 12.0, 13.0, 14.0, 15.0)
    )


def test_short_segment_still_produces_complete_windows() -> None:
    windows = build_analysis_windows(artifact(start_sec=386.233, end_sec=387.3))

    assert windows == (
        AnalysisWindow(0, 386.233, 389.233),
        AnalysisWindow(0, 387.233, 390.233),
    )


def test_post_padding_controls_available_windows() -> None:
    segments = artifact(start_sec=10.0, end_sec=15.0)

    without_padding = build_analysis_windows(
        segments,
        InferenceConfig(post_padding_sec=0.0),
    )
    with_padding = build_analysis_windows(segments)

    assert len(without_padding) == 3
    assert len(with_padding) == 6
    assert with_padding[-1] == AnalysisWindow(0, 15.0, 18.0)


def test_neighboring_segment_spans_and_windows_may_overlap() -> None:
    segments = SegmentsArtifact(
        segments=(
            Segment(300, 450, 10.0, 15.0, 5.0),
            Segment(480, 570, 16.0, 19.0, 3.0),
        ),
        fps=30.0,
    )

    spans = build_analysis_spans(segments)
    windows = build_analysis_windows(segments)

    assert spans[0].analysis_end_sec == 18.0
    assert spans[1].analysis_start_sec == 16.0
    assert AnalysisWindow(0, 15.0, 18.0) in windows
    assert AnalysisWindow(1, 16.0, 19.0) in windows


def test_planning_does_not_modify_upstream_segment() -> None:
    segments = artifact(start_sec=10.0, end_sec=15.0)
    original = asdict(segments.segments[0])

    build_analysis_windows(segments)

    assert asdict(segments.segments[0]) == original


def test_media_duration_clamps_windows() -> None:
    windows = build_analysis_windows(
        artifact(start_sec=10.0, end_sec=15.0),
        media_duration_sec=17.4,
    )

    assert windows[-1] == AnalysisWindow(0, 14.0, 17.0)
    assert all(window.end_sec <= 17.4 for window in windows)


def test_partial_final_window_is_not_created() -> None:
    windows = build_analysis_windows(
        artifact(start_sec=10.0, end_sec=14.5),
    )

    assert windows[-1] == AnalysisWindow(0, 14.0, 17.0)
    assert AnalysisWindow(0, 15.0, 18.0) not in windows


def test_custom_window_size() -> None:
    windows = build_analysis_windows(
        artifact(start_sec=10.0, end_sec=15.0),
        InferenceConfig(window_sec=2.0, post_padding_sec=0.0),
    )

    assert windows == tuple(
        AnalysisWindow(0, start, start + 2.0)
        for start in (10.0, 11.0, 12.0, 13.0)
    )


def test_custom_hop_size() -> None:
    windows = build_analysis_windows(
        artifact(start_sec=10.0, end_sec=15.0),
        InferenceConfig(hop_sec=2.0),
    )

    assert windows == (
        AnalysisWindow(0, 10.0, 13.0),
        AnalysisWindow(0, 12.0, 15.0),
        AnalysisWindow(0, 14.0, 17.0),
    )


def test_timestamps_are_absolute_match_time() -> None:
    windows = build_analysis_windows(artifact(start_sec=386.233, end_sec=387.3))

    assert windows[0].start_sec == 386.233
    assert windows[0].end_sec == 389.233


def test_segment_index_is_the_positional_index() -> None:
    segments = load_segments_artifact(FIXTURES / "segments_sample.json")

    windows = build_analysis_windows(segments)

    assert {window.segment_index for window in windows} == {0, 1, 2}
    assert next(window for window in windows if window.segment_index == 1).start_sec == 391.433


def test_real_segments_regression_fixture() -> None:
    segments = load_segments_artifact(FIXTURES / "segments_sample.json")

    spans = build_analysis_spans(segments)
    windows = build_analysis_windows(segments)

    assert segments.fps == 30.0
    assert spans[0].analysis_start_sec == 386.233
    assert spans[0].analysis_end_sec == 390.3
    counts = [sum(window.segment_index == index for window in windows) for index in range(3)]
    assert counts == [2, 13, 17]
