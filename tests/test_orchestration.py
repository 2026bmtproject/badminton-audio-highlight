from __future__ import annotations

import pytest

from audio_highlight.contracts import Segment, SegmentsArtifact
from audio_highlight.orchestration import InferenceConfig, build_analysis_spans


def artifact() -> SegmentsArtifact:
    return SegmentsArtifact(
        segments=(
            Segment(
                start_frame=30,
                end_frame=120,
                start_sec=1.0,
                end_sec=4.0,
                duration_sec=3.0,
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


def test_media_cannot_end_before_upstream_segment() -> None:
    with pytest.raises(ValueError, match="precedes an upstream segment end"):
        build_analysis_spans(artifact(), media_duration_sec=3.5)
