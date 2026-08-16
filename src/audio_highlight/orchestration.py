"""Segment-level planning that never changes upstream rally boundaries."""

from __future__ import annotations

import math
from dataclasses import dataclass

from audio_highlight.contracts import SegmentsArtifact


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    """Outer audio analysis settings (not YAMNet's internal patch settings)."""

    sample_rate_hz: int = 16_000
    window_sec: float = 3.0
    hop_sec: float = 1.0
    post_padding_sec: float = 3.0

    def __post_init__(self) -> None:
        if self.sample_rate_hz != 16_000:
            raise ValueError("YAMNet input sample rate must be 16000 Hz")
        for name in ("window_sec", "hop_sec"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not math.isfinite(self.post_padding_sec) or self.post_padding_sec < 0:
            raise ValueError("post_padding_sec must be non-negative and finite")


@dataclass(frozen=True, slots=True)
class SegmentAnalysisSpan:
    """An immutable upstream segment plus its separate padded audio read span."""

    segment_index: int
    segment_start_sec: float
    segment_end_sec: float
    analysis_start_sec: float
    analysis_end_sec: float


def build_analysis_spans(
    artifact: SegmentsArtifact,
    config: InferenceConfig | None = None,
    *,
    media_duration_sec: float | None = None,
) -> tuple[SegmentAnalysisSpan, ...]:
    """Plan absolute audio ranges, adding post-padding without mutating segments."""

    settings = config or InferenceConfig()
    if media_duration_sec is not None:
        if not math.isfinite(media_duration_sec) or media_duration_sec < 0:
            raise ValueError("media_duration_sec must be non-negative and finite")
        latest_segment_end = max(segment.end_sec for segment in artifact.segments)
        if media_duration_sec < latest_segment_end:
            raise ValueError("media_duration_sec precedes an upstream segment end")

    spans: list[SegmentAnalysisSpan] = []
    for indexed in artifact.indexed_segments:
        segment = indexed.segment
        analysis_end = segment.end_sec + settings.post_padding_sec
        if media_duration_sec is not None:
            analysis_end = min(analysis_end, media_duration_sec)
        spans.append(
            SegmentAnalysisSpan(
                segment_index=indexed.segment_index,
                segment_start_sec=segment.start_sec,
                segment_end_sec=segment.end_sec,
                analysis_start_sec=segment.start_sec,
                analysis_end_sec=analysis_end,
            )
        )
    return tuple(spans)
