"""Segment-level planning that never changes upstream rally boundaries."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal

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


@dataclass(frozen=True, slots=True)
class AnalysisWindow:
    """One complete outer analysis window in absolute match timestamps."""

    segment_index: int
    start_sec: float
    end_sec: float


def _decimal(value: float) -> Decimal:
    """Preserve the user-facing decimal value instead of accumulating float error."""

    return Decimal(str(value))


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

    spans: list[SegmentAnalysisSpan] = []
    for indexed in artifact.indexed_segments:
        segment = indexed.segment
        analysis_end = _decimal(segment.end_sec) + _decimal(settings.post_padding_sec)
        if media_duration_sec is not None:
            analysis_end = min(analysis_end, _decimal(media_duration_sec))
        spans.append(
            SegmentAnalysisSpan(
                segment_index=indexed.segment_index,
                segment_start_sec=segment.start_sec,
                segment_end_sec=segment.end_sec,
                analysis_start_sec=segment.start_sec,
                analysis_end_sec=float(analysis_end),
            )
        )
    return tuple(spans)


def build_analysis_windows(
    artifact: SegmentsArtifact,
    config: InferenceConfig | None = None,
    *,
    media_duration_sec: float | None = None,
) -> tuple[AnalysisWindow, ...]:
    """Split every padded segment span into complete absolute-time windows.

    Windows start at ``analysis_start_sec + n * hop_sec``. A window is included
    exactly when its full end timestamp is no later than ``analysis_end_sec``.
    Trailing partial windows are discarded, and windows belonging to neighboring
    segments may overlap.
    """

    settings = config or InferenceConfig()
    window_size = _decimal(settings.window_sec)
    hop_size = _decimal(settings.hop_sec)
    windows: list[AnalysisWindow] = []

    for span in build_analysis_spans(
        artifact,
        settings,
        media_duration_sec=media_duration_sec,
    ):
        analysis_start = _decimal(span.analysis_start_sec)
        analysis_end = _decimal(span.analysis_end_sec)
        last_full_start = analysis_end - window_size
        if last_full_start < analysis_start:
            continue

        window_count = int((last_full_start - analysis_start) // hop_size) + 1
        for window_index in range(window_count):
            start = analysis_start + window_index * hop_size
            windows.append(
                AnalysisWindow(
                    segment_index=span.segment_index,
                    start_sec=float(start),
                    end_sec=float(start + window_size),
                )
            )

    return tuple(windows)
