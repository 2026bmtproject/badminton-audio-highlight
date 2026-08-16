"""Reusable badminton audio-highlight domain package."""

from audio_highlight.contracts import (
    Highlight,
    HighlightsArtifact,
    IndexedSegment,
    Segment,
    SegmentsArtifact,
    load_segments_artifact,
)
from audio_highlight.orchestration import (
    InferenceConfig,
    SegmentAnalysisSpan,
    build_analysis_spans,
)

__all__ = [
    "Highlight",
    "HighlightsArtifact",
    "IndexedSegment",
    "InferenceConfig",
    "Segment",
    "SegmentAnalysisSpan",
    "SegmentsArtifact",
    "build_analysis_spans",
    "load_segments_artifact",
]

__version__ = "0.1.0"
