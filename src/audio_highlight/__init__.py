"""Reusable badminton audio-highlight domain package."""

from audio_highlight.contracts import Segment, SegmentsArtifact, load_segments_artifact
from audio_highlight.windows import (
    AnalysisWindow,
    InferenceConfig,
    SegmentAnalysisSpan,
    build_analysis_spans,
    build_analysis_windows,
)

__all__ = [
    "AnalysisWindow",
    "InferenceConfig",
    "Segment",
    "SegmentAnalysisSpan",
    "SegmentsArtifact",
    "build_analysis_spans",
    "build_analysis_windows",
    "load_segments_artifact",
]

__version__ = "0.1.0"
