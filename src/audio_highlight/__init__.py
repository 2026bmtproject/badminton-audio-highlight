"""Reusable badminton audio-highlight domain package."""

from audio_highlight.audio import (
    AudioRangeError,
    AudioWindow,
    FFmpegAudioNormalizer,
    NormalizedAudioSource,
    timestamp_to_sample_index,
)
from audio_highlight.contracts import (
    Highlight,
    HighlightsArtifact,
    IndexedSegment,
    Segment,
    SegmentsArtifact,
    load_segments_artifact,
)
from audio_highlight.orchestration import (
    AnalysisWindow,
    InferenceConfig,
    SegmentAnalysisSpan,
    build_analysis_spans,
    build_analysis_windows,
)

__all__ = [
    "AnalysisWindow",
    "AudioRangeError",
    "AudioWindow",
    "FFmpegAudioNormalizer",
    "Highlight",
    "HighlightsArtifact",
    "IndexedSegment",
    "InferenceConfig",
    "NormalizedAudioSource",
    "Segment",
    "SegmentAnalysisSpan",
    "SegmentsArtifact",
    "build_analysis_spans",
    "build_analysis_windows",
    "load_segments_artifact",
    "timestamp_to_sample_index",
]

__version__ = "0.1.0"
