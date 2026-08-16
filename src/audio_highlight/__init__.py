"""Reusable badminton audio-highlight domain package."""

from audio_highlight.audio import (
    AudioRangeError,
    AudioSlice,
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
from audio_highlight.training import (
    FeatureBuildResult,
    FeatureDataset,
    ImportedLabels,
    LabeledWindow,
    TrainingDataError,
    build_feature_dataset,
    import_cheer_labels,
)
from audio_highlight.yamnet import (
    EmbeddedWindow,
    YamNetEmbeddingExtractor,
    YamNetError,
    mean_pool_embeddings,
)

__all__ = [
    "AnalysisWindow",
    "AudioRangeError",
    "AudioSlice",
    "AudioWindow",
    "EmbeddedWindow",
    "FFmpegAudioNormalizer",
    "FeatureBuildResult",
    "FeatureDataset",
    "Highlight",
    "HighlightsArtifact",
    "IndexedSegment",
    "InferenceConfig",
    "ImportedLabels",
    "LabeledWindow",
    "NormalizedAudioSource",
    "Segment",
    "SegmentAnalysisSpan",
    "SegmentsArtifact",
    "TrainingDataError",
    "YamNetEmbeddingExtractor",
    "YamNetError",
    "build_analysis_spans",
    "build_analysis_windows",
    "build_feature_dataset",
    "import_cheer_labels",
    "load_segments_artifact",
    "mean_pool_embeddings",
    "timestamp_to_sample_index",
]

__version__ = "0.1.0"
