"""YAMNet inference and deterministic mean pooling for outer audio windows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from audio_highlight.audio import AudioWindow, MONO_CHANNELS, YAMNET_SAMPLE_RATE_HZ

YAMNET_MODEL_HANDLE = "https://tfhub.dev/google/yamnet/1"
YAMNET_EMBEDDING_SIZE = 1024


class YamNetError(RuntimeError):
    """Raised when YAMNet input or output violates the embedding contract."""


class YamNetModel(Protocol):
    """Minimal callable surface exposed by the TensorFlow Hub SavedModel."""

    def __call__(self, waveform: object) -> tuple[object, object, object]:
        ...


@dataclass(frozen=True, slots=True)
class EmbeddedWindow:
    """One mean-pooled YAMNet embedding tied to an inference window."""

    segment_index: int
    start_sec: float
    end_sec: float
    embedding: NDArray[np.float32]

    def __post_init__(self) -> None:
        if self.embedding.shape != (YAMNET_EMBEDDING_SIZE,):
            raise ValueError(
                f"embedding must have shape ({YAMNET_EMBEDDING_SIZE},), "
                f"got {self.embedding.shape}"
            )
        if self.embedding.dtype != np.float32:
            raise ValueError("embedding must have dtype float32")
        if not np.isfinite(self.embedding).all():
            raise ValueError("embedding must contain only finite values")


def _as_numpy(value: object) -> NDArray[Any]:
    numpy_method = getattr(value, "numpy", None)
    if callable(numpy_method):
        value = numpy_method()
    return np.asarray(value)


def validate_raw_embeddings(value: object) -> NDArray[np.float32]:
    """Validate YAMNet's patch matrix without changing its patch semantics."""

    embeddings = _as_numpy(value)
    if embeddings.ndim != 2:
        raise YamNetError(
            f"YAMNet embeddings must have rank 2, got shape {embeddings.shape}"
        )
    if embeddings.shape[1] != YAMNET_EMBEDDING_SIZE:
        raise YamNetError(
            f"YAMNet embedding width must be {YAMNET_EMBEDDING_SIZE}, "
            f"got {embeddings.shape[1]}"
        )
    if embeddings.shape[0] == 0:
        raise YamNetError("YAMNet returned no embedding patches")
    if not np.issubdtype(embeddings.dtype, np.floating):
        raise YamNetError(f"YAMNet embeddings must be floating point, got {embeddings.dtype}")
    if not np.isfinite(embeddings).all():
        raise YamNetError("YAMNet embeddings contain NaN or Inf")
    return np.asarray(embeddings, dtype=np.float32)


def mean_pool_embeddings(value: object) -> NDArray[np.float32]:
    """Mean-pool ``(num_patches, 1024)`` into one read-only ``(1024,)`` vector."""

    embeddings = validate_raw_embeddings(value)
    pooled = np.mean(embeddings, axis=0, dtype=np.float32)
    pooled = np.asarray(pooled, dtype=np.float32)
    if pooled.shape != (YAMNET_EMBEDDING_SIZE,):
        raise YamNetError(f"pooled embedding has unexpected shape {pooled.shape}")
    if not np.isfinite(pooled).all():
        raise YamNetError("pooled embedding contains NaN or Inf")
    pooled.setflags(write=False)
    return pooled


def _validate_audio_window(window: AudioWindow) -> NDArray[np.float32]:
    if window.sample_rate_hz != YAMNET_SAMPLE_RATE_HZ:
        raise YamNetError(f"YAMNet input must be {YAMNET_SAMPLE_RATE_HZ} Hz")
    if window.channels != MONO_CHANNELS:
        raise YamNetError("YAMNet input must be mono")

    waveform = np.asarray(window.samples)
    if waveform.ndim != 1:
        raise YamNetError(f"YAMNet waveform must be one-dimensional, got {waveform.shape}")
    if waveform.dtype != np.float32:
        raise YamNetError(f"YAMNet waveform must have dtype float32, got {waveform.dtype}")
    if waveform.size == 0:
        raise YamNetError("YAMNet waveform must not be empty")
    if not np.isfinite(waveform).all():
        raise YamNetError("YAMNet waveform contains NaN or Inf")
    minimum = float(np.min(waveform))
    maximum = float(np.max(waveform))
    if minimum < -1.0 or maximum > 1.0:
        raise YamNetError(
            "YAMNet waveform amplitude must be within [-1.0, 1.0], "
            f"got [{minimum}, {maximum}]"
        )
    return waveform


def _load_hub_model(model_handle: str | Path) -> tuple[YamNetModel, Any]:
    try:
        import tensorflow as tf
        import tensorflow_hub as hub
    except ImportError as exc:  # pragma: no cover - dependencies are declared by the project
        raise YamNetError(
            "TensorFlow and tensorflow-hub are required for YAMNet inference"
        ) from exc
    return hub.load(str(model_handle)), tf


class YamNetEmbeddingExtractor:
    """Load YAMNet once and reuse it across any number of ``AudioWindow`` objects."""

    def __init__(
        self,
        model: YamNetModel | None = None,
        *,
        model_handle: str | Path = YAMNET_MODEL_HANDLE,
    ) -> None:
        if model is None:
            self._model, self._tensorflow = _load_hub_model(model_handle)
        else:
            self._model = model
            self._tensorflow = None

    def extract_raw(self, window: AudioWindow) -> NDArray[np.float32]:
        """Run one existing outer window and return YAMNet's internal patch matrix."""

        waveform = _validate_audio_window(window)
        model_input: object = waveform
        if self._tensorflow is not None:
            model_input = self._tensorflow.convert_to_tensor(
                waveform,
                dtype=self._tensorflow.float32,
            )
        try:
            _, raw_embeddings, _ = self._model(model_input)
        except (TypeError, ValueError) as exc:
            raise YamNetError(
                "YAMNet model must return (scores, embeddings, spectrogram)"
            ) from exc
        return validate_raw_embeddings(raw_embeddings)

    def extract(self, window: AudioWindow) -> EmbeddedWindow:
        """Return the fixed-size mean-pooled embedding for one outer window."""

        embedding = mean_pool_embeddings(self.extract_raw(window))
        return EmbeddedWindow(
            segment_index=window.segment_index,
            start_sec=window.start_sec,
            end_sec=window.end_sec,
            embedding=embedding,
        )
