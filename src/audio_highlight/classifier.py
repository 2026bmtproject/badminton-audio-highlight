"""Interfaces for cheer inference from YAMNet embeddings."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from audio_highlight.yamnet import TimedEmbedding


@dataclass(frozen=True, slots=True)
class ClassifierMetadata:
    """Identity of the supported downstream classifier family."""

    algorithm: Literal["logistic_regression"] = "logistic_regression"
    model_version: str | None = None


@dataclass(frozen=True, slots=True)
class CheerPrediction:
    """Cheer probability for an absolute match-time embedding interval."""

    start_sec: float
    end_sec: float
    probability: float


class CheerClassifier(Protocol):
    """Logistic-regression inference boundary; no model loading is implemented yet."""

    @property
    def metadata(self) -> ClassifierMetadata:
        ...

    def predict(self, embeddings: Sequence[TimedEmbedding]) -> Sequence[CheerPrediction]:
        ...
