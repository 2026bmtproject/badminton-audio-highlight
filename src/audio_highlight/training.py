"""Typed inputs for future logistic-regression training; training is not implemented."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from audio_highlight.classifier import CheerClassifier


@dataclass(frozen=True, slots=True)
class LabeledEmbedding:
    """One supervised YAMNet embedding labeled for cheer detection."""

    match_id: str
    start_sec: float
    end_sec: float
    embedding: tuple[float, ...]
    label: Literal[0, 1]


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Configuration reserved for the supported logistic-regression trainer."""

    random_seed: int = 0
    regularization_strength: float = 1.0


class ClassifierTrainer(Protocol):
    """Future trainer interface, deliberately restricted to logistic regression."""

    def fit(
        self,
        examples: tuple[LabeledEmbedding, ...],
        config: TrainingConfig,
    ) -> CheerClassifier:
        ...
