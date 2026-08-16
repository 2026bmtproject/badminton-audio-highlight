"""Interfaces for cheer inference from YAMNet embeddings."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from audio_highlight.yamnet import EmbeddedWindow

LOGISTIC_REGRESSION_C = 1.0
LOGISTIC_REGRESSION_MAX_ITER = 2_000
PREDICTION_THRESHOLD = 0.5


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

    def predict(self, embeddings: Sequence[EmbeddedWindow]) -> Sequence[CheerPrediction]:
        ...


def build_baseline_classifier() -> Pipeline:
    """Build the fixed Logistic Regression baseline without tuning."""

    return Pipeline(
        steps=[
            ("standard_scaler", StandardScaler()),
            (
                "logistic_regression",
                LogisticRegression(
                    C=LOGISTIC_REGRESSION_C,
                    class_weight=None,
                    max_iter=LOGISTIC_REGRESSION_MAX_ITER,
                    solver="lbfgs",
                ),
            ),
        ]
    )
