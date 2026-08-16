"""Typed evaluation results; metric computation is not implemented yet."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BinaryClassificationMetrics:
    """Future cheer-classifier evaluation summary."""

    sample_count: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None = None
