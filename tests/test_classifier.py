from __future__ import annotations

from audio_highlight.classifier import (
    LOGISTIC_REGRESSION_C,
    LOGISTIC_REGRESSION_MAX_ITER,
    PREDICTION_THRESHOLD,
    build_baseline_classifier,
)


def test_logistic_regression_baseline_is_fixed() -> None:
    pipeline = build_baseline_classifier()
    classifier = pipeline.named_steps["logistic_regression"]

    assert list(pipeline.named_steps) == ["standard_scaler", "logistic_regression"]
    assert LOGISTIC_REGRESSION_C == 1.0
    assert LOGISTIC_REGRESSION_MAX_ITER == 2_000
    assert PREDICTION_THRESHOLD == 0.5
    assert classifier.C == 1.0
    assert classifier.solver == "lbfgs"
    assert classifier.max_iter == 2_000
    assert classifier.class_weight is None
