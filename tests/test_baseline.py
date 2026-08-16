from __future__ import annotations

import json
from pathlib import Path

from audio_highlight.baseline import (
    BASELINE_ID,
    baseline_metadata,
    write_baseline_metadata,
)


def test_frozen_baseline_configuration_is_complete() -> None:
    value = baseline_metadata()

    assert BASELINE_ID == "yamnet_mean_lr_v1"
    assert value == {
        "baseline_id": "yamnet_mean_lr_v1",
        "feature_extractor": {
            "model_identifier": "https://tfhub.dev/google/yamnet/1",
            "pooling": "mean",
            "embedding_dimension": 1024,
        },
        "audio": {
            "sample_rate_hz": 16000,
            "window_sec": 3.0,
            "hop_sec": 1.0,
            "post_padding_sec": 3.0,
        },
        "classifier": {
            "preprocessing": "StandardScaler",
            "type": "LogisticRegression",
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 2000,
            "class_weight": None,
            "threshold": 0.5,
        },
        "evaluation": {"protocol": "Leave-One-Match-Out"},
    }


def test_baseline_json_round_trip_contains_no_fitted_model(tmp_path: Path) -> None:
    path = write_baseline_metadata(tmp_path / "baseline.json")
    value = json.loads(path.read_text(encoding="utf-8"))

    assert value == baseline_metadata()
    assert path.suffix == ".json"
    assert "pickle" not in path.read_text(encoding="utf-8").lower()
