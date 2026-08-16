"""Machine-readable identity for the frozen detector baseline."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from audio_highlight.audio import YAMNET_SAMPLE_RATE_HZ
from audio_highlight.classifier import (
    LOGISTIC_REGRESSION_C,
    LOGISTIC_REGRESSION_MAX_ITER,
    PREDICTION_THRESHOLD,
)
from audio_highlight.windows import InferenceConfig
from audio_highlight.yamnet import YAMNET_EMBEDDING_SIZE, YAMNET_MODEL_HANDLE

BASELINE_ID = "yamnet_mean_lr_v1"
DEFAULT_BASELINE_PATH = Path("artifacts/baselines") / f"{BASELINE_ID}.json"


def baseline_metadata() -> dict[str, Any]:
    """Return the complete immutable configuration of the v1 detector baseline."""

    planner = InferenceConfig()
    return {
        "baseline_id": BASELINE_ID,
        "feature_extractor": {
            "model_identifier": YAMNET_MODEL_HANDLE,
            "pooling": "mean",
            "embedding_dimension": YAMNET_EMBEDDING_SIZE,
        },
        "audio": {
            "sample_rate_hz": YAMNET_SAMPLE_RATE_HZ,
            "window_sec": planner.window_sec,
            "hop_sec": planner.hop_sec,
            "post_padding_sec": planner.post_padding_sec,
        },
        "classifier": {
            "preprocessing": "StandardScaler",
            "type": "LogisticRegression",
            "C": LOGISTIC_REGRESSION_C,
            "solver": "lbfgs",
            "max_iter": LOGISTIC_REGRESSION_MAX_ITER,
            "class_weight": None,
            "threshold": PREDICTION_THRESHOLD,
        },
        "evaluation": {"protocol": "Leave-One-Match-Out"},
    }


def write_baseline_metadata(path: str | Path = DEFAULT_BASELINE_PATH) -> Path:
    """Atomically write baseline configuration; no fitted model is serialized."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(baseline_metadata(), file, ensure_ascii=False, indent=2)
            file.write("\n")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output
