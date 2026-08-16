"""Frozen Logistic Regression baseline and sklearn-independent inference."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from audio_highlight.yamnet import (
    YAMNET_EMBEDDING_SIZE,
    YAMNET_MODEL_HANDLE,
    EmbeddedWindow,
)
from audio_highlight.audio import YAMNET_SAMPLE_RATE_HZ
from audio_highlight.windows import InferenceConfig

BASELINE_ID = "yamnet_mean_lr_v1"
LOGISTIC_REGRESSION_C = 1.0
LOGISTIC_REGRESSION_MAX_ITER = 2_000
PREDICTION_THRESHOLD = 0.5
_MODEL_ARRAY_NAMES = {
    "scaler_mean",
    "scaler_scale",
    "lr_coef",
    "lr_intercept",
    "classes",
}


class ModelArtifactError(ValueError):
    """Raised when an exported detector artifact violates its frozen contract."""


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
    @property
    def metadata(self) -> ClassifierMetadata:
        ...

    def predict(self, embeddings: Sequence[EmbeddedWindow]) -> Sequence[CheerPrediction]:
        ...


def build_baseline_classifier() -> Any:
    """Build the fixed sklearn training pipeline without tuning."""

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

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


def stable_sigmoid(logits: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
    """Compute sigmoid without overflow for arbitrarily large finite logits."""

    values = np.asarray(logits, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ModelArtifactError("logits must be finite")
    result = np.empty(values.shape, dtype=np.float64)
    nonnegative = values >= 0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exp_values = np.exp(values[~nonnegative])
    result[~nonnegative] = exp_values / (1.0 + exp_values)
    return result


class ExportedCheerDetector:
    """Pure-numpy inference over an exported StandardScaler + LR detector."""

    __slots__ = (
        "classes",
        "lr_coef",
        "lr_intercept",
        "metadata",
        "scaler_mean",
        "scaler_scale",
        "threshold",
    )

    def __init__(
        self,
        *,
        scaler_mean: NDArray[np.float64],
        scaler_scale: NDArray[np.float64],
        lr_coef: NDArray[np.float64],
        lr_intercept: float,
        classes: NDArray[np.int64],
        metadata: dict[str, Any],
    ) -> None:
        self.scaler_mean = _validated_float_vector(
            scaler_mean, "scaler_mean", positive=False
        )
        self.scaler_scale = _validated_float_vector(
            scaler_scale, "scaler_scale", positive=True
        )
        self.lr_coef = _validated_float_vector(lr_coef, "lr_coef", positive=False)
        if not math.isfinite(lr_intercept):
            raise ModelArtifactError("lr_intercept must be finite")
        self.lr_intercept = float(lr_intercept)
        class_values = np.asarray(classes)
        if class_values.shape != (2,) or not np.array_equal(class_values, [0, 1]):
            raise ModelArtifactError("classes must equal [0, 1]")
        self.classes = np.asarray(class_values, dtype=np.int64)
        self.classes.setflags(write=False)
        self.metadata = _validate_metadata(metadata)
        self.threshold = float(self.metadata["classifier"]["threshold"])

    @classmethod
    def load(cls, model_dir: str | Path) -> ExportedCheerDetector:
        """Load numeric arrays with ``allow_pickle=False`` and validate integrity."""

        directory = Path(model_dir)
        model_path = directory / "model.npz"
        metadata_path = directory / "metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelArtifactError(f"invalid metadata.json: {exc}") from exc
        validated_metadata = _validate_metadata(metadata)
        expected_sha256 = validated_metadata.get("model_sha256")
        if expected_sha256 is not None:
            actual_sha256 = _sha256(model_path)
            if expected_sha256 != actual_sha256:
                raise ModelArtifactError("model.npz SHA-256 does not match metadata")
        try:
            with np.load(model_path, allow_pickle=False) as values:
                missing = _MODEL_ARRAY_NAMES - set(values.files)
                if missing:
                    raise ModelArtifactError(
                        "model.npz is missing arrays: " + ", ".join(sorted(missing))
                    )
                scaler_mean = np.asarray(values["scaler_mean"])
                scaler_scale = np.asarray(values["scaler_scale"])
                lr_coef = np.asarray(values["lr_coef"])
                lr_intercept_array = np.asarray(values["lr_intercept"])
                classes = np.asarray(values["classes"])
        except (OSError, ValueError) as exc:
            raise ModelArtifactError(f"invalid model.npz: {exc}") from exc
        if lr_intercept_array.shape != () or lr_intercept_array.dtype != np.float64:
            raise ModelArtifactError("lr_intercept must be a float64 scalar")
        return cls(
            scaler_mean=scaler_mean,
            scaler_scale=scaler_scale,
            lr_coef=lr_coef,
            lr_intercept=float(lr_intercept_array.item()),
            classes=classes,
            metadata=validated_metadata,
        )

    def positive_probability(self, embedding: object) -> float:
        """Return P(cheer) for exactly one ``(1024,)`` embedding."""

        values = _validate_embedding(embedding, expected_ndim=1)
        standardized = (values - self.scaler_mean) / self.scaler_scale
        logit = np.asarray(
            np.dot(self.lr_coef, standardized) + self.lr_intercept,
            dtype=np.float64,
        )
        return float(stable_sigmoid(logit))

    def positive_probabilities(
        self, embeddings: object
    ) -> NDArray[np.float64]:
        """Return P(cheer) for a non-empty ``(N, 1024)`` embedding matrix."""

        values = _validate_embedding(embeddings, expected_ndim=2)
        if values.shape[0] == 0:
            raise ModelArtifactError("embedding batch must not be empty")
        standardized = (values - self.scaler_mean) / self.scaler_scale
        logits = standardized @ self.lr_coef + self.lr_intercept
        probabilities = stable_sigmoid(logits)
        probabilities.setflags(write=False)
        return probabilities

    def predict_embedding(self, embedding: object) -> int:
        return int(self.positive_probability(embedding) >= self.threshold)

    def predict_embeddings(self, embeddings: object) -> NDArray[np.uint8]:
        labels = (self.positive_probabilities(embeddings) >= self.threshold).astype(
            np.uint8
        )
        labels.setflags(write=False)
        return labels


def _validated_float_vector(
    values: object, name: str, *, positive: bool
) -> NDArray[np.float64]:
    array = np.asarray(values)
    if array.shape != (YAMNET_EMBEDDING_SIZE,) or array.dtype != np.float64:
        raise ModelArtifactError(
            f"{name} must be float64 with shape ({YAMNET_EMBEDDING_SIZE},)"
        )
    if not np.isfinite(array).all():
        raise ModelArtifactError(f"{name} must contain only finite values")
    if positive and np.any(array <= 0):
        raise ModelArtifactError(f"{name} values must be greater than zero")
    result = np.array(array, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _validate_embedding(values: object, *, expected_ndim: int) -> NDArray[np.float64]:
    array = np.asarray(values)
    expected_shape = (
        f"({YAMNET_EMBEDDING_SIZE},)"
        if expected_ndim == 1
        else f"(N, {YAMNET_EMBEDDING_SIZE})"
    )
    if array.ndim != expected_ndim:
        raise ModelArtifactError(f"embeddings must have shape {expected_shape}")
    if array.shape[-1] != YAMNET_EMBEDDING_SIZE:
        raise ModelArtifactError(f"embeddings must have shape {expected_shape}")
    if not (
        np.issubdtype(array.dtype, np.floating)
        or np.issubdtype(array.dtype, np.integer)
    ) or np.issubdtype(array.dtype, np.bool_):
        raise ModelArtifactError("embeddings must contain numeric values")
    converted = np.asarray(array, dtype=np.float64)
    if not np.isfinite(converted).all():
        raise ModelArtifactError("embeddings must contain only finite values")
    return converted


def _validate_metadata(metadata: object) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise ModelArtifactError("metadata must be a JSON object")
    try:
        if metadata["model_id"] != BASELINE_ID:
            raise ModelArtifactError(f"model_id must be {BASELINE_ID!r}")
        if metadata["model_type"] != "logistic_regression":
            raise ModelArtifactError("model_type must be logistic_regression")
        feature = metadata["feature_extractor"]
        if feature["model_identifier"] != YAMNET_MODEL_HANDLE:
            raise ModelArtifactError("unexpected YAMNet model identifier")
        if feature["pooling"] != "mean":
            raise ModelArtifactError("feature pooling must be mean")
        if feature["embedding_dimension"] != YAMNET_EMBEDDING_SIZE:
            raise ModelArtifactError("embedding_dimension must be 1024")
        audio = metadata["audio"]
        planner = InferenceConfig()
        if audio["sample_rate_hz"] != YAMNET_SAMPLE_RATE_HZ:
            raise ModelArtifactError("audio sample_rate_hz must be 16000")
        for name, expected in (
            ("window_sec", planner.window_sec),
            ("hop_sec", planner.hop_sec),
            ("post_padding_sec", planner.post_padding_sec),
        ):
            if audio[name] != expected:
                raise ModelArtifactError(
                    f"audio {name} does not match frozen baseline"
                )
        classifier = metadata["classifier"]
        if classifier["preprocessing"] != "StandardScaler":
            raise ModelArtifactError("classifier preprocessing must be StandardScaler")
        if classifier["C"] != LOGISTIC_REGRESSION_C:
            raise ModelArtifactError("classifier C does not match frozen baseline")
        if classifier["solver"] != "lbfgs":
            raise ModelArtifactError("classifier solver must be lbfgs")
        if classifier["max_iter"] != LOGISTIC_REGRESSION_MAX_ITER:
            raise ModelArtifactError("classifier max_iter does not match baseline")
        if classifier["class_weight"] is not None:
            raise ModelArtifactError("classifier class_weight must be null")
        threshold = float(classifier["threshold"])
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ModelArtifactError("classifier threshold must be in [0, 1]")
        if threshold != PREDICTION_THRESHOLD:
            raise ModelArtifactError("classifier threshold must remain 0.5")
        training = metadata["training"]
        if not isinstance(training, dict):
            raise ModelArtifactError("training metadata must be an object")
        matches = training["matches"]
        if (
            not isinstance(matches, list)
            or not matches
            or any(not isinstance(match_id, str) or not match_id for match_id in matches)
            or len(set(matches)) != len(matches)
        ):
            raise ModelArtifactError("training matches must be unique non-empty strings")
        counts = {
            name: training[name]
            for name in ("sample_count", "positive_count", "negative_count")
        }
        if any(type(value) is not int or value < 0 for value in counts.values()):
            raise ModelArtifactError("training counts must be non-negative integers")
        if counts["sample_count"] <= 0 or counts["sample_count"] != (
            counts["positive_count"] + counts["negative_count"]
        ):
            raise ModelArtifactError("training class counts do not sum to sample_count")
        if counts["positive_count"] == 0 or counts["negative_count"] == 0:
            raise ModelArtifactError("training metadata must contain both classes")
        if training["converged"] is not True:
            raise ModelArtifactError("exported classifier must have converged")
        if type(training["iterations"]) is not int or training["iterations"] <= 0:
            raise ModelArtifactError("training iterations must be a positive integer")
        if training["scaler_fit_sample_count"] != counts["sample_count"]:
            raise ModelArtifactError("StandardScaler fit count must equal sample_count")
        for name in ("sklearn_version", "numpy_version", "trained_at"):
            if not isinstance(training[name], str) or not training[name]:
                raise ModelArtifactError(f"training {name} must be a non-empty string")
        datasets = training["datasets"]
        if not isinstance(datasets, list) or len(datasets) != len(matches):
            raise ModelArtifactError("training datasets must align with matches")
        dataset_sample_count = 0
        for expected_match, dataset in zip(matches, datasets, strict=True):
            if not isinstance(dataset, dict) or dataset["match_id"] != expected_match:
                raise ModelArtifactError("training dataset match_id order is invalid")
            path = dataset["path"]
            if (
                not isinstance(path, str)
                or not path
                or Path(path).is_absolute()
                or Path(path).drive
            ):
                raise ModelArtifactError("training dataset paths must be relative")
            _validate_sha256(dataset["sha256"], "training dataset sha256")
            samples = dataset["samples"]
            if type(samples) is not int or samples <= 0:
                raise ModelArtifactError("training dataset samples must be positive")
            dataset_sample_count += samples
        if dataset_sample_count != counts["sample_count"]:
            raise ModelArtifactError("training dataset samples do not sum to sample_count")
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelArtifactError(f"invalid metadata.json structure: {exc}") from exc
    _validate_sha256(metadata.get("model_sha256"), "model_sha256")
    return metadata


def _validate_sha256(value: object, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ModelArtifactError(f"{name} must be a lowercase SHA-256 digest")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ModelArtifactError(f"cannot read model.npz: {exc}") from exc
    return digest.hexdigest()
