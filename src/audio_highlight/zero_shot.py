"""Threshold-free RMS and YAMNet zero-shot baseline comparison."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score, roc_auc_score

from audio_highlight.audio import (
    AudioSlice,
    FFmpegAudioNormalizer,
    NormalizedAudioSource,
)
from audio_highlight.classifier import BASELINE_ID
from audio_highlight.dataset import FeatureDataset
from audio_highlight.yamnet import (
    YAMNET_CLASS_COUNT,
    YAMNET_MODEL_HANDLE,
    YamNetEmbeddingExtractor,
    validate_raw_scores,
)

EXPERIMENT_ID = "zero_shot_v1"
RMS_EPSILON = 1e-12
REQUIRED_AUDIOSET_CLASSES = ("Cheering", "Applause", "Crowd")
METHODS = (
    "rms",
    "yamnet_cheering",
    "yamnet_crowd_combo",
    "embedding_lr",
)


class ZeroShotComparisonError(ValueError):
    """Raised when zero-shot comparison inputs or identities are invalid."""


@dataclass(frozen=True, slots=True)
class AudioSetClass:
    name: str
    index: int


@dataclass(frozen=True, slots=True)
class NativeClassScores:
    cheering: float
    applause: float
    crowd: float

    @property
    def crowd_combo(self) -> float:
        return max(self.cheering, self.applause, self.crowd)


@dataclass(frozen=True, slots=True)
class SampleIdentity:
    match_id: str
    sample_rank: int
    segment_index: int
    window_index_in_segment: int
    start_sec: float
    end_sec: float


@dataclass(frozen=True, slots=True)
class SupervisedReference:
    identity: SampleIdentity
    true_label: int
    positive_probability: float


@dataclass(frozen=True, slots=True)
class BaselinePrediction:
    match_id: str
    sample_rank: int
    segment_index: int
    window_index_in_segment: int
    start_sec: float
    end_sec: float
    true_label: int
    log_rms_db: float
    yamnet_cheering_score: float
    yamnet_applause_score: float
    yamnet_crowd_score: float
    yamnet_crowd_combo_score: float
    supervised_lr_probability: float


@dataclass(frozen=True, slots=True)
class RankingMetrics:
    roc_auc: float
    average_precision: float


@dataclass(frozen=True, slots=True)
class BaselineComparisonResult:
    experiment_id: str
    development_matches: tuple[str, ...]
    external_match: str
    predictions: tuple[BaselinePrediction, ...]
    metrics: dict[str, Any]
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class BaselineComparisonArtifactPaths:
    predictions_csv: Path
    metrics_json: Path
    summary_csv: Path
    metadata_json: Path


class NativeScoreProvider(Protocol):
    model_identifier: str
    class_map_path: Path
    classes: tuple[AudioSetClass, ...]

    def score(self, audio: AudioSlice) -> NativeClassScores:
        ...


def resolve_audioset_classes(
    class_map_path: str | Path,
    required_names: Sequence[str] = REQUIRED_AUDIOSET_CLASSES,
) -> tuple[AudioSetClass, ...]:
    """Resolve exact display names from the official class-map CSV."""

    path = Path(class_map_path)
    resolved: dict[str, int] = {}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            required_fields = {"index", "display_name"}
            missing_fields = required_fields - set(reader.fieldnames or ())
            if missing_fields:
                raise ZeroShotComparisonError(
                    "AudioSet class map is missing fields: "
                    + ", ".join(sorted(missing_fields))
                )
            for row_number, row in enumerate(reader, start=2):
                name = row["display_name"]
                if name not in required_names:
                    continue
                if name in resolved:
                    raise ZeroShotComparisonError(
                        f"AudioSet class map contains duplicate exact class {name!r}"
                    )
                try:
                    index = int(row["index"])
                except (TypeError, ValueError) as exc:
                    raise ZeroShotComparisonError(
                        f"AudioSet class map row {row_number}: invalid index"
                    ) from exc
                if not 0 <= index < YAMNET_CLASS_COUNT:
                    raise ZeroShotComparisonError(
                        f"AudioSet class {name!r} index is outside YAMNet output"
                    )
                resolved[name] = index
    except OSError:
        raise
    missing_names = [name for name in required_names if name not in resolved]
    if missing_names:
        raise ZeroShotComparisonError(
            "AudioSet class map is missing exact classes: " + ", ".join(missing_names)
        )
    return tuple(AudioSetClass(name, resolved[name]) for name in required_names)


def aggregate_mean_class_scores(
    patch_scores: object,
    classes: Sequence[AudioSetClass],
) -> NativeClassScores:
    """Mean-pool patches for the three fixed exact AudioSet classes."""

    scores = validate_raw_scores(patch_scores)
    indexed = {item.name: item.index for item in classes}
    missing = [name for name in REQUIRED_AUDIOSET_CLASSES if name not in indexed]
    if missing:
        raise ZeroShotComparisonError(
            "resolved classes are incomplete: " + ", ".join(missing)
        )
    means = np.mean(scores, axis=0, dtype=np.float64)
    return NativeClassScores(
        cheering=float(means[indexed["Cheering"]]),
        applause=float(means[indexed["Applause"]]),
        crowd=float(means[indexed["Crowd"]]),
    )


def waveform_log_rms_db(
    samples: object,
    *,
    epsilon: float = RMS_EPSILON,
) -> float:
    """Return ``20*log10(sqrt(mean(x^2)) + epsilon)`` in float64."""

    waveform = np.asarray(samples)
    if (
        waveform.ndim != 1
        or waveform.size == 0
        or not np.issubdtype(waveform.dtype, np.floating)
        or not np.isfinite(waveform).all()
    ):
        raise ZeroShotComparisonError(
            "RMS requires a non-empty finite floating-point waveform"
        )
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ZeroShotComparisonError("RMS epsilon must be positive and finite")
    values = np.asarray(waveform, dtype=np.float64)
    rms = math.sqrt(float(np.mean(values * values)))
    return 20.0 * math.log10(rms + epsilon)


class YamNetZeroShotScoreExtractor:
    """Load YAMNet once and expose only fixed native-class mean scores."""

    def __init__(
        self,
        extractor: YamNetEmbeddingExtractor | None = None,
        *,
        model_identifier: str = YAMNET_MODEL_HANDLE,
    ) -> None:
        self._extractor = extractor or YamNetEmbeddingExtractor(
            model_handle=model_identifier
        )
        self.model_identifier = model_identifier
        self.class_map_path = self._extractor.class_map_path()
        self.classes = resolve_audioset_classes(self.class_map_path)

    def score(self, audio: AudioSlice) -> NativeClassScores:
        patch_scores, _ = self._extractor.extract_outputs(audio)
        return aggregate_mean_class_scores(patch_scores, self.classes)


def load_supervised_references(
    path: str | Path,
    *,
    match_field: str,
) -> tuple[SupervisedReference, ...]:
    """Load frozen probabilities with complete identity fields, never by row order."""

    required_fields = {
        match_field,
        "sample_rank",
        "segment_index",
        "window_index_in_segment",
        "start_sec",
        "end_sec",
        "true_label",
        "positive_probability",
    }
    records: list[SupervisedReference] = []
    seen: set[SampleIdentity] = set()
    try:
        with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            missing = required_fields - set(reader.fieldnames or ())
            if missing:
                raise ZeroShotComparisonError(
                    "supervised predictions are missing fields: "
                    + ", ".join(sorted(missing))
                )
            for row_number, row in enumerate(reader, start=2):
                try:
                    identity = SampleIdentity(
                        match_id=row[match_field].strip(),
                        sample_rank=int(row["sample_rank"]),
                        segment_index=int(row["segment_index"]),
                        window_index_in_segment=int(row["window_index_in_segment"]),
                        start_sec=float(row["start_sec"]),
                        end_sec=float(row["end_sec"]),
                    )
                    true_label = int(row["true_label"])
                    probability = float(row["positive_probability"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ZeroShotComparisonError(
                        f"supervised predictions row {row_number}: invalid value"
                    ) from exc
                if not identity.match_id:
                    raise ZeroShotComparisonError("supervised prediction match_id is empty")
                if true_label not in {0, 1}:
                    raise ZeroShotComparisonError("supervised labels must be binary")
                if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                    raise ZeroShotComparisonError(
                        "supervised probabilities must be in [0, 1]"
                    )
                if identity in seen:
                    raise ZeroShotComparisonError(
                        f"duplicate supervised sample identity: {identity}"
                    )
                seen.add(identity)
                records.append(SupervisedReference(identity, true_label, probability))
    except OSError:
        raise
    if not records:
        raise ZeroShotComparisonError("supervised predictions contain no rows")
    return tuple(records)


def compare_zero_shot_baselines(
    development_matches: Sequence[str],
    external_match: str,
    *,
    artifact_root: str | Path = "artifacts",
    local_data_root: str | Path = "local_data",
    development_predictions_path: str | Path = (
        "artifacts/cross_match/evaluation/cross_match_predictions.csv"
    ),
    external_predictions_path: str | Path | None = None,
    model_path: str | Path = "artifacts/models/yamnet_mean_lr_v1/model.npz",
    score_provider: NativeScoreProvider | None = None,
) -> BaselineComparisonResult:
    """Score identical canonical windows and compare threshold-free discrimination."""

    development = tuple(development_matches)
    if (
        not development
        or any(not match_id for match_id in development)
        or len(set(development)) != len(development)
    ):
        raise ZeroShotComparisonError(
            "development matches must be unique non-empty identifiers"
        )
    if not external_match or external_match in development:
        raise ZeroShotComparisonError(
            "external match must be non-empty and excluded from development"
        )
    artifacts = Path(artifact_root)
    local_data = Path(local_data_root)
    development_predictions = Path(development_predictions_path)
    external_predictions = Path(
        external_predictions_path
        or artifacts / external_match / "external_validation" / "predictions.csv"
    )
    frozen_model = Path(model_path)
    frozen_model_metadata = frozen_model.with_name("metadata.json")
    _validate_frozen_model_metadata(
        frozen_model_metadata,
        model_path=frozen_model,
        external_match=external_match,
    )
    provider = score_provider or YamNetZeroShotScoreExtractor()
    development_references = load_supervised_references(
        development_predictions, match_field="test_match_id"
    )
    external_references = load_supervised_references(
        external_predictions, match_field="match_id"
    )
    references = _index_references(
        development_references,
        external_references,
        development_matches=development,
        external_match=external_match,
    )

    predictions: list[BaselinePrediction] = []
    dataset_metadata: list[dict[str, Any]] = []
    for match_id in (*development, external_match):
        feature_path = artifacts / match_id / "features" / "features.npz"
        labels_path = artifacts / match_id / "labeling" / "labels.csv"
        manifest_path = artifacts / match_id / "labeling" / "sample_manifest.json"
        audio_cache_path = artifacts / match_id / "audio" / "audio.f32le"
        dataset = FeatureDataset.load(feature_path)
        if dataset.match_id != match_id:
            raise ZeroShotComparisonError(
                f"feature match_id {dataset.match_id!r} does not match {match_id!r}"
            )
        if dataset.model_identifier != provider.model_identifier:
            raise ZeroShotComparisonError(
                f"feature YAMNet identifier differs for {match_id!r}"
            )
        if dataset.window_sec != 3.0:
            raise ZeroShotComparisonError(
                f"feature windows for {match_id!r} must be 3.0 seconds"
            )
        _validate_dataset_reference_alignment(dataset, references[match_id])
        if not audio_cache_path.is_file():
            video_path = local_data / match_id / "match.mp4"
            source = FFmpegAudioNormalizer().normalize(video_path, audio_cache_path)
        else:
            source = NormalizedAudioSource(audio_cache_path)
        reference_map = {
            reference.identity: reference for reference in references[match_id]
        }
        try:
            for index in range(dataset.labels.size):
                identity = _dataset_identity(dataset, index)
                audio = source.slice_absolute(
                    float(dataset.start_secs[index]), float(dataset.end_secs[index])
                )
                native = provider.score(audio)
                reference = reference_map[identity]
                predictions.append(
                    BaselinePrediction(
                        match_id=match_id,
                        sample_rank=identity.sample_rank,
                        segment_index=identity.segment_index,
                        window_index_in_segment=identity.window_index_in_segment,
                        start_sec=identity.start_sec,
                        end_sec=identity.end_sec,
                        true_label=int(dataset.labels[index]),
                        log_rms_db=waveform_log_rms_db(audio.samples),
                        yamnet_cheering_score=native.cheering,
                        yamnet_applause_score=native.applause,
                        yamnet_crowd_score=native.crowd,
                        yamnet_crowd_combo_score=native.crowd_combo,
                        supervised_lr_probability=reference.positive_probability,
                    )
                )
        finally:
            source.close()
        dataset_metadata.append(
            {
                "match_id": match_id,
                "feature": _artifact_record(feature_path),
                "labels": _artifact_record(labels_path),
                "manifest": _artifact_record(manifest_path),
                "audio_cache": _artifact_record(audio_cache_path),
            }
        )

    prediction_tuple = tuple(predictions)
    metrics = _comparison_metrics(prediction_tuple, development, external_match)
    metadata = {
        "experiment_id": EXPERIMENT_ID,
        "yamnet": {
            "model_identifier": provider.model_identifier,
            "patch_aggregation": "mean",
            "class_map": _artifact_record(provider.class_map_path),
            "classes": [asdict(item) for item in provider.classes],
        },
        "crowd_combination_rule": "max_of_mean_class_scores",
        "rms": {
            "waveform_source": "canonical_16khz_mono_float32_audio_cache",
            "definition": "20*log10(sqrt(mean(x^2)) + epsilon)",
            "epsilon": RMS_EPSILON,
        },
        "evaluation_matches": {
            "development": list(development),
            "external": external_match,
        },
        "supervised_reference": {
            "development": "existing_leave_one_match_out_oof_predictions",
            "external": "existing_frozen_final_detector_predictions",
        },
        "input_artifacts": {
            "development_predictions": _artifact_record(development_predictions),
            "external_predictions": _artifact_record(external_predictions),
            "frozen_model": _artifact_record(frozen_model),
            "frozen_model_metadata": _artifact_record(frozen_model_metadata),
            "datasets": dataset_metadata,
        },
    }
    return BaselineComparisonResult(
        experiment_id=EXPERIMENT_ID,
        development_matches=development,
        external_match=external_match,
        predictions=prediction_tuple,
        metrics=metrics,
        metadata=metadata,
    )


def write_baseline_comparison_artifacts(
    result: BaselineComparisonResult,
    output_dir: str | Path,
) -> BaselineComparisonArtifactPaths:
    """Atomically serialize deterministic diagnostic CSV and JSON artifacts."""

    output = Path(output_dir)
    _validate_output_path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = BaselineComparisonArtifactPaths(
        predictions_csv=output / "predictions.csv",
        metrics_json=output / "metrics.json",
        summary_csv=output / "summary.csv",
        metadata_json=output / "metadata.json",
    )
    _write_predictions(result.predictions, paths.predictions_csv)
    _write_json(result.metrics, paths.metrics_json)
    _write_summary(result, paths.summary_csv)
    _write_json(result.metadata, paths.metadata_json)
    return paths


def _index_references(
    development_records: Sequence[SupervisedReference],
    external_records: Sequence[SupervisedReference],
    *,
    development_matches: Sequence[str],
    external_match: str,
) -> dict[str, tuple[SupervisedReference, ...]]:
    development_set = set(development_matches)
    grouped: dict[str, list[SupervisedReference]] = {
        match_id: [] for match_id in (*development_matches, external_match)
    }
    for record in development_records:
        if record.identity.match_id in development_set:
            grouped[record.identity.match_id].append(record)
        elif record.identity.match_id == external_match:
            raise ZeroShotComparisonError(
                "external match appears in development OOF predictions"
            )
    for record in external_records:
        if record.identity.match_id != external_match:
            raise ZeroShotComparisonError(
                "external prediction artifact contains an unexpected match_id"
            )
        grouped[external_match].append(record)
    if any(not values for values in grouped.values()):
        missing = [match_id for match_id, values in grouped.items() if not values]
        raise ZeroShotComparisonError(
            "supervised references missing matches: " + ", ".join(missing)
        )
    return {match_id: tuple(values) for match_id, values in grouped.items()}


def _validate_dataset_reference_alignment(
    dataset: FeatureDataset,
    references: Sequence[SupervisedReference],
) -> None:
    dataset_entries = {
        _dataset_identity(dataset, index): int(dataset.labels[index])
        for index in range(dataset.labels.size)
    }
    if len(dataset_entries) != dataset.labels.size:
        raise ZeroShotComparisonError(
            f"feature dataset {dataset.match_id!r} has duplicate sample identities"
        )
    reference_entries = {
        reference.identity: reference.true_label for reference in references
    }
    if len(reference_entries) != len(references):
        raise ZeroShotComparisonError(
            f"supervised references {dataset.match_id!r} have duplicate identities"
        )
    if dataset_entries != reference_entries:
        raise ZeroShotComparisonError(
            f"sample identity or label mismatch for {dataset.match_id!r}"
        )


def _dataset_identity(dataset: FeatureDataset, index: int) -> SampleIdentity:
    return SampleIdentity(
        match_id=dataset.match_id,
        sample_rank=int(dataset.sample_ranks[index]),
        segment_index=int(dataset.segment_indices[index]),
        window_index_in_segment=int(dataset.window_indices[index]),
        start_sec=float(dataset.start_secs[index]),
        end_sec=float(dataset.end_secs[index]),
    )


def _comparison_metrics(
    predictions: Sequence[BaselinePrediction],
    development_matches: Sequence[str],
    external_match: str,
) -> dict[str, Any]:
    grouped = {
        match_id: tuple(item for item in predictions if item.match_id == match_id)
        for match_id in (*development_matches, external_match)
    }
    development: dict[str, Any] = {}
    for match_id in development_matches:
        development[match_id] = _match_metrics(grouped[match_id])
    development["macro_mean"] = {
        method: {
            "roc_auc": float(
                np.mean(
                    [development[match_id]["methods"][method]["roc_auc"] for match_id in development_matches]
                )
            ),
            "average_precision": float(
                np.mean(
                    [
                        development[match_id]["methods"][method]["average_precision"]
                        for match_id in development_matches
                    ]
                )
            ),
        }
        for method in METHODS
    }
    return {
        "experiment_id": EXPERIMENT_ID,
        "development": development,
        "external": {external_match: _match_metrics(grouped[external_match])},
    }


def _match_metrics(predictions: Sequence[BaselinePrediction]) -> dict[str, Any]:
    if not predictions:
        raise ZeroShotComparisonError("cannot compute metrics for an empty match")
    labels = np.asarray([item.true_label for item in predictions], dtype=np.uint8)
    if not np.isin(labels, [0, 1]).all() or np.unique(labels).size != 2:
        raise ZeroShotComparisonError("comparison match must contain both labels")
    method_scores = {
        "rms": [item.log_rms_db for item in predictions],
        "yamnet_cheering": [item.yamnet_cheering_score for item in predictions],
        "yamnet_crowd_combo": [
            item.yamnet_crowd_combo_score for item in predictions
        ],
        "embedding_lr": [item.supervised_lr_probability for item in predictions],
    }
    return {
        "sample_count": int(labels.size),
        "positive_count": int(np.count_nonzero(labels == 1)),
        "negative_count": int(np.count_nonzero(labels == 0)),
        "prevalence": float(np.mean(labels)),
        "methods": {
            method: asdict(_ranking_metrics(labels, np.asarray(scores)))
            for method, scores in method_scores.items()
        },
    }


def _ranking_metrics(
    labels: NDArray[np.uint8], scores: NDArray[np.floating[Any]]
) -> RankingMetrics:
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != labels.shape or not np.isfinite(values).all():
        raise ZeroShotComparisonError("comparison scores must be aligned and finite")
    return RankingMetrics(
        roc_auc=float(roc_auc_score(labels, values)),
        average_precision=float(average_precision_score(labels, values)),
    )


def _artifact_record(path: Path) -> dict[str, str]:
    return {"path": _portable_path(path), "sha256": _sha256(path)}


def _validate_frozen_model_metadata(
    metadata_path: Path,
    *,
    model_path: Path,
    external_match: str,
) -> None:
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["model_id"] != BASELINE_ID:
            raise ZeroShotComparisonError("frozen model_id does not match baseline")
        training_matches = metadata["training"]["matches"]
        if external_match in training_matches:
            raise ZeroShotComparisonError(
                "external match appears in frozen model training metadata"
            )
        expected_hash = metadata["model_sha256"]
    except (OSError, json.JSONDecodeError):
        raise
    except (KeyError, TypeError) as exc:
        raise ZeroShotComparisonError("invalid frozen model metadata") from exc
    if expected_hash != _sha256(model_path):
        raise ZeroShotComparisonError("frozen model SHA-256 does not match metadata")


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.name


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ZeroShotComparisonError(f"cannot hash input artifact {path}: {exc}") from exc
    return digest.hexdigest()


def _validate_output_path(path: Path) -> None:
    parts = tuple(part.lower() for part in path.resolve(strict=False).parts)
    forbidden_pairs = {
        ("artifacts", "baselines"),
        ("artifacts", "models"),
        ("artifacts", "cross_match"),
    }
    if any(pair in forbidden_pairs for pair in zip(parts, parts[1:])) or (
        "external_validation" in parts
    ):
        raise ZeroShotComparisonError(
            "baseline comparison output must be separate from frozen artifacts"
        )


def _write_predictions(
    predictions: Sequence[BaselinePrediction], path: Path
) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file, fieldnames=list(BaselinePrediction.__dataclass_fields__)
            )
            writer.writeheader()
            for prediction in predictions:
                writer.writerow(asdict(prediction))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(value: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(value, file, ensure_ascii=False, indent=2)
            file.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_summary(result: BaselineComparisonResult, path: Path) -> None:
    fieldnames = ("scope", "match_id", "method", "roc_auc", "average_precision")
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            development = result.metrics["development"]
            for match_id in result.development_matches:
                for method in METHODS:
                    writer.writerow(
                        {
                            "scope": "development",
                            "match_id": match_id,
                            "method": method,
                            **development[match_id]["methods"][method],
                        }
                    )
            for method in METHODS:
                writer.writerow(
                    {
                        "scope": "development_macro",
                        "match_id": "macro_mean",
                        "method": method,
                        **development["macro_mean"][method],
                    }
                )
            external = result.metrics["external"][result.external_match]
            for method in METHODS:
                writer.writerow(
                    {
                        "scope": "external",
                        "match_id": result.external_match,
                        "method": method,
                        **external["methods"][method],
                    }
                )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
