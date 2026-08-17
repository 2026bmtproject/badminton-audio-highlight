"""Command-line entry points for the canonical audio-highlight workflow."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

import numpy as np

from audio_highlight.contracts import ContractError, load_segments_artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="audio-highlight")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate-segments",
        help="validate an upstream match_segmentation segments.json artifact",
    )
    validate.add_argument("path")

    smoke = subparsers.add_parser(
        "smoke-test-yamnet",
        help="load YAMNet and embed a synthetic three-second sine wave",
    )
    smoke.add_argument(
        "--model-handle",
        default="https://tfhub.dev/google/yamnet/1",
    )

    features = subparsers.add_parser(
        "build-features",
        help="build canonical YAMNet features from completed blind labels",
    )
    features.add_argument("--match-id", required=True)
    features.add_argument("--video", type=Path)
    features.add_argument("--labels", type=Path)
    features.add_argument("--manifest", type=Path)
    features.add_argument("--audio-cache", type=Path)
    features.add_argument("--output", type=Path)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="run leave-one-match-out Logistic Regression evaluation",
    )
    evaluate.add_argument(
        "features",
        nargs="+",
        help="canonical per-match NPZ feature datasets",
    )
    evaluate.add_argument(
        "--output-dir",
        default="artifacts/evaluation",
        help="evaluation artifact directory",
    )
    calibration = subparsers.add_parser(
        "diagnose-calibration",
        help="describe calibration and score distributions from existing OOF predictions",
    )
    calibration.add_argument(
        "--predictions",
        default="artifacts/cross_match/evaluation/cross_match_predictions.csv",
        help="existing cross-match OOF predictions CSV",
    )
    calibration.add_argument(
        "--output-dir",
        default="artifacts/cross_match/evaluation/calibration",
        help="calibration diagnostic artifact directory",
    )
    train_model = subparsers.add_parser(
        "train-model",
        help="fit and export the frozen detector from all supplied clean features",
    )
    train_model.add_argument(
        "features",
        nargs="*",
        help="canonical per-match NPZ feature datasets",
    )
    train_model.add_argument(
        "--matches",
        nargs="+",
        help="match IDs resolved as artifacts/<match_id>/features/features.npz",
    )
    train_model.add_argument(
        "--baseline-id",
        default="yamnet_mean_lr_v1",
        help="frozen baseline identity",
    )
    train_model.add_argument(
        "--output-dir",
        type=Path,
        help="model artifact directory (default: artifacts/models/<baseline-id>)",
    )
    external = subparsers.add_parser(
        "evaluate-external",
        help="evaluate a frozen exported detector on one untouched match",
    )
    external.add_argument("--match-id", required=True)
    external.add_argument(
        "--features",
        type=Path,
        help="feature NPZ (default: artifacts/<match-id>/features/features.npz)",
    )
    external.add_argument(
        "--model-dir",
        type=Path,
        default=Path("artifacts/models/yamnet_mean_lr_v1"),
        help="directory containing model.npz and metadata.json",
    )
    external.add_argument(
        "--output-dir",
        type=Path,
        help="output directory (default: artifacts/<match-id>/external_validation)",
    )
    return parser


def _match_paths(args: argparse.Namespace) -> dict[str, Path]:
    local_root = Path("local_data") / args.match_id
    artifact_root = Path("artifacts") / args.match_id
    return {
        "video": args.video or local_root / "match.mp4",
        "labels": args.labels or artifact_root / "labeling" / "labels.csv",
        "manifest": (
            args.manifest or artifact_root / "labeling" / "sample_manifest.json"
        ),
        "audio_cache": (
            args.audio_cache or artifact_root / "audio" / "audio.f32le"
        ),
        "output": args.output or artifact_root / "features" / "features.npz",
    }


def run(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate-segments":
        try:
            artifact = load_segments_artifact(args.path)
        except (ContractError, OSError) as exc:
            print(f"invalid segments artifact: {exc}")
            return 2
        print(
            f"valid segments artifact: {len(artifact.segments)} segments, "
            f"fps={artifact.fps:g}"
        )
        return 0

    if args.command == "smoke-test-yamnet":
        from audio_highlight.audio import AudioWindow
        from audio_highlight.yamnet import YamNetEmbeddingExtractor, mean_pool_embeddings

        sample_rate_hz = 16_000
        time = np.arange(3 * sample_rate_hz, dtype=np.float32) / sample_rate_hz
        samples = (0.1 * np.sin(2.0 * np.pi * 440.0 * time)).astype(np.float32)
        samples.setflags(write=False)
        window = AudioWindow(0, 0.0, 3.0, sample_rate_hz, 1, samples)
        extractor = YamNetEmbeddingExtractor(model_handle=args.model_handle)
        raw = extractor.extract_raw(window)
        pooled = mean_pool_embeddings(raw)
        print(f"raw_embeddings_shape={raw.shape}")
        print(f"pooled_embedding_shape={pooled.shape}")
        return 0

    if args.command == "build-features":
        from audio_highlight.audio import AudioError
        from audio_highlight.dataset import DatasetError, build_feature_dataset
        from audio_highlight.labeling import LabelingError
        from audio_highlight.yamnet import YamNetError

        paths = _match_paths(args)
        try:
            result = build_feature_dataset(
                video_path=paths["video"],
                labels_path=paths["labels"],
                manifest_path=paths["manifest"],
                audio_cache_path=paths["audio_cache"],
                output_path=paths["output"],
            )
        except (AudioError, DatasetError, LabelingError, YamNetError, OSError) as exc:
            print(f"feature build failed: {exc}")
            return 2
        print(f"reviewed={result.reviewed}")
        print(f"binary_included={result.binary_included}")
        print(f"ambiguous_excluded={result.ambiguous_excluded}")
        print(f"feature_shape={result.dataset.embeddings.shape}")
        print(f"output_path={result.output_path}")
        return 0

    if args.command == "evaluate":
        from audio_highlight.dataset import DatasetError
        from audio_highlight.evaluation import (
            EvaluationError,
            evaluate_cross_match,
            write_evaluation_artifacts,
        )

        try:
            result = evaluate_cross_match(args.features)
            artifacts = write_evaluation_artifacts(result, args.output_dir)
        except (DatasetError, EvaluationError, OSError) as exc:
            print(f"evaluation failed: {exc}")
            return 2
        for fold in result.folds:
            metrics = fold.metrics
            print(f"Fold: {', '.join(fold.train_matches)} -> {fold.test_match}")
            print(f"train_samples={fold.train_samples}")
            print(f"test_samples={fold.test_samples}")
            print(f"accuracy={metrics.accuracy:.6f}")
            print(f"precision={metrics.precision:.6f}")
            print(f"recall={metrics.recall:.6f}")
            print(f"f1={metrics.f1:.6f}")
            print(f"converged={fold.converged} iterations={fold.iterations}")
        print("Macro mean")
        for name, value in result.macro_mean.items():
            print(f"{name}={'undefined' if value is None else f'{value:.6f}'}")
        print(f"predictions_csv={artifacts.predictions_csv}")
        print(f"metrics_json={artifacts.metrics_json}")
        return 0

    if args.command == "diagnose-calibration":
        from audio_highlight.baseline import write_baseline_metadata
        from audio_highlight.calibration import (
            CalibrationError,
            diagnose_calibration,
            load_predictions,
            write_calibration_artifacts,
        )

        try:
            records = load_predictions(args.predictions)
            result = diagnose_calibration(records)
            artifacts = write_calibration_artifacts(result, args.output_dir)
            baseline_path = write_baseline_metadata()
        except (CalibrationError, OSError) as exc:
            print(f"calibration diagnostic failed: {exc}")
            return 2
        for item in result.matches:
            print(f"match={item.match_id}")
            print(f"sample_count={item.sample_count}")
            print(f"prevalence={item.prevalence:.6f}")
            print(f"predicted_positive_rate={item.predicted_positive_rate:.6f}")
            print(f"brier_score={item.brier_score:.6f}")
            print(f"log_loss={item.log_loss:.6f}")
            print(f"ece={item.ece:.6f}")
        print(f"baseline_metadata={baseline_path}")
        print(f"calibration_metrics={artifacts.metrics_json}")
        print(f"calibration_summary={artifacts.summary_csv}")
        print(f"combined_reliability={artifacts.combined_reliability_plot}")
        return 0

    if args.command == "train-model":
        from audio_highlight.classifier import ModelArtifactError
        from audio_highlight.dataset import DatasetError
        from audio_highlight.model_export import (
            ModelTrainingError,
            train_and_export_model,
        )

        if bool(args.features) == bool(args.matches):
            print(
                "model training failed: supply either explicit feature paths or "
                "--matches, but not both"
            )
            return 2
        feature_paths = (
            args.features
            if args.features
            else [
                Path("artifacts") / match_id / "features" / "features.npz"
                for match_id in args.matches
            ]
        )
        output_dir = args.output_dir or (
            Path("artifacts") / "models" / args.baseline_id
        )
        try:
            result = train_and_export_model(
                feature_paths,
                baseline_id=args.baseline_id,
                output_dir=output_dir,
            )
        except (
            DatasetError,
            ModelArtifactError,
            ModelTrainingError,
            OSError,
        ) as exc:
            print(f"model training failed: {exc}")
            return 2
        print(f"baseline_id={result.baseline_id}")
        print(f"training_matches={','.join(result.training_matches)}")
        print(f"sample_count={result.sample_count}")
        print(f"positive_count={result.positive_count}")
        print(f"negative_count={result.negative_count}")
        print(f"embedding_dimension={result.embedding_dimension}")
        print(f"converged={result.converged}")
        print(f"iterations={result.iterations}")
        print(f"model_path={result.model_path}")
        print(f"metadata_path={result.metadata_path}")
        print(f"model_sha256={result.model_sha256}")
        print(f"model_size_bytes={result.model_size_bytes}")
        print(f"max_probability_difference={result.max_probability_difference:.17g}")
        print(f"binary_predictions_equal={result.binary_predictions_equal}")
        return 0

    if args.command == "evaluate-external":
        from audio_highlight.calibration import CalibrationError
        from audio_highlight.classifier import ModelArtifactError
        from audio_highlight.dataset import DatasetError
        from audio_highlight.evaluation import EvaluationError
        from audio_highlight.external_validation import (
            ExternalValidationError,
            evaluate_external_match,
            write_external_validation_artifacts,
        )

        features_path = args.features or (
            Path("artifacts") / args.match_id / "features" / "features.npz"
        )
        output_dir = args.output_dir or (
            Path("artifacts") / args.match_id / "external_validation"
        )
        try:
            result = evaluate_external_match(
                features_path,
                args.model_dir,
                expected_match_id=args.match_id,
            )
            artifacts = write_external_validation_artifacts(result, output_dir)
        except (
            CalibrationError,
            DatasetError,
            EvaluationError,
            ExternalValidationError,
            ModelArtifactError,
            OSError,
        ) as exc:
            print(f"external validation failed: {exc}")
            return 2
        metrics = result.metrics
        print(f"validation_type={result.validation_type}")
        print(f"baseline_id={result.baseline_id}")
        print(f"match_id={result.match_id}")
        print(f"sample_count={result.sample_count}")
        print(f"positive_count={result.positive_count}")
        print(f"negative_count={result.negative_count}")
        print(f"threshold={result.threshold:g}")
        print(f"accuracy={metrics.accuracy:.6f}")
        print(f"precision={metrics.precision:.6f}")
        print(f"recall={metrics.recall:.6f}")
        print(f"f1={metrics.f1:.6f}")
        print(
            "roc_auc="
            + ("undefined" if metrics.roc_auc is None else f"{metrics.roc_auc:.6f}")
        )
        print(
            "average_precision="
            + (
                "undefined"
                if metrics.average_precision is None
                else f"{metrics.average_precision:.6f}"
            )
        )
        print(f"brier_score={result.brier_score:.6f}")
        print(f"log_loss={result.log_loss:.6f}")
        print(f"ece={result.ece:.6f}")
        print(f"observed_prevalence={result.observed_prevalence:.6f}")
        print(f"predicted_positive_rate={result.predicted_positive_rate:.6f}")
        print(
            "positive_probability_median="
            f"{result.positive_probability_summary.median:.6f}"
        )
        print(
            "negative_probability_median="
            f"{result.negative_probability_summary.median:.6f}"
        )
        print(f"tn={metrics.tn} fp={metrics.fp} fn={metrics.fn} tp={metrics.tp}")
        print(f"model_sha256={result.model_sha256}")
        print(f"feature_sha256={result.feature_sha256}")
        print(f"predictions_csv={artifacts.predictions_csv}")
        print(f"metrics_json={artifacts.metrics_json}")
        return 0

    raise AssertionError(f"unhandled command: {args.command}")


def main() -> NoReturn:
    raise SystemExit(run())
