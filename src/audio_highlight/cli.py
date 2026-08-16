"""Command-line entry point for scaffold-level contract validation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
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
        help="manually load real YAMNet and embed a synthetic 3-second sine wave",
    )
    smoke.add_argument(
        "--model-handle",
        default="https://tfhub.dev/google/yamnet/1",
        help="TensorFlow Hub URL or local SavedModel path",
    )
    features = subparsers.add_parser(
        "build-features",
        help="build one match's reviewed YAMNet feature dataset",
    )
    features.add_argument("--match-id", required=True)
    features.add_argument("--video", required=True)
    features.add_argument("--labels", required=True)
    features.add_argument("--output", required=True)
    features.add_argument(
        "--audio-cache",
        help="optional rebuildable 16 kHz mono float32 cache path",
    )
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate-segments":
        try:
            artifact = load_segments_artifact(args.path)
        except (ContractError, OSError) as exc:
            print(f"invalid segments artifact: {exc}")
            return 2
        print(f"valid segments artifact: {len(artifact.segments)} segments, fps={artifact.fps:g}")
        return 0
    if args.command == "smoke-test-yamnet":
        from audio_highlight.audio import AudioWindow
        from audio_highlight.yamnet import (
            YamNetEmbeddingExtractor,
            mean_pool_embeddings,
        )

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
        from audio_highlight.training import TrainingDataError, build_feature_dataset
        from audio_highlight.yamnet import YamNetError

        try:
            result = build_feature_dataset(
                match_id=args.match_id,
                video_path=args.video,
                labels_path=args.labels,
                output_path=args.output,
                audio_cache_path=args.audio_cache,
            )
        except (AudioError, TrainingDataError, YamNetError, OSError) as exc:
            print(f"feature build failed: {exc}")
            return 2
        summary = result.labels
        print(f"total_rows={summary.total_rows}")
        print(f"reviewed_rows={summary.reviewed_rows}")
        print(f"skipped_unreviewed_rows={summary.skipped_unreviewed_rows}")
        print(f"cheer_0={summary.negative_rows}")
        print(f"cheer_1={summary.positive_rows}")
        print(f"feature_shape={result.dataset.embeddings.shape}")
        print(f"output_path={result.output_path}")
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


def main() -> NoReturn:
    raise SystemExit(run())
