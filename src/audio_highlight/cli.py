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
    raise AssertionError(f"unhandled command: {args.command}")


def main() -> NoReturn:
    raise SystemExit(run())
