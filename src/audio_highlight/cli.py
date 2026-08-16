"""Command-line entry point for scaffold-level contract validation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import NoReturn

from audio_highlight.contracts import ContractError, load_segments_artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="audio-highlight")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser(
        "validate-segments",
        help="validate an upstream match_segmentation segments.json artifact",
    )
    validate.add_argument("path")
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
    raise AssertionError(f"unhandled command: {args.command}")


def main() -> NoReturn:
    raise SystemExit(run())
