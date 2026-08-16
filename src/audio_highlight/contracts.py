"""Typed boundary objects for upstream segmentation and highlight artifacts."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class ContractError(ValueError):
    """Raised when an artifact does not satisfy the expected JSON contract."""


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{field} must be finite")
    return result


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{field} must be an integer")
    return value


@dataclass(frozen=True, slots=True)
class Segment:
    """One unchanged record from upstream ``segments.json``."""

    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    duration_sec: float

    def __post_init__(self) -> None:
        if self.start_frame < 0:
            raise ContractError("start_frame must be non-negative")
        if self.end_frame < self.start_frame:
            raise ContractError("end_frame must not precede start_frame")
        if self.start_sec < 0:
            raise ContractError("start_sec must be non-negative")
        if self.end_sec < self.start_sec:
            raise ContractError("end_sec must not precede start_sec")
        if self.duration_sec < 0:
            raise ContractError("duration_sec must be non-negative")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Segment:
        try:
            return cls(
                start_frame=_integer(value["start_frame"], "start_frame"),
                end_frame=_integer(value["end_frame"], "end_frame"),
                start_sec=_finite_number(value["start_sec"], "start_sec"),
                end_sec=_finite_number(value["end_sec"], "end_sec"),
                duration_sec=_finite_number(value["duration_sec"], "duration_sec"),
            )
        except KeyError as exc:
            raise ContractError(f"segment is missing {exc.args[0]!r}") from exc


@dataclass(frozen=True, slots=True)
class IndexedSegment:
    """A segment paired with its authoritative array-position index."""

    segment_index: int
    segment: Segment


@dataclass(frozen=True, slots=True)
class SegmentsArtifact:
    """Parsed upstream ``segments.json`` envelope."""

    segments: tuple[Segment, ...]
    fps: float

    def __post_init__(self) -> None:
        if not self.segments:
            raise ContractError("segments must not be empty")
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ContractError("fps must be a positive finite number")

    @property
    def indexed_segments(self) -> tuple[IndexedSegment, ...]:
        return tuple(IndexedSegment(index, segment) for index, segment in enumerate(self.segments))

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> SegmentsArtifact:
        raw_segments = value.get("segments")
        if isinstance(raw_segments, (str, bytes)) or not isinstance(raw_segments, Sequence):
            raise ContractError("artifact must contain a 'segments' array")

        segments: list[Segment] = []
        for index, raw_segment in enumerate(raw_segments):
            if not isinstance(raw_segment, Mapping):
                raise ContractError(f"segments[{index}] must be an object")
            try:
                segments.append(Segment.from_mapping(raw_segment))
            except ContractError as exc:
                raise ContractError(f"invalid segments[{index}]: {exc}") from exc

        if "fps" not in value:
            raise ContractError("artifact is missing 'fps'")
        return cls(segments=tuple(segments), fps=_finite_number(value["fps"], "fps"))


def load_segments_artifact(path: str | Path) -> SegmentsArtifact:
    """Load and validate an upstream ``segments.json`` file."""

    artifact_path = Path(path)
    try:
        with artifact_path.open("r", encoding="utf-8") as file:
            value: Any = json.load(file)
    except json.JSONDecodeError as exc:
        raise ContractError(f"invalid JSON in {artifact_path}: {exc.msg}") from exc
    if not isinstance(value, Mapping):
        raise ContractError("segments artifact must be a JSON object")
    return SegmentsArtifact.from_mapping(value)


@dataclass(frozen=True, slots=True)
class Highlight:
    """One upstream-compatible ``highlights.json`` record."""

    segment_index: int
    score: float

    def __post_init__(self) -> None:
        if self.segment_index < 0:
            raise ContractError("segment_index must be non-negative")
        if not math.isfinite(self.score):
            raise ContractError("score must be finite")


@dataclass(frozen=True, slots=True)
class HighlightsArtifact:
    """Upstream-compatible output envelope."""

    highlights: tuple[Highlight, ...]

    def to_mapping(self) -> dict[str, list[dict[str, object]]]:
        return {"highlights": [asdict(highlight) for highlight in self.highlights]}
