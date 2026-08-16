"""Current-segments candidate sampling and blind-label persistence."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from audio_highlight.contracts import SegmentsArtifact, load_segments_artifact
from audio_highlight.windows import InferenceConfig, build_analysis_windows

SAMPLING_ALGORITHM_VERSION = 1
LABEL_SOURCE = "current_segments_blind_human"
_LABEL_FIELDS = (
    "match_id",
    "segments_sha256",
    "sample_rank",
    "segment_index",
    "window_index_in_segment",
    "window_start_sec",
    "window_end_sec",
    "has_cheer",
    "is_ambiguous",
    "notes",
    "reviewed_at",
)


class LabelingError(ValueError):
    """Raised when current-segments sampling or labels are inconsistent."""


@dataclass(frozen=True, slots=True)
class CandidateWindow:
    segment_index: int
    window_index_in_segment: int
    candidate_count_in_segment: int
    start_sec: float
    end_sec: float

    @property
    def relative_window_position(self) -> float:
        if self.candidate_count_in_segment == 1:
            return 0.5
        return self.window_index_in_segment / (self.candidate_count_in_segment - 1)


@dataclass(frozen=True, slots=True)
class SampleWindow:
    sample_rank: int
    segment_index: int
    window_index_in_segment: int
    candidate_count_in_segment: int
    relative_window_position: float
    start_sec: float
    end_sec: float

    def __post_init__(self) -> None:
        if self.sample_rank <= 0 or self.segment_index < 0:
            raise LabelingError("sample rank and segment index must be valid")
        if not 0 <= self.window_index_in_segment < self.candidate_count_in_segment:
            raise LabelingError("window index must fit its candidate count")
        if not 0.0 <= self.relative_window_position <= 1.0:
            raise LabelingError("relative window position must be in [0, 1]")
        if self.start_sec < 0 or self.end_sec <= self.start_sec:
            raise LabelingError("sample must use absolute start/end timestamps")


@dataclass(frozen=True, slots=True)
class SampleManifest:
    match_id: str
    sample_size: int
    seed: int
    sampling_algorithm_version: int
    segments_sha256: str
    planner: InferenceConfig
    candidate_window_count: int
    eligible_segment_count: int
    windows: tuple[SampleWindow, ...]

    def __post_init__(self) -> None:
        if not self.match_id or len(self.segments_sha256) != 64:
            raise LabelingError("manifest identity and SHA-256 must be valid")
        if self.sampling_algorithm_version != SAMPLING_ALGORITHM_VERSION:
            raise LabelingError(
                "unsupported sampling algorithm version: "
                f"{self.sampling_algorithm_version}"
            )
        if self.sample_size != len(self.windows) or self.sample_size <= 0:
            raise LabelingError("manifest sample_size must match its windows")
        if self.candidate_window_count < self.sample_size:
            raise LabelingError("manifest has fewer candidates than samples")
        if self.eligible_segment_count <= 0:
            raise LabelingError("manifest must contain eligible current segments")
        ranks = [window.sample_rank for window in self.windows]
        if ranks != list(range(1, self.sample_size + 1)):
            raise LabelingError("manifest sample ranks must be contiguous from 1")
        identities = {
            (
                window.segment_index,
                window.window_index_in_segment,
                window.start_sec,
                window.end_sec,
            )
            for window in self.windows
        }
        if len(identities) != self.sample_size:
            raise LabelingError("manifest contains duplicate candidate windows")


@dataclass(frozen=True, slots=True)
class ManifestResult:
    manifest: SampleManifest
    path: Path
    created: bool


@dataclass(frozen=True, slots=True)
class LabelDecision:
    match_id: str
    segments_sha256: str
    sample_rank: int
    segment_index: int
    window_index_in_segment: int
    window_start_sec: float
    window_end_sec: float
    has_cheer: int | None
    is_ambiguous: bool
    notes: str
    reviewed_at: str

    def __post_init__(self) -> None:
        if self.is_ambiguous:
            if self.has_cheer is not None:
                raise LabelingError("ambiguous labels must remain non-binary")
        elif self.has_cheer not in {0, 1}:
            raise LabelingError("non-ambiguous labels require 0 or 1")
        if not self.reviewed_at:
            raise LabelingError("reviewed_at must not be empty")


@dataclass(frozen=True, slots=True)
class LabelStatistics:
    sample_size: int
    reviewed: int
    remaining: int
    cheer_count: int
    no_cheer_count: int
    ambiguous_count: int
    unique_segments: int
    max_samples_per_segment: int


def default_segments_path(video_path: str | Path) -> Path:
    """Resolve the user-defined colocated current segments.json convention."""

    return Path(video_path).parent / "segments.json"


def segments_sha256(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_candidate_windows(
    artifact: SegmentsArtifact,
    config: InferenceConfig | None = None,
    *,
    media_duration_sec: float | None = None,
) -> tuple[CandidateWindow, ...]:
    """Add structural indices around windows from the existing inference planner."""

    planned = build_analysis_windows(
        artifact,
        config or InferenceConfig(),
        media_duration_sec=media_duration_sec,
    )
    grouped: dict[int, list[Any]] = defaultdict(list)
    for window in planned:
        grouped[window.segment_index].append(window)
    candidates: list[CandidateWindow] = []
    for segment_index in sorted(grouped):
        segment_windows = grouped[segment_index]
        count = len(segment_windows)
        for window_index, window in enumerate(segment_windows):
            candidates.append(
                CandidateWindow(
                    segment_index=segment_index,
                    window_index_in_segment=window_index,
                    candidate_count_in_segment=count,
                    start_sec=window.start_sec,
                    end_sec=window.end_sec,
                )
            )
    return tuple(candidates)


def sample_segment_diverse_windows(
    candidates: Sequence[CandidateWindow],
    *,
    sample_size: int = 100,
    seed: int = 42,
) -> tuple[SampleWindow, ...]:
    """Use one-per-segment first, then deterministic balanced max-distance fill."""

    if sample_size <= 0:
        raise LabelingError("sample_size must be positive")
    if sample_size > len(candidates):
        raise LabelingError(
            f"requested {sample_size} samples from only {len(candidates)} candidates"
        )
    by_segment: dict[int, list[CandidateWindow]] = defaultdict(list)
    for candidate in candidates:
        by_segment[candidate.segment_index].append(candidate)
    if not by_segment:
        raise LabelingError("current segments produce no complete candidate windows")
    for segment_candidates in by_segment.values():
        segment_candidates.sort(key=lambda item: item.window_index_in_segment)

    rng = random.Random(seed)
    segment_order = list(sorted(by_segment))
    rng.shuffle(segment_order)
    selected: list[CandidateWindow] = []
    selected_indices: dict[int, set[int]] = defaultdict(set)

    for segment_index in segment_order:
        options = by_segment[segment_index]
        choice_index = random.Random(
            f"clean-v{SAMPLING_ALGORITHM_VERSION}:{seed}:{segment_index}"
        ).randrange(len(options))
        choice = options[choice_index]
        selected.append(choice)
        selected_indices[segment_index].add(choice.window_index_in_segment)
        if len(selected) == sample_size:
            break

    while len(selected) < sample_size:
        available_segments = [
            segment_index
            for segment_index in segment_order
            if len(selected_indices[segment_index]) < len(by_segment[segment_index])
        ]
        if not available_segments:
            raise LabelingError("sampling exhausted candidates unexpectedly")
        minimum_count = min(len(selected_indices[index]) for index in available_segments)
        balanced_segments = [
            index
            for index in available_segments
            if len(selected_indices[index]) == minimum_count
        ]
        for segment_index in balanced_segments:
            choice = _most_temporally_distant(
                by_segment[segment_index],
                selected_indices[segment_index],
                seed=seed,
                segment_index=segment_index,
            )
            selected.append(choice)
            selected_indices[segment_index].add(choice.window_index_in_segment)
            if len(selected) == sample_size:
                break

    return tuple(
        SampleWindow(
            sample_rank=rank,
            segment_index=candidate.segment_index,
            window_index_in_segment=candidate.window_index_in_segment,
            candidate_count_in_segment=candidate.candidate_count_in_segment,
            relative_window_position=candidate.relative_window_position,
            start_sec=candidate.start_sec,
            end_sec=candidate.end_sec,
        )
        for rank, candidate in enumerate(selected, start=1)
    )


def _most_temporally_distant(
    candidates: Sequence[CandidateWindow],
    selected_indices: set[int],
    *,
    seed: int,
    segment_index: int,
) -> CandidateWindow:
    remaining = [
        candidate
        for candidate in candidates
        if candidate.window_index_in_segment not in selected_indices
    ]
    distances = {
        candidate.window_index_in_segment: min(
            abs(candidate.window_index_in_segment - chosen)
            for chosen in selected_indices
        )
        for candidate in remaining
    }
    maximum_distance = max(distances.values())
    tied = [
        candidate
        for candidate in remaining
        if distances[candidate.window_index_in_segment] == maximum_distance
    ]
    tied.sort(
        key=lambda candidate: hashlib.sha256(
            f"{seed}:{segment_index}:{candidate.window_index_in_segment}".encode()
        ).digest()
    )
    return tied[0]


def create_or_load_manifest(
    *,
    match_id: str,
    segments_path: str | Path,
    manifest_path: str | Path,
    sample_size: int = 100,
    seed: int = 42,
    config: InferenceConfig | None = None,
    media_duration_sec: float | None = None,
) -> ManifestResult:
    """Create immutable membership once or validate and reuse an existing manifest."""

    settings = config or InferenceConfig()
    source_path = Path(segments_path)
    output_path = Path(manifest_path)
    fingerprint = segments_sha256(source_path)
    artifact = load_segments_artifact(source_path)
    candidates = build_candidate_windows(
        artifact, settings, media_duration_sec=media_duration_sec
    )
    eligible_segments = len({candidate.segment_index for candidate in candidates})

    if output_path.exists():
        manifest = load_manifest(output_path)
        _validate_existing_manifest(
            manifest,
            match_id=match_id,
            sample_size=sample_size,
            seed=seed,
            fingerprint=fingerprint,
            config=settings,
            candidates=candidates,
        )
        return ManifestResult(manifest, output_path, False)

    windows = sample_segment_diverse_windows(
        candidates, sample_size=sample_size, seed=seed
    )
    manifest = SampleManifest(
        match_id=match_id,
        sample_size=sample_size,
        seed=seed,
        sampling_algorithm_version=SAMPLING_ALGORITHM_VERSION,
        segments_sha256=fingerprint,
        planner=settings,
        candidate_window_count=len(candidates),
        eligible_segment_count=eligible_segments,
        windows=windows,
    )
    write_manifest(manifest, output_path)
    return ManifestResult(manifest, output_path, True)


def _validate_existing_manifest(
    manifest: SampleManifest,
    *,
    match_id: str,
    sample_size: int,
    seed: int,
    fingerprint: str,
    config: InferenceConfig,
    candidates: Sequence[CandidateWindow],
) -> None:
    expected = (match_id, sample_size, seed, fingerprint, config)
    actual = (
        manifest.match_id,
        manifest.sample_size,
        manifest.seed,
        manifest.segments_sha256,
        manifest.planner,
    )
    if actual != expected:
        raise LabelingError(
            "existing manifest does not match match/seed/sample/planner/segments SHA-256"
        )
    candidate_identities = {
        (
            candidate.segment_index,
            candidate.window_index_in_segment,
            candidate.start_sec,
            candidate.end_sec,
        )
        for candidate in candidates
    }
    if any(
        (
            window.segment_index,
            window.window_index_in_segment,
            window.start_sec,
            window.end_sec,
        )
        not in candidate_identities
        for window in manifest.windows
    ):
        raise LabelingError("existing manifest membership is not in current planner output")


def write_manifest(manifest: SampleManifest, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    value = {
        "match_id": manifest.match_id,
        "sample_size": manifest.sample_size,
        "seed": manifest.seed,
        "sampling_algorithm_version": manifest.sampling_algorithm_version,
        "segments_sha256": manifest.segments_sha256,
        "planner": asdict(manifest.planner),
        "candidate_window_count": manifest.candidate_window_count,
        "eligible_segment_count": manifest.eligible_segment_count,
        "windows": [asdict(window) for window in manifest.windows],
    }
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(value, file, ensure_ascii=False, indent=2)
            file.write("\n")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def load_manifest(path: str | Path) -> SampleManifest:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        planner = InferenceConfig(**value["planner"])
        windows = tuple(SampleWindow(**item) for item in value["windows"])
        return SampleManifest(
            match_id=value["match_id"],
            sample_size=value["sample_size"],
            seed=value["seed"],
            sampling_algorithm_version=value["sampling_algorithm_version"],
            segments_sha256=value["segments_sha256"],
            planner=planner,
            candidate_window_count=value["candidate_window_count"],
            eligible_segment_count=value["eligible_segment_count"],
            windows=windows,
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise LabelingError(f"invalid manifest: {exc}") from exc


class LabelStore:
    """Atomic label CSV keyed only by immutable manifest membership."""

    def __init__(self, path: str | Path, manifest: SampleManifest) -> None:
        self.path = Path(path)
        self.manifest = manifest
        self._windows = {window.sample_rank: window for window in manifest.windows}
        self._decisions = self._load() if self.path.exists() else {}

    @property
    def decisions(self) -> dict[int, LabelDecision]:
        return dict(self._decisions)

    def record_decision(
        self,
        sample_rank: int,
        *,
        has_cheer: int | None,
        is_ambiguous: bool,
        notes: str = "",
        reviewed_at: str | None = None,
    ) -> LabelDecision:
        try:
            window = self._windows[sample_rank]
        except KeyError as exc:
            raise LabelingError(f"unknown sample rank: {sample_rank}") from exc
        decision = LabelDecision(
            match_id=self.manifest.match_id,
            segments_sha256=self.manifest.segments_sha256,
            sample_rank=sample_rank,
            segment_index=window.segment_index,
            window_index_in_segment=window.window_index_in_segment,
            window_start_sec=window.start_sec,
            window_end_sec=window.end_sec,
            has_cheer=None if is_ambiguous else has_cheer,
            is_ambiguous=is_ambiguous,
            notes=notes,
            reviewed_at=reviewed_at or datetime.now(UTC).isoformat(),
        )
        updated = dict(self._decisions)
        updated[sample_rank] = decision
        self._write(updated)
        self._decisions = updated
        return decision

    def _load(self) -> dict[int, LabelDecision]:
        decisions: dict[int, LabelDecision] = {}
        with self.path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            missing = set(_LABEL_FIELDS) - set(reader.fieldnames or ())
            if missing:
                raise LabelingError(
                    f"label CSV is missing fields: {', '.join(sorted(missing))}"
                )
            for row_number, row in enumerate(reader, start=2):
                try:
                    ambiguous = _required_bool(row["is_ambiguous"])
                    decision = LabelDecision(
                        match_id=row["match_id"],
                        segments_sha256=row["segments_sha256"],
                        sample_rank=int(row["sample_rank"]),
                        segment_index=int(row["segment_index"]),
                        window_index_in_segment=int(row["window_index_in_segment"]),
                        window_start_sec=float(row["window_start_sec"]),
                        window_end_sec=float(row["window_end_sec"]),
                        has_cheer=(
                            None if row["has_cheer"] == "" else int(row["has_cheer"])
                        ),
                        is_ambiguous=ambiguous,
                        notes=row["notes"],
                        reviewed_at=row["reviewed_at"],
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise LabelingError(
                        f"label CSV row {row_number}: invalid value: {exc}"
                    ) from exc
                if decision.sample_rank in decisions:
                    raise LabelingError(
                        f"label CSV row {row_number}: duplicate sample rank"
                    )
                window = self._windows.get(decision.sample_rank)
                if window is None or not _decision_matches(
                    decision, self.manifest, window
                ):
                    raise LabelingError(
                        f"label CSV row {row_number}: manifest identity mismatch"
                    )
                decisions[decision.sample_rank] = decision
        return decisions

    def _write(self, decisions: dict[int, LabelDecision]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=_LABEL_FIELDS)
                writer.writeheader()
                for sample_rank in sorted(decisions):
                    decision = decisions[sample_rank]
                    row = asdict(decision)
                    row["has_cheer"] = (
                        "" if decision.has_cheer is None else decision.has_cheer
                    )
                    row["is_ambiguous"] = (
                        "true" if decision.is_ambiguous else "false"
                    )
                    writer.writerow(row)
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)


def label_statistics(
    manifest: SampleManifest,
    decisions: dict[int, LabelDecision],
) -> LabelStatistics:
    counts: dict[int, int] = defaultdict(int)
    for window in manifest.windows:
        counts[window.segment_index] += 1
    values = list(decisions.values())
    return LabelStatistics(
        sample_size=manifest.sample_size,
        reviewed=len(values),
        remaining=manifest.sample_size - len(values),
        cheer_count=sum(item.has_cheer == 1 for item in values),
        no_cheer_count=sum(item.has_cheer == 0 for item in values),
        ambiguous_count=sum(item.is_ambiguous for item in values),
        unique_segments=len(counts),
        max_samples_per_segment=max(counts.values()),
    )


def _decision_matches(
    decision: LabelDecision,
    manifest: SampleManifest,
    window: SampleWindow,
) -> bool:
    return (
        decision.match_id == manifest.match_id
        and decision.segments_sha256 == manifest.segments_sha256
        and decision.segment_index == window.segment_index
        and decision.window_index_in_segment == window.window_index_in_segment
        and math.isclose(
            decision.window_start_sec, window.start_sec, rel_tol=0.0, abs_tol=1e-9
        )
        and math.isclose(
            decision.window_end_sec, window.end_sec, rel_tol=0.0, abs_tol=1e-9
        )
    )


def _required_bool(value: str) -> bool:
    if value not in {"true", "false"}:
        raise ValueError("expected true or false")
    return value == "true"
