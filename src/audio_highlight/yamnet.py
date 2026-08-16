"""Typed YAMNet embedding boundary; TensorFlow inference is not implemented yet."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from audio_highlight.audio import AudioWindow


@dataclass(frozen=True, slots=True)
class TimedEmbedding:
    """One YAMNet embedding with absolute match-time bounds."""

    start_sec: float
    end_sec: float
    values: tuple[float, ...]


class YamNetEmbeddingExtractor(Protocol):
    """CPU-compatible YAMNet adapter boundary.

    The input is an outer 3-second audio window. Implementations may produce multiple
    embeddings because YAMNet performs its own internal patching; those internal patches
    must not be treated as replacements for outer window planning.
    """

    def extract(self, clip: AudioWindow) -> Sequence[TimedEmbedding]:
        """Return YAMNet embeddings with timestamps anchored to ``clip.start_sec``."""
        ...
