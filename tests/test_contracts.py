from __future__ import annotations

import json

import pytest

from audio_highlight.contracts import (
    ContractError,
    Highlight,
    HighlightsArtifact,
    SegmentsArtifact,
    load_segments_artifact,
)


def valid_envelope() -> dict[str, object]:
    return {
        "segments": [
            {
                "start_frame": 30,
                "end_frame": 120,
                "start_sec": 1.0,
                "end_sec": 4.0,
                "duration_sec": 3.0,
            },
            {
                "start_frame": 180,
                "end_frame": 300,
                "start_sec": 6.0,
                "end_sec": 10.0,
                "duration_sec": 4.0,
            },
        ],
        "fps": 30.0,
    }


def test_segments_use_array_positions_as_indices() -> None:
    artifact = SegmentsArtifact.from_mapping(valid_envelope())

    assert [item.segment_index for item in artifact.indexed_segments] == [0, 1]
    assert artifact.indexed_segments[1].segment.start_frame == 180


def test_load_segments_artifact(tmp_path) -> None:
    path = tmp_path / "segments.json"
    path.write_text(json.dumps(valid_envelope()), encoding="utf-8")

    artifact = load_segments_artifact(path)

    assert artifact.fps == 30.0
    assert len(artifact.segments) == 2


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda value: value.pop("fps"), "missing 'fps'"),
        (lambda value: value.update(segments=[]), "must not be empty"),
        (lambda value: value["segments"][0].pop("start_sec"), "missing 'start_sec'"),
    ],
)
def test_invalid_contract_is_rejected(change, message: str) -> None:
    value = valid_envelope()
    change(value)

    with pytest.raises(ContractError, match=message):
        SegmentsArtifact.from_mapping(value)


def test_highlights_serialize_to_upstream_envelope() -> None:
    artifact = HighlightsArtifact((Highlight(segment_index=1, score=0.75),))

    assert artifact.to_mapping() == {
        "highlights": [{"segment_index": 1, "score": 0.75}]
    }
