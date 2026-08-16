from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import audio_highlight.dataset as dataset_module
from audio_highlight.cli import run


def test_validate_segments_command(tmp_path, capsys) -> None:
    path = tmp_path / "segments.json"
    path.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "start_frame": 0,
                        "end_frame": 90,
                        "start_sec": 0.0,
                        "end_sec": 3.0,
                        "duration_sec": 3.0,
                    }
                ],
                "fps": 30.0,
            }
        ),
        encoding="utf-8",
    )

    assert run(["validate-segments", str(path)]) == 0
    assert "1 segments, fps=30" in capsys.readouterr().out


def test_build_features_infers_canonical_paths(monkeypatch, capsys) -> None:
    captured = {}

    def fake_build_feature_dataset(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            reviewed=100,
            binary_included=98,
            ambiguous_excluded=2,
            dataset=SimpleNamespace(embeddings=np.zeros((98, 1024))),
            output_path=kwargs["output_path"],
        )

    monkeypatch.setattr(dataset_module, "build_feature_dataset", fake_build_feature_dataset)

    assert run(["build-features", "--match-id", "match_002"]) == 0
    assert captured == {
        "video_path": Path("local_data/match_002/match.mp4"),
        "labels_path": Path("artifacts/match_002/labeling/labels.csv"),
        "manifest_path": Path(
            "artifacts/match_002/labeling/sample_manifest.json"
        ),
        "audio_cache_path": Path("artifacts/match_002/audio/audio.f32le"),
        "output_path": Path("artifacts/match_002/features/features.npz"),
    }
    assert "binary_included=98" in capsys.readouterr().out
