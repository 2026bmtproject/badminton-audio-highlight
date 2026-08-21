from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import audio_highlight.dataset as dataset_module
import audio_highlight.full_inference as full_inference_module
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


def test_infer_match_uses_generic_canonical_paths(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    captured = {}

    def fake_infer_full_match(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            match_id="match_006",
            inference_runtime_sec=1.25,
            summary={
                "candidate_window_count": 7,
                "segment_count": 2,
                "eligible_segment_count": 2,
                "probability": {"min": 0.1, "median": 0.4, "max": 0.9},
                "descriptive_threshold_counts": {
                    "p_ge_0_5": {"count": 2, "rate": 2 / 7},
                    "p_ge_0_9": {"count": 1, "rate": 1 / 7},
                    "p_ge_0_99": {"count": 0, "rate": 0.0},
                },
                "sampled_external_probability_equivalence": {
                    "matched_window_count": 0,
                    "max_absolute_probability_difference": None,
                },
            },
        )

    def fake_write(result, output_dir):
        captured["output_dir"] = output_dir
        output = Path(output_dir)
        return SimpleNamespace(
            cheer_windows_csv=output / "cheer_windows.csv",
            summary_json=output / "summary.json",
            metadata_json=output / "metadata.json",
            sampling_distribution_json=output / "sampling_distribution.json",
            sampling_windows_csv=output / "sampling_windows.csv",
            sampling_relative_position_plot=output / "sampling_relative_position.png",
        )

    monkeypatch.setattr(full_inference_module, "infer_full_match", fake_infer_full_match)
    monkeypatch.setattr(
        full_inference_module, "write_full_match_inference_artifacts", fake_write
    )

    assert run(["infer-match", "--match-id", "match_006"]) == 0
    assert captured["video_path"] == Path("local_data/match_006/match.mp4")
    assert captured["segments_path"] == Path("local_data/match_006/segments.json")
    assert captured["audio_cache_path"] == Path(
        "artifacts/match_006/audio/audio.f32le"
    )
    assert captured["manifest_path"] is None
    assert captured["external_predictions_path"] is None
    assert captured["output_dir"] == Path("artifacts/match_006/inference")
    assert "candidate_window_count=7" in capsys.readouterr().out
