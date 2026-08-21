from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import audio_highlight.clipping_audit as audit_module
from audio_highlight.clipping_audit import (
    ClippingAuditError,
    ClippingAuditResult,
    MatchClippingAudit,
    WindowAudit,
    compute_preclip_statistics,
    compute_rail_statistics,
    run_clipping_audit,
    select_top_clipped_windows,
    summarize_candidate_windows,
    summarize_labeled_windows,
    summarize_loudness_concentration,
    waveform_rms,
    write_clipping_audit_artifacts,
)
from audio_highlight.contracts import SegmentsArtifact, Segment
from audio_highlight.labeling import (
    LabelStore,
    SampleManifest,
    SampleWindow,
    build_candidate_windows,
    segments_sha256,
    write_manifest,
)
from audio_highlight.windows import InferenceConfig


def window(
    *,
    index: int,
    rail_ratio: float,
    log_rms_db: float,
    label: int | None = None,
    overflow_ratio: float = 0.0,
    preclip_peak: float = 1.0,
) -> WindowAudit:
    return WindowAudit(
        match_id="match_test",
        segment_index=index,
        window_index_in_segment=0,
        start_sec=float(index * 3),
        end_sec=float(index * 3 + 3),
        rail_sample_count=int(round(rail_ratio * 48_000)),
        rail_ratio=rail_ratio,
        has_any_rail=rail_ratio > 0,
        max_abs=1.0 if rail_ratio else 0.9,
        rms=10 ** (log_rms_db / 20),
        log_rms_db=log_rms_db,
        preclip_max_abs=preclip_peak,
        preclip_overflow_count=int(round(overflow_ratio * 48_000)),
        preclip_overflow_ratio=overflow_ratio,
        label=label,
    )


def test_exact_positive_negative_rail_count_and_ratio() -> None:
    values = np.asarray([1.0, -1.0, 0.9999999, -0.9999999, 0.0], dtype=np.float32)

    result = compute_rail_statistics(values)

    assert result.positive_rail_samples == 1
    assert result.negative_rail_samples == 1
    assert result.rail_samples == 2
    assert result.rail_ratio == pytest.approx(0.4)
    assert result.max_abs == 1.0


def test_no_clipping_and_fully_clipped_waveforms() -> None:
    clear = compute_rail_statistics(np.asarray([-0.5, 0.0, 0.5], dtype=np.float32))
    clipped = compute_rail_statistics(np.asarray([1.0, -1.0, 1.0], dtype=np.float32))

    assert clear.rail_samples == 0
    assert clear.rail_ratio == 0.0
    assert clear.clipped_run_count == 0
    assert clipped.rail_samples == 3
    assert clipped.rail_ratio == 1.0
    assert clipped.clipped_run_count == 1
    assert clipped.max_clipped_run_samples == 3


def test_consecutive_runs_cross_chunk_boundary_and_duration() -> None:
    values = np.asarray([0, 1, 1, 1, 1, 0, -1, -1, 0], dtype=np.float32)

    result = compute_rail_statistics(values, sample_rate_hz=1_000, chunk_samples=3)

    assert result.clipped_run_count == 2
    assert result.max_clipped_run_samples == 4
    assert result.max_clipped_run_ms == pytest.approx(4.0)


def test_preclip_thresholds_peak_and_max_excess() -> None:
    values = np.asarray(
        [0.5, 1.0, 1.001, -1.02, 1.06, -1.11], dtype=np.float32
    )

    result = compute_preclip_statistics(values, chunk_samples=2)

    assert result.samples_abs_gt_1 == 4
    assert result.samples_abs_gt_1_01 == 3
    assert result.samples_abs_gt_1_05 == 2
    assert result.samples_abs_gt_1_10 == 1
    assert result.overflow_ratio_gt_1 == pytest.approx(4 / 6)
    assert result.max_abs == pytest.approx(1.11)
    assert result.max_excess == pytest.approx(0.11)


def test_window_threshold_counts_and_nonzero_summaries() -> None:
    windows = (
        window(index=0, rail_ratio=0.0, log_rms_db=-20),
        window(index=1, rail_ratio=0.0001, log_rms_db=-10),
        window(index=2, rail_ratio=0.001, log_rms_db=-5),
        window(index=3, rail_ratio=0.01, log_rms_db=0),
    )

    result = summarize_candidate_windows(windows)

    assert result["windows_with_any_rail"] == 3
    assert result["windows_rail_ratio_ge_0_0001"] == 3
    assert result["windows_rail_ratio_ge_0_001"] == 2
    assert result["windows_rail_ratio_ge_0_01"] == 1
    assert result["max_window_rail_ratio"] == 0.01
    assert result["median_nonzero_window_rail_ratio"] == pytest.approx(0.001)


def test_no_nonzero_window_summary_uses_null_not_nan() -> None:
    result = summarize_candidate_windows(
        (window(index=0, rail_ratio=0.0, log_rms_db=-20),)
    )

    assert result["median_nonzero_window_rail_ratio"] is None
    assert result["p95_nonzero_window_rail_ratio"] is None


def test_rms_correctness_and_silence_is_finite() -> None:
    rms, log_rms = waveform_rms(np.asarray([3.0, 4.0], dtype=np.float32))
    silent_rms, silent_log = waveform_rms(np.zeros(48_000, dtype=np.float32))

    assert rms == pytest.approx(np.sqrt(12.5))
    assert log_rms == pytest.approx(20 * np.log10(np.sqrt(12.5) + 1e-12))
    assert silent_rms == 0.0
    assert silent_log == pytest.approx(-240.0)
    assert np.isfinite(silent_log)


def test_cheer_and_no_cheer_stratification() -> None:
    windows = (
        window(index=0, rail_ratio=0.01, log_rms_db=-2, label=1, overflow_ratio=0.02, preclip_peak=1.2),
        window(index=1, rail_ratio=0.00, log_rms_db=-8, label=1),
        window(index=2, rail_ratio=0.00, log_rms_db=-12, label=0),
        window(index=3, rail_ratio=0.001, log_rms_db=-10, label=0, overflow_ratio=0.002, preclip_peak=1.05),
    )

    result = summarize_labeled_windows(windows)

    assert result["cheer"]["sample_count"] == 2
    assert result["cheer"]["fraction_with_any_rail"] == 0.5
    assert result["cheer"]["mean_rail_ratio"] == pytest.approx(0.005)
    assert result["no_cheer"]["fraction_with_overflow"] == 0.5


def test_top_rms_groups_use_loudness_not_labels() -> None:
    windows = tuple(
        window(
            index=index,
            rail_ratio=0.1 if index == 99 else 0.0,
            log_rms_db=float(index),
            label=index % 2,
            overflow_ratio=0.2 if index == 99 else 0.0,
            preclip_peak=1.3 if index == 99 else 0.9,
        )
        for index in range(100)
    )

    result = summarize_loudness_concentration(windows)

    assert result["top_1_percent"]["window_count"] == 1
    assert result["top_5_percent"]["window_count"] == 5
    assert result["top_10_percent"]["window_count"] == 10
    assert result["top_1_percent"]["fraction_with_overflow"] == 1.0
    assert result["all_windows"]["window_count"] == 100


def test_top_clipping_order_prefers_overflow_then_peak() -> None:
    windows = (
        window(index=0, rail_ratio=0.2, log_rms_db=-2, overflow_ratio=0.01, preclip_peak=1.5),
        window(index=1, rail_ratio=0.1, log_rms_db=-3, overflow_ratio=0.02, preclip_peak=1.1),
        window(index=2, rail_ratio=0.3, log_rms_db=-1, overflow_ratio=0.02, preclip_peak=1.4),
    )

    ordered = select_top_clipped_windows(windows, limit=3)

    assert [item.segment_index for item in ordered] == [2, 1, 0]


def build_synthetic_match(tmp_path: Path):
    artifact_root = tmp_path / "artifacts"
    local_root = tmp_path / "local_data"
    match_id = "match_test"
    match_artifacts = artifact_root / match_id
    match_local = local_root / match_id
    match_local.mkdir(parents=True)
    (match_local / "match.mp4").write_bytes(b"synthetic-media")
    segment_artifact = SegmentsArtifact(
        segments=(Segment(0, 270, 0.0, 9.0, 9.0),), fps=30.0
    )
    segments_path = match_local / "segments.json"
    segments_path.write_text(
        json.dumps(
            {
                "fps": 30.0,
                "segments": [
                    {
                        "start_frame": 0,
                        "end_frame": 270,
                        "start_sec": 0.0,
                        "end_sec": 9.0,
                        "duration_sec": 9.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    candidates = build_candidate_windows(
        segment_artifact, InferenceConfig(), media_duration_sec=12.0
    )
    selected = (candidates[0], candidates[-1])
    manifest = SampleManifest(
        match_id=match_id,
        sample_size=2,
        seed=42,
        sampling_algorithm_version=1,
        segments_sha256=segments_sha256(segments_path),
        planner=InferenceConfig(),
        candidate_window_count=len(candidates),
        eligible_segment_count=1,
        windows=tuple(
            SampleWindow(
                sample_rank=index,
                segment_index=item.segment_index,
                window_index_in_segment=item.window_index_in_segment,
                candidate_count_in_segment=item.candidate_count_in_segment,
                relative_window_position=item.relative_window_position,
                start_sec=item.start_sec,
                end_sec=item.end_sec,
            )
            for index, item in enumerate(selected, start=1)
        ),
    )
    manifest_path = match_artifacts / "labeling" / "sample_manifest.json"
    write_manifest(manifest, manifest_path)
    labels_path = match_artifacts / "labeling" / "labels.csv"
    store = LabelStore(labels_path, manifest)
    store.record_decision(1, has_cheer=0, is_ambiguous=False, reviewed_at="test")
    store.record_decision(2, has_cheer=1, is_ambiguous=False, reviewed_at="test")

    preclip = np.zeros(12 * 16_000, dtype="<f4")
    preclip[1_000:1_010] = 1.2
    preclip[150_000:150_020] = -1.1
    canonical = np.clip(preclip, -1.0, 1.0).astype("<f4")
    cache_path = match_artifacts / "audio" / "audio.f32le"
    cache_path.parent.mkdir(parents=True)
    canonical.tofile(cache_path)
    return artifact_root, local_root, preclip, cache_path, labels_path


def install_fake_ffmpeg(
    monkeypatch: pytest.MonkeyPatch, preclip: np.ndarray
) -> None:
    def fake_run(command, **kwargs):
        if command[1] == "-version":
            return SimpleNamespace(returncode=0, stdout="ffmpeg synthetic\n", stderr="")
        np.asarray(preclip, dtype="<f4").tofile(Path(command[-1]))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(audit_module.subprocess, "run", fake_run)


def test_full_audit_aligns_labels_and_never_overwrites_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root, local_root, preclip, cache_path, _ = build_synthetic_match(tmp_path)
    install_fake_ffmpeg(monkeypatch, preclip)
    before = cache_path.read_bytes()
    feature_path = artifact_root / "match_test" / "features" / "features.npz"
    feature_path.parent.mkdir(parents=True)
    feature_path.write_bytes(b"frozen-feature")
    feature_before = feature_path.read_bytes()

    result = run_clipping_audit(
        ("match_test",),
        artifact_root=artifact_root,
        local_data_root=local_root,
        ffmpeg_executable="fake-ffmpeg",
    )

    match = result.matches[0]
    assert match.whole_match["total_samples"] == 12 * 16_000
    assert match.whole_match["preclip_max_abs"] == pytest.approx(1.2)
    assert match.candidate_windows["total_window_count"] == 10
    assert sum(item.label is not None for item in match.windows) == 2
    assert match.labeled_stratification["cheer"]["sample_count"] == 1
    assert match.labeled_stratification["no_cheer"]["sample_count"] == 1
    assert cache_path.read_bytes() == before
    assert feature_path.read_bytes() == feature_before
    assert not list(artifact_root.glob("**/*.preclip.f32le"))


def test_mismatched_label_identity_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root, local_root, preclip, _, labels_path = build_synthetic_match(tmp_path)
    install_fake_ffmpeg(monkeypatch, preclip)
    rows = list(csv.DictReader(labels_path.open(encoding="utf-8", newline="")))
    rows[0]["window_start_sec"] = "0.25"
    with labels_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="manifest identity mismatch"):
        run_clipping_audit(
            ("match_test",),
            artifact_root=artifact_root,
            local_data_root=local_root,
            ffmpeg_executable="fake-ffmpeg",
        )


def test_preclip_and_postclip_sample_count_must_align(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root, local_root, preclip, _, _ = build_synthetic_match(tmp_path)
    install_fake_ffmpeg(monkeypatch, preclip[:-1])

    with pytest.raises(ClippingAuditError, match="sample counts differ"):
        run_clipping_audit(
            ("match_test",),
            artifact_root=artifact_root,
            local_data_root=local_root,
            ffmpeg_executable="fake-ffmpeg",
        )


def test_audit_does_not_require_features_model_or_yamnet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root, local_root, preclip, _, _ = build_synthetic_match(tmp_path)
    install_fake_ffmpeg(monkeypatch, preclip)

    result = run_clipping_audit(
        ("match_test",),
        artifact_root=artifact_root,
        local_data_root=local_root,
        ffmpeg_executable="fake-ffmpeg",
    )

    assert result.experiment_id == "clipping_v1"
    assert not (artifact_root / "models").exists()
    assert not (artifact_root / "match_test" / "features").exists()


def test_artifact_round_trip_is_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root, local_root, preclip, _, _ = build_synthetic_match(tmp_path)
    install_fake_ffmpeg(monkeypatch, preclip)
    result = run_clipping_audit(
        ("match_test",),
        artifact_root=artifact_root,
        local_data_root=local_root,
        ffmpeg_executable="fake-ffmpeg",
    )

    first = write_clipping_audit_artifacts(result, tmp_path / "audit-a")
    second = write_clipping_audit_artifacts(result, tmp_path / "audit-b")

    for left, right in (
        (first.summary_csv, second.summary_csv),
        (first.metrics_json, second.metrics_json),
        (first.window_audit_csv, second.window_audit_csv),
        (first.top_clipped_windows_csv, second.top_clipped_windows_csv),
        (first.metadata_json, second.metadata_json),
    ):
        assert left.read_bytes() == right.read_bytes()
    metrics = json.loads(first.metrics_json.read_text(encoding="utf-8"))
    metadata = json.loads(first.metadata_json.read_text(encoding="utf-8"))
    assert metrics["experiment_id"] == "clipping_v1"
    assert metadata["canonical_cache_waveform"] == "post_clipped"
    assert metadata["preclip_measurement_available"] is True


def test_output_path_cannot_overlap_frozen_artifacts(tmp_path: Path) -> None:
    result = ClippingAuditResult(
        experiment_id="clipping_v1",
        matches=(),
        metadata={"experiment_id": "clipping_v1"},
    )

    with pytest.raises(ClippingAuditError, match="separate"):
        write_clipping_audit_artifacts(result, tmp_path / "artifacts" / "models")
