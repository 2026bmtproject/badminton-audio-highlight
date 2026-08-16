"""Streamlit UI for blind labels sampled from current inference segments."""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import streamlit as st
from numpy.typing import NDArray

from audio_highlight.audio import FFmpegAudioNormalizer, NormalizedAudioSource
from audio_highlight.labeling import (
    LabelDecision,
    LabelStore,
    LabelingError,
    SampleManifest,
    SampleWindow,
    label_statistics,
    create_or_load_manifest,
    default_segments_path,
)
from audio_highlight.spectrogram import (
    DEFAULT_SPECTROGRAM_CONFIG,
    SpectrogramConfig,
    audio_slice_to_wav_bytes,
    mel_spectrogram_db,
    render_spectrogram_rgb,
)

PAGE_SIZE = 20


@dataclass(frozen=True, slots=True)
class AppArguments:
    match_id: str
    video: Path
    segments: Path
    manifest: Path
    labels: Path
    audio_cache: Path
    sample_size: int
    seed: int


def parse_app_args(argv: Sequence[str] | None = None) -> AppArguments:
    parser = argparse.ArgumentParser(description="Current-segments blind cheer labeling")
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--segments", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--audio-cache", type=Path)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    values = parser.parse_args(argv)
    video = values.video or Path("local_data") / values.match_id / "match.mp4"
    artifact_root = Path("artifacts") / values.match_id
    labeling_root = artifact_root / "labeling"
    return AppArguments(
        match_id=values.match_id,
        video=video,
        segments=(
            values.segments
            if values.segments is not None
            else default_segments_path(video)
        ),
        manifest=(
            values.manifest
            if values.manifest is not None
            else labeling_root / "sample_manifest.json"
        ),
        labels=(
            values.labels
            if values.labels is not None
            else labeling_root / "labels.csv"
        ),
        audio_cache=(
            values.audio_cache
            if values.audio_cache is not None
            else artifact_root / "audio" / "audio.f32le"
        ),
        sample_size=values.sample_size,
        seed=values.seed,
    )


@st.cache_resource(show_spinner="Normalizing 16 kHz mono audio once...")
def _normalized_source(video: str, cache: str) -> NormalizedAudioSource:
    return FFmpegAudioNormalizer().normalize(video, cache)


def _source_hash(source: NormalizedAudioSource) -> tuple[str, int, int]:
    stat = source.path.stat()
    return str(source.path.resolve()), stat.st_size, stat.st_mtime_ns


@st.cache_data(
    show_spinner=False,
    max_entries=500,
    hash_funcs={NormalizedAudioSource: _source_hash},
)
def _window_media(
    source: NormalizedAudioSource,
    start_sec: float,
    end_sec: float,
    config: SpectrogramConfig,
) -> tuple[bytes, NDArray[np.uint8]]:
    audio = source.slice_absolute(start_sec, end_sec)
    spectrogram = mel_spectrogram_db(audio.samples, config)
    return audio_slice_to_wav_bytes(audio), render_spectrogram_rgb(spectrogram, config)


def validate_app_args(args: AppArguments) -> None:
    protected = {args.video.resolve(), args.segments.resolve()}
    if args.manifest.resolve() in protected or args.labels.resolve() in protected:
        raise LabelingError("labeling outputs must not overwrite video or segments.json")
    if args.manifest.resolve() == args.labels.resolve():
        raise LabelingError("manifest and labels must use different files")


def _render_progress(
    manifest: SampleManifest,
    decisions: dict[int, LabelDecision],
) -> None:
    stats = label_statistics(manifest, decisions)
    st.sidebar.subheader("Labeling progress")
    st.sidebar.write(f"Sample size: {stats.sample_size}")
    st.sidebar.write(f"Reviewed: {stats.reviewed}")
    st.sidebar.write(f"Remaining: {stats.remaining}")
    st.sidebar.write(f"Cheer: {stats.cheer_count}")
    st.sidebar.write(f"No cheer: {stats.no_cheer_count}")
    st.sidebar.write(f"Ambiguous: {stats.ambiguous_count}")
    st.sidebar.write(f"Unique segments: {stats.unique_segments}")
    st.sidebar.write(f"Max samples/segment: {stats.max_samples_per_segment}")


def _render_label_card(
    window: SampleWindow,
    *,
    source: NormalizedAudioSource,
    store: LabelStore,
    decision: LabelDecision | None,
) -> None:
    st.markdown(f"**Sample {window.sample_rank}**")
    st.caption(
        f"Window {window.start_sec:.3f} ??{window.end_sec:.3f} sec 繚 "
        f"segment_index {window.segment_index} 繚 position "
        f"{window.window_index_in_segment + 1}/{window.candidate_count_in_segment}"
    )
    wav_bytes, spectrogram_rgb = _window_media(
        source,
        window.start_sec,
        window.end_sec,
        DEFAULT_SPECTROGRAM_CONFIG,
    )
    st.image(spectrogram_rgb, width="stretch")
    st.audio(wav_bytes, format="audio/wav")
    notes = st.text_input(
        "Notes",
        value="" if decision is None else decision.notes,
        key=f"label-notes:{window.sample_rank}",
        label_visibility="collapsed",
        placeholder="Optional notes",
    )
    controls = st.columns(3)
    choices = ((0, "0 No cheer"), (1, "1 Cheer"), (None, "? Ambiguous"))
    for control, (label, text) in zip(controls, choices, strict=True):
        if control.button(
            text,
            key=f"label-decision:{window.sample_rank}:{text}",
            width="stretch",
        ):
            store.record_decision(
                window.sample_rank,
                has_cheer=label,
                is_ambiguous=label is None,
                notes=notes,
            )
            st.rerun()
    if decision is not None:
        value = "? ambiguous" if decision.is_ambiguous else str(decision.has_cheer)
        st.success(f"Saved label: {value}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_app_args(sys.argv[1:] if argv is None else argv)
    st.set_page_config(page_title="Current Segments Clean Labeling", layout="wide")
    st.title("Current Segments Clean Labeling")
    st.caption(
        "Blind labels from the current inference window planner; model predictions "
        "are not loaded."
    )
    try:
        validate_app_args(args)
        source = _normalized_source(str(args.video), str(args.audio_cache))
        result = create_or_load_manifest(
            match_id=args.match_id,
            segments_path=args.segments,
            manifest_path=args.manifest,
            sample_size=args.sample_size,
            seed=args.seed,
            media_duration_sec=source.duration_sec,
        )
        manifest = result.manifest
        store = LabelStore(args.labels, manifest)
        decisions = store.decisions
    except (LabelingError, OSError, ValueError) as exc:
        st.error(str(exc))
        st.stop()

    st.caption(
        f"segments SHA-256: `{manifest.segments_sha256}` 繚 "
        f"manifest: `{args.manifest}` 繚 membership "
        f"{'created' if result.created else 'resumed'}"
    )
    _render_progress(manifest, decisions)
    page_count = math.ceil(manifest.sample_size / PAGE_SIZE)
    st.sidebar.subheader("Page")
    if "label_page" not in st.session_state:
        st.session_state.label_page = 1
    navigation = st.sidebar.columns(2)
    if navigation[0].button("Previous", disabled=st.session_state.label_page <= 1):
        st.session_state.label_page -= 1
    if navigation[1].button(
        "Next", disabled=st.session_state.label_page >= page_count
    ):
        st.session_state.label_page += 1
    page_number = st.sidebar.number_input(
        "Page number",
        min_value=1,
        max_value=page_count,
        value=min(st.session_state.label_page, page_count),
        step=1,
    )
    st.session_state.label_page = int(page_number)
    st.sidebar.write(f"Page {page_number} / {page_count}")

    start = (int(page_number) - 1) * PAGE_SIZE
    page = manifest.windows[start : start + PAGE_SIZE]
    columns = st.columns(4)
    for index, window in enumerate(page):
        with columns[index % 4]:
            with st.container(border=True):
                _render_label_card(
                    window,
                    source=source,
                    store=store,
                    decision=decisions.get(window.sample_rank),
                )


if __name__ == "__main__":
    main()
