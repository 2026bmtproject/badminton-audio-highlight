from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest

import audio_highlight.yamnet as yamnet_module
from audio_highlight.audio import AudioWindow
from audio_highlight.yamnet import (
    EmbeddedWindow,
    YamNetEmbeddingExtractor,
    YamNetError,
    mean_pool_embeddings,
)


class FakeYamNet:
    def __init__(self, embeddings: np.ndarray) -> None:
        self.embeddings = embeddings
        self.call_count = 0

    def __call__(self, waveform: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.call_count += 1
        samples = np.asarray(waveform)
        return (
            np.zeros((self.embeddings.shape[0], 521), dtype=np.float32),
            self.embeddings,
            np.zeros((self.embeddings.shape[0], 64), dtype=np.float32),
        )


def audio_window(
    *,
    segment_index: int = 7,
    start_sec: float = 10.0,
    samples: np.ndarray | None = None,
) -> AudioWindow:
    waveform = (
        np.zeros(48_000, dtype=np.float32)
        if samples is None
        else np.asarray(samples, dtype=np.float32)
    )
    waveform.setflags(write=False)
    return AudioWindow(
        segment_index=segment_index,
        start_sec=start_sec,
        end_sec=start_sec + 3.0,
        sample_rate_hz=16_000,
        channels=1,
        samples=waveform,
    )


def test_patch_embeddings_are_mean_pooled_to_1024() -> None:
    raw = np.stack(
        [
            np.zeros(1024, dtype=np.float32),
            np.full(1024, 2.0, dtype=np.float32),
        ]
    )

    pooled = mean_pool_embeddings(raw)

    assert pooled.shape == (1024,)
    np.testing.assert_array_equal(pooled, np.ones(1024, dtype=np.float32))


def test_single_patch_is_preserved() -> None:
    raw = np.arange(1024, dtype=np.float32)[None, :]

    pooled = mean_pool_embeddings(raw)

    np.testing.assert_array_equal(pooled, raw[0])


def test_multiple_patches_use_mean_not_another_aggregation() -> None:
    raw = np.stack(
        [
            np.full(1024, 1.0, dtype=np.float32),
            np.full(1024, 3.0, dtype=np.float32),
            np.full(1024, 8.0, dtype=np.float32),
        ]
    )

    pooled = mean_pool_embeddings(raw)

    np.testing.assert_allclose(pooled, np.full(1024, 4.0, dtype=np.float32))


def test_output_dtype_shape_and_metadata() -> None:
    model = FakeYamNet(np.ones((4, 1024), dtype=np.float64))

    result = YamNetEmbeddingExtractor(model).extract(
        audio_window(segment_index=23, start_sec=386.233)
    )

    assert isinstance(result, EmbeddedWindow)
    assert result.embedding.shape == (1024,)
    assert result.embedding.dtype == np.float32
    assert not result.embedding.flags.writeable
    assert result.segment_index == 23
    assert result.start_sec == 386.233
    assert result.end_sec == 389.233


def test_input_audio_window_is_not_modified() -> None:
    window = audio_window()
    original_fields = {key: value for key, value in asdict(window).items() if key != "samples"}
    original_samples = window.samples.copy()

    YamNetEmbeddingExtractor(FakeYamNet(np.ones((2, 1024), dtype=np.float32))).extract(window)

    assert {key: value for key, value in asdict(window).items() if key != "samples"} == original_fields
    np.testing.assert_array_equal(window.samples, original_samples)


def test_model_is_loaded_once_and_reused_across_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeYamNet(np.ones((2, 1024), dtype=np.float32))
    loaded_handles: list[str] = []

    def fake_load(handle: str) -> tuple[FakeYamNet, None]:
        loaded_handles.append(handle)
        return model, None

    monkeypatch.setattr(yamnet_module, "_load_hub_model", fake_load)
    extractor = YamNetEmbeddingExtractor()

    extractor.extract(audio_window(segment_index=0))
    extractor.extract(audio_window(segment_index=1))

    assert loaded_handles == [yamnet_module.YAMNET_MODEL_HANDLE]
    assert model.call_count == 2


def test_invalid_embedding_rank_fails() -> None:
    with pytest.raises(YamNetError, match="rank 2"):
        mean_pool_embeddings(np.ones(1024, dtype=np.float32))


def test_invalid_embedding_width_fails() -> None:
    with pytest.raises(YamNetError, match="width must be 1024"):
        mean_pool_embeddings(np.ones((3, 512), dtype=np.float32))


def test_empty_embedding_output_fails() -> None:
    with pytest.raises(YamNetError, match="no embedding patches"):
        mean_pool_embeddings(np.empty((0, 1024), dtype=np.float32))


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_non_finite_waveform_fails(invalid_value: float) -> None:
    window = audio_window()
    mutable = window.samples.copy()
    mutable[0] = invalid_value
    mutable.setflags(write=False)
    object.__setattr__(window, "samples", mutable)
    extractor = YamNetEmbeddingExtractor(FakeYamNet(np.ones((2, 1024), dtype=np.float32)))

    with pytest.raises(YamNetError, match="NaN or Inf"):
        extractor.extract(window)


def test_non_mono_waveform_fails() -> None:
    window = audio_window()
    object.__setattr__(window, "channels", 2)
    extractor = YamNetEmbeddingExtractor(FakeYamNet(np.ones((2, 1024), dtype=np.float32)))

    with pytest.raises(YamNetError, match="must be mono"):
        extractor.extract(window)


def test_out_of_range_amplitude_fails_without_clipping() -> None:
    window = audio_window()
    mutable = window.samples.copy()
    mutable[0] = 1.01
    mutable.setflags(write=False)
    object.__setattr__(window, "samples", mutable)
    extractor = YamNetEmbeddingExtractor(FakeYamNet(np.ones((2, 1024), dtype=np.float32)))

    with pytest.raises(YamNetError, match=r"within \[-1.0, 1.0\]"):
        extractor.extract(window)
