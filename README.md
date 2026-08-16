# badminton-audio-highlight

本 repository 提供羽球比賽的 YAMNet-based crowd cheer / audio highlight detection
核心邏輯，供
[badminton-analysis-system](https://github.com/2026bmtproject/badminton-analysis-system)
整合使用。

## 與主系統的責任邊界

本 repository 不負責 rally segmentation。主系統的 `match_segmentation` 是 segment
boundary 的唯一來源；本 package 只讀取其產生的 `segments.json`，包含頂層的 `fps`，
並以每筆資料在 `segments` array 中的位置作為 `segment_index`。

未來主系統會由 `modules/audio_highlight` 提供薄 adapter：讀取
`stages/match_segmentation/segments.json` 與 match video、呼叫本 package，最後寫入
`stages/audio_highlight/highlights.json`。輸出 record 格式為
`{"segment_index": int, "score": float}`。主系統的 pipeline runner 與 `BaseModule`
不屬於本 repository，也不會複製到此處。

## 預定 inference pipeline

```text
match video
+ segments.json
↓
16 kHz mono audio
↓
3.0 秒 analysis windows / 1.0 秒 hop
↓
YAMNet embeddings
↓
Logistic Regression
↓
segment-level audio highlight score
```

3.0 秒 analysis window 與 1.0 秒 hop 是 dataset / inference 的外層切窗設定，不能與
YAMNet 內部的 audio patching 混淆。提供給 YAMNet 的 audio 必須是 16 kHz mono。

## Segment boundary、analysis span 與 post padding

- **upstream segment boundary**：由 `segments.json` 中的 `start_frame`、`end_frame`、
  `start_sec`、`end_sec` 與 `duration_sec` 定義。本 repository 不會重新執行
  segmentation，也不會修改這些欄位。
- **audio analysis span**：本 package 實際讀取與分析 audio 的絕對 match timestamp
  範圍，與 upstream segment boundary 分開表示。
- **configurable post-segment padding**：為保留 rally 結束後的觀眾反應，analysis
  span 可在 segment 結尾加入 padding，目前預設為 3.0 秒。post padding 只延長 audio
  analysis span，絕不修改原始 `segments.json` 的 segment boundary。

## 目前實作狀態

目前已完成：

- `segments.json` typed contract parsing 與驗證。
- 以 array position 建立 `segment_index`。
- 16 kHz mono audio extraction、YAMNet embedding extraction 與 Logistic Regression
  classifier 的 typed interfaces。
- 使用 FFmpeg 一次建立可重建的 16 kHz mono float32 audio cache，並依 absolute
  match timestamps 對多個 `AnalysisWindow` 進行 random-access waveform slicing。
- 使用 TensorFlow Hub 官方 pretrained YAMNet，將每個既有 `AudioWindow` 的 internal
  patch embeddings 以 mean pooling 彙整為固定 `(1024,)` 的 `EmbeddedWindow`。
- 依 historical CSV 的 absolute `window_start_sec` / `window_end_sec` 匯入 reviewed
  training labels，並建立每場 match 一個安全、可重建的 YAMNet feature NPZ。
- 使用絕對 match timestamp 的 post-padding analysis span 規劃。
- upstream-compatible `highlights.json` output schema。
- `validate-segments` CLI、training / evaluation schemas 與 pytest infrastructure。

目前尚未實作：

- Logistic Regression training 與實際 classifier inference。

本 repository 不會加入額外的 CNN 或其他 ML model。

## 開發與執行

需要 Python `>=3.12,<3.13`、[uv](https://docs.astral.sh/uv/) 與可由 `PATH` 找到的
FFmpeg executable。正規化後的 `*.f32le` audio cache 是可重建的衍生資料，不應提交
至 Git。

```console
uv sync --dev
uv run pytest
uv run audio-highlight validate-segments path/to/segments.json
```

真正的 YAMNet smoke test 是手動操作，可能由 TensorFlow Hub 下載並快取 model：

```console
uv run audio-highlight smoke-test-yamnet
```

Training feature build 直接從完整 match video 依 CSV absolute timestamps 切 audio；
`segment_id` 僅保存為 `source_segment_id`，不會與 inference `segment_index` 對齊，
也不會讀取 legacy `wav_path` 或 upstream `segments.json`：

```console
uv run audio-highlight build-features \
  --match-id match_002 \
  --video local_data/match_002/match.mp4 \
  --labels local_data/match_002/labels/cheer_labels.csv \
  --output features/match_002.npz
```

`local_data/` 與 `features/` 都不提交 Git；前者保存本機 source data，後者是可重建的
generated feature artifact。
