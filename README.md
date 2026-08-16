# badminton-audio-highlight

本 repository 提供羽球賽事的 YAMNet-based crowd cheer / audio highlight detection 核心邏輯，以及 blind labeling、feature dataset 建立與 Logistic Regression evaluation 工具。

它不負責 rally segmentation。上游 [`badminton-analysis-system`](https://github.com/2026bmtproject/badminton-analysis-system) 的 `match_segmentation` 是 segment boundary 的唯一來源；本套件只讀取 `segments.json`，不修改其中的 segment。未來主系統會由 `modules/audio_highlight` 透過薄 adapter 呼叫本套件。

## Pipeline 與時間語意

```text
match.mp4 + segments.json
↓
build_analysis_windows()
↓
deterministic 100-window sample
↓
blind human cheer labeling
↓
16 kHz mono audio + YAMNet embeddings
↓
Logistic Regression classifier / evaluation
```

`segment_index` 是 `segments.json` 中 `segments` array 的 zero-based positional index。所有 analysis window 都使用 absolute match timestamps。

預設 outer-window planner 使用 3.0 秒 window、1.0 秒 hop 與 3.0 秒 post-segment padding。post padding 只擴大 audio analysis span，不會修改 upstream segment boundary，也不會 clamp 到下一個 segment；相鄰 segment 的 analysis windows 可以重疊。planner 只產生完整 window，不產生 partial window 或 zero padding。

這裡的 outer windowing 與 YAMNet 自己的 internal patching 是不同層次。音訊會先以 FFmpeg 正規化為 16 kHz mono float32 cache，再由多個 absolute-time windows 共用；YAMNet 使用官方 `https://tfhub.dev/google/yamnet/1`，每個 window 的 patch embeddings 以 mean pooling 產生 1024-dimensional embedding。

## 資料與 artifacts

輸入固定放在影片同層：

```text
local_data/
└── match_002/
    ├── match.mp4
    └── segments.json
```

所有可重建或人工標註產物使用同一 canonical layout：

```text
artifacts/
└── match_002/
    ├── audio/
    │   ├── audio.f32le
    │   └── audio.f32le.json
    ├── labeling/
    │   ├── sample_manifest.json
    │   └── labels.csv
    ├── features/
    │   └── features.npz
    └── evaluation/
```

`sample_manifest.json` 保存 `sample_size`、`seed`、sampling algorithm version、raw `segments.json` SHA-256、planner 設定與固定 sample membership。`labels.csv` 僅保存 current `segment_index`、absolute window timestamps 與 blind human label，不包含 historical segmentation identity。

## 環境

需求：

- Python `>=3.12,<3.13`
- [uv](https://docs.astral.sh/uv/)
- 可由 `PATH` 找到的 FFmpeg executable
- CPU 可執行 TensorFlow；不需要 CUDA-specific dependency

安裝與測試：

```powershell
uv sync
uv run pytest
```

## Blind labeling

預設會從 `local_data/match_002/` 讀取影片與 `segments.json`，建立 seed `42`、algorithm version `1` 的 100-window deterministic sample。若有至少 100 個 eligible segments，第一輪 sampling 會優先從不同 segments 各取一個 window。

```powershell
uv run streamlit run src/audio_highlight/label_app.py -- `
  --match-id match_002
```

可用 `--video`、`--segments`、`--audio-cache`、`--manifest`、`--labels`、`--sample-size` 與 `--seed` 明確覆寫路徑或 sampling 設定。UI 不讀取 historical labels 或 model predictions，標註以 atomic CSV replacement 儲存並可 resume。

## 建立 features

完成全部 100 筆 review 後執行：

```powershell
uv run audio-highlight build-features --match-id match_002
```

指令會自動使用 canonical video、manifest、labels、audio cache 與 output 路徑。可用 `--video`、`--manifest`、`--labels`、`--audio-cache`、`--output` 覆寫。ambiguous samples 不會進入 binary feature dataset；尚未 review 完成時會在載入音訊或 YAMNet 前停止。

## Cross-match evaluation

evaluation 以 match 為 hold-out 單位，不做 random window split。baseline 固定為 training-fold `StandardScaler` 加 `LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, class_weight=None)`，prediction threshold 固定為 `0.5`。

```powershell
uv run audio-highlight evaluate `
  artifacts/match_002/features/features.npz `
  artifacts/match_003/features/features.npz `
  --output-dir artifacts/cross_match/evaluation
```

## Detector baseline

Current frozen detector baseline：`yamnet_mean_lr_v1`。

它固定使用官方 YAMNet、3 秒 waveform、YAMNet patch embeddings 的 mean pooling、1024-dimensional embeddings，以及 training-fold `StandardScaler` 加 `LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, class_weight=None)`。Prediction threshold 固定為 `0.5`，evaluation protocol 為 Leave-One-Match-Out。

Machine-readable configuration 位於：

```text
artifacts/baselines/yamnet_mean_lr_v1.json
```

此 JSON 只保存 frozen configuration，不包含 sklearn pickle 或 fitted model。

## Final model training / export

使用全部指定的 clean feature datasets，依 frozen baseline 設定 fit 一個 deployment instance：

```powershell
uv run audio-highlight train-model `
  --matches match_001 match_002 match_003 match_004 `
  --baseline-id yamnet_mean_lr_v1
```

亦可將 `--matches` 改為逐一提供 `features.npz` 路徑，並用 `--output-dir` 指定輸出位置。預設輸出為：

```text
artifacts/models/yamnet_mean_lr_v1/
├── model.npz
└── metadata.json
```

`model.npz` 僅包含 `StandardScaler` 與 Logistic Regression 的 numeric arrays，loader 以 `allow_pickle=False` 載入；部署推論使用 NumPy 手動計算，不需反序列化 sklearn estimator。`metadata.json` 保存 frozen configuration、training dataset identity、版本與 SHA-256。

這個 final detector 是使用目前所有 clean data 訓練的 deployment instance，不是新的 test evaluation。模型品質的正式 checkpoint 仍是既有的 4-match LOMO 結果；不得把 final fit 的 training-set 表現解讀為 test performance。實驗紀錄見 [`docs/experiments/yamnet_mean_lr_v1.md`](docs/experiments/yamnet_mean_lr_v1.md)。

## Calibration diagnostic

Calibration diagnostic 只讀取既有 out-of-fold predictions，不重新訓練 classifier、不重新載入 YAMNet、不重建 embeddings，也不搜尋或修改 threshold：

```powershell
uv run audio-highlight diagnose-calibration
```

亦可明確指定路徑：

```powershell
uv run audio-highlight diagnose-calibration `
  --predictions artifacts/cross_match/evaluation/cross_match_predictions.csv `
  --output-dir artifacts/cross_match/evaluation/calibration
```

輸出包含每場 match 的 probability distribution、10-bin uniform reliability diagram、Brier score、log loss、ECE、ROC-AUC 與 average precision，以及跨 match reliability comparison。

ROC-AUC 與 average precision 描述 ranking/discrimination；Brier score、log loss 與 reliability diagram 描述 probability quality 與 calibration-sensitive behavior。這些結果只用於診斷，現行 threshold 仍固定為 `0.5`，不會進行 Platt scaling、isotonic regression、temperature scaling 或 threshold optimization。

## 其他指令

```powershell
uv run audio-highlight validate-segments local_data/match_002/segments.json
uv run audio-highlight smoke-test-yamnet
```

`local_data/` 與 `artifacts/` 不提交至 Git。audio cache 是可重建資料，不是 source of truth；人工 `labels.csv` 應另外妥善備份。
