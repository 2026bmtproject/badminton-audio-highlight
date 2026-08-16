# YAMNet Mean + Logistic Regression Baseline v1

本文件是 frozen detector baseline `yamnet_mean_lr_v1` 的實驗紀錄，不是使用說明。表格與數值直接整理自下列 machine-readable artifact snapshot：

- `artifacts/baselines/yamnet_mean_lr_v1.json`
- `artifacts/cross_match/evaluation/cross_match_metrics.json`
- `artifacts/cross_match/evaluation/calibration/calibration_metrics.json`
- `artifacts/cross_match/evaluation/calibration/calibration_summary.csv`

## 1. 實驗目的

本實驗的任務是對 3 秒 audio window 執行 crowd cheer binary detection。Baseline 的目的，是驗證 YAMNet pretrained representation 加上 linear classifier，是否具備跨羽球比賽的歡呼辨識能力。

## 2. Frozen Model Configuration

| 項目 | Frozen value |
|---|---|
| `baseline_id` | `yamnet_mean_lr_v1` |
| YAMNet identifier | `https://tfhub.dev/google/yamnet/1` |
| Sample rate | 16,000 Hz mono |
| Window / hop / post-padding | 3.0 s / 1.0 s / 3.0 s |
| Pooling | YAMNet patch embeddings 沿 patch axis 取 mean |
| Embedding dimension | 1024 |
| Preprocessing | training-fold `StandardScaler` |
| Classifier | `LogisticRegression` |
| `C` | 1.0 |
| Solver | `lbfgs` |
| `max_iter` | 2000 |
| `class_weight` | `None` |
| Threshold | 0.5 |

此 baseline 已 freeze。後續 diagnostic 不會改變 YAMNet、pooling、classifier parameters 或 threshold。

## 3. Dataset

| Match | Samples | Cheer | No cheer |
|---|---:|---:|---:|
| `match_001` | 100 | 33 | 67 |
| `match_002` | 100 | 31 | 69 |
| `match_003` | 100 | 55 | 45 |
| `match_004` | 100 | 31 | 69 |

每場資料皆為 100 samples，來自 100 個 unique current segments。Sampling 採用 segment-diverse deterministic sampling，標籤由 blind human labeling 產生，ambiguous sample 數量為 0。Historical labels 不參與此 baseline 的 sampling、training 或 evaluation。

## 4. Evaluation Protocol

Evaluation 採 Leave-One-Match-Out（LOMO）。每個 fold 將一場完整 match 留作 held-out test match，其餘三場作為 training data；每個 fold 為 300 training samples 與 100 test samples。

`StandardScaler` 只能在 training fold 上執行 fit，再套用至 held-out match。這不是 random window split，且同一 held-out match 的 windows 不會進入該 fold 的 training data。

## 5. LOMO Results

| Held-out match | Accuracy | Precision | Recall | F1 | ROC-AUC | Average Precision |
|---|---:|---:|---:|---:|---:|---:|
| `match_001` | 0.940 | 0.966 | 0.848 | 0.903 | 0.990 | 0.983 |
| `match_002` | 0.870 | 1.000 | 0.581 | 0.735 | 0.934 | 0.904 |
| `match_003` | 0.870 | 0.850 | 0.927 | 0.887 | 0.958 | 0.968 |
| `match_004` | 0.850 | 0.711 | 0.871 | 0.783 | 0.947 | 0.907 |
| **Macro mean** | **0.883** | **0.882** | **0.807** | **0.827** | **0.957** | **0.940** |

各 fold confusion matrix 的排列為 `[[TN, FP], [FN, TP]]`：

| Held-out match | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| `match_001` | 66 | 1 | 5 | 28 |
| `match_002` | 69 | 0 | 13 | 18 |
| `match_003` | 36 | 9 | 4 | 51 |
| `match_004` | 58 | 11 | 4 | 27 |

ROC-AUC 衡量 ranking/discrimination，不能解讀為 accuracy。

## 6. Per-Match Observations

- `match_001` 的 ROC-AUC 與 Average Precision 最高，在目前四個 held-out matches 中 discrimination 最強。
- `match_002` precision 為 1.000，但 recall 為 0.581，代表 threshold 0.5 下漏掉較多 positive samples。
- `match_003` recall 為 0.927，並維持 0.850 precision。
- `match_004` recall 為 0.871，高於 0.711 precision。
- 四個 held-out matches 的 ROC-AUC 均大於 0.93。

以上是目前資料的 descriptive interpretation，不代表已證明模型可普遍泛化至所有羽球比賽。

## 7. Calibration Diagnostic

| Match | Prevalence | Predicted Positive Rate | Brier Score | Log Loss | ECE | Positive Probability Median | Negative Probability Median |
|---|---:|---:|---:|---:|---:|---:|---:|
| `match_001` | 0.330 | 0.290 | 0.0380 | 0.1246 | 0.0375 | 0.9993 | 0.001988 |
| `match_002` | 0.310 | 0.180 | 0.1124 | 0.4801 | 0.1310 | 0.6182 | 0.000249 |
| `match_003` | 0.550 | 0.600 | 0.1007 | 0.3402 | 0.0958 | 0.9981 | 0.007030 |
| `match_004` | 0.310 | 0.380 | 0.1209 | 0.5277 | 0.1318 | 0.9998 | 0.000209 |

ROC-AUC 與 Average Precision 衡量 discrimination；Brier score、Log Loss 與 reliability diagram 對 probability calibration 更敏感。ECE 使用固定 10 個 uniform bins，應與其他 calibration evidence 一起解讀，而不是單獨視為完整結論。

![四場 held-out match 的 reliability diagram](../../artifacts/cross_match/evaluation/calibration/all_matches_reliability.png)

## 8. match_002 vs match_004

`match_002` 與 `match_004` 的 observed cheer prevalence 都是 31%，但 threshold 0.5 下 predicted positive rate 分別為 18% 與 38%。Positive probability median 也分別為 0.6182 與 0.9998；相較之下，兩場 negative probability median 都接近零。

這表示不同 match 存在不同 score-distribution behavior，尤其反映在 positive-class scores 與 threshold crossing。這是 descriptive evidence，不代表 causal domain shift 已被證明。

## 9. Current Interpretation

YAMNet mean-pooled embeddings 加上 Logistic Regression，在目前四場 clean matches 的 LOMO evaluation 中呈現穩定的 cross-match discrimination ability。主要剩餘問題之一，是不同 match 間的 score calibration 與 threshold behavior。

本結果不應描述成「95.7% accuracy」；0.957 是 macro ROC-AUC，而 macro accuracy 是 0.883。這份實驗也尚未證明模型可泛化至所有羽球比賽，更不代表 cheer detection 已完全解決。

## 10. Limitations

- 目前只有 4 個 matches。
- 每場只有 100 個 labeled samples。
- Match domain diversity 仍有限。
- Classification threshold 固定為 0.5。
- 尚未進行 calibration fitting。
- Cheer intensity 尚未實作。
- Final untouched external validation 尚未完成。
- 每場只有 100 samples，10-bin reliability curve 的個別 bin 估計可能不穩定。

## 11. Next Steps

1. Train/export deployable final detector。
2. 實作 cheer intensity。
3. 實作 segment-level aggregation。
4. 整合 upstream `modules/audio_highlight`。
5. 執行 runtime benchmark。
6. 執行 untouched unseen-match validation。
7. 持續增加 match diversity。

Calibration fitting 與 threshold tuning 尚未決定；本實驗沒有加入相關結果。

## 12. Reproduction

建立每場 canonical features：

```powershell
uv run audio-highlight build-features --match-id match_001
uv run audio-highlight build-features --match-id match_002
uv run audio-highlight build-features --match-id match_003
uv run audio-highlight build-features --match-id match_004
```

執行四場 LOMO evaluation：

```powershell
uv run audio-highlight evaluate `
  artifacts/match_001/features/features.npz `
  artifacts/match_002/features/features.npz `
  artifacts/match_003/features/features.npz `
  artifacts/match_004/features/features.npz `
  --output-dir artifacts/cross_match/evaluation
```

從既有 OOF predictions 執行 calibration diagnostic：

```powershell
uv run audio-highlight diagnose-calibration `
  --predictions artifacts/cross_match/evaluation/cross_match_predictions.csv `
  --output-dir artifacts/cross_match/evaluation/calibration
```

## 13. Artifacts

- `artifacts/baselines/yamnet_mean_lr_v1.json`
- `artifacts/cross_match/evaluation/cross_match_metrics.json`
- `artifacts/cross_match/evaluation/cross_match_predictions.csv`
- `artifacts/cross_match/evaluation/calibration/calibration_metrics.json`
- `artifacts/cross_match/evaluation/calibration/calibration_summary.csv`
- `artifacts/cross_match/evaluation/calibration/all_matches_reliability.png`

Probability-distribution 與 per-match reliability plots 位於 `artifacts/cross_match/evaluation/calibration/`，檔名分別為 `match_00x_probability_distribution.png` 與 `match_00x_reliability.png`。本文件只以 relative path 引用既有圖片，沒有複製 binary image。
