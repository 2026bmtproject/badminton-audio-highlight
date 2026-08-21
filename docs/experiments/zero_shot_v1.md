# Zero-shot / Trivial Baseline Comparison v1

本文件記錄 `zero_shot_v1` baseline diagnostic。所有結果直接整理自既有 machine-readable artifacts：

- `artifacts/baseline_comparison/zero_shot_v1/metrics.json`
- `artifacts/baseline_comparison/zero_shot_v1/metadata.json`
- `artifacts/baseline_comparison/zero_shot_v1/predictions.csv`
- `artifacts/baseline_comparison/zero_shot_v1/summary.csv`

本實驗不取代 [`yamnet_mean_lr_v1.md`](yamnet_mean_lr_v1.md) 的 frozen 4-match LOMO checkpoint，也不將 `match_005` 併入 development macro mean。

## 1. 實驗目的

本實驗比較以下四種 3 秒 audio-window discrimination scores：

1. RMS / log-RMS energy baseline。
2. YAMNet 原生 `Cheering` class score。
3. YAMNet 原生 crowd-combination score。
4. 既有 YAMNet mean-pooled embedding 加 supervised Logistic Regression。

核心問題是：1024-dimensional YAMNet embeddings 加上 domain-specific supervised head，是否優於直接使用 YAMNet AudioSet head，以及目前 detector 是否可能主要依賴 loudness proxy。

本輪只比較 threshold-free discrimination，主要 metrics 為 ROC-AUC 與 Average Precision（AP）。沒有搜尋 threshold、最佳 F1、class combination、pooling 或其他 hyperparameters。

## 2. Frozen Scope

下列既有結果與模型均保持 frozen：

- `match_001`～`match_004` 的 4-match LOMO OOF predictions。
- `yamnet_mean_lr_v1` final detector；只由 `match_001`～`match_004` 訓練。
- `match_005` untouched external-validation predictions。
- 五場既有 features、blind labels 與 sample manifests。

Supervised LR 沒有在本實驗中重新訓練：development 直接讀取既有 LOMO OOF probabilities，external 直接讀取 frozen final detector 對 `match_005` 的 probabilities。沒有執行 `StandardScaler.fit()`、`LogisticRegression.fit()`、threshold tuning 或 calibration fitting。

## 3. Dataset 與 Alignment

| Scope | Match | Samples | Cheer | No cheer | Prevalence |
|---|---|---:|---:|---:|---:|
| Development | `match_001` | 100 | 33 | 67 | 0.33 |
| Development | `match_002` | 100 | 31 | 69 | 0.31 |
| Development | `match_003` | 100 | 55 | 45 | 0.55 |
| Development | `match_004` | 100 | 31 | 69 | 0.31 |
| External | `match_005` | 100 | 34 | 66 | 0.34 |

全部方法使用相同的 500 個既有 blind-labeled windows，沒有重新 sampling。每筆資料依下列完整 identity 對齊，而不是依 CSV 或 NPZ row position 猜測：

- `match_id`
- `sample_rank`
- `segment_index`
- `window_index_in_segment`
- `start_sec`
- `end_sec`
- `true_label`

Development 與 external 嚴格分開。`match_005` 沒有參與 development macro、class selection、aggregation selection 或其他 tuning。

## 4. Baseline Definitions

### 4.1 RMS

RMS 使用 canonical 16 kHz mono float32 waveform：

```text
rms = sqrt(mean(x^2))
log_rms_db = 20 * log10(rms + 1e-12)
```

ROC-AUC 與 AP 使用 `log_rms_db` 作為 score。沒有使用 labels 做 normalization，也沒有尋找 RMS threshold。

### 4.2 YAMNet Cheering

Official YAMNet：`https://tfhub.dev/google/yamnet/1`。

對每個 3 秒 waveform，YAMNet 產生 patch-level AudioSet scores；`Cheering` score 固定為：

```text
mean(patch-level Cheering probabilities)
```

### 4.3 YAMNet Crowd Combo

三個 class 分別沿 patch axis 取 mean，再使用預先固定的組合規則：

```text
max(
    mean(Cheering),
    mean(Applause),
    mean(Crowd)
)
```

沒有使用 max-patch pooling、weighted combination、額外 AudioSet classes 或 learned combination。

### 4.4 Embedding + LR

Supervised reference 使用 frozen `yamnet_mean_lr_v1` 設定：YAMNet patch embeddings mean pooling、1024-dimensional embedding、`StandardScaler` 與 Logistic Regression。

- `match_001`～`match_004`：既有 LOMO OOF probabilities。
- `match_005`：既有 frozen final detector probabilities。

## 5. AudioSet Class Resolution

Class indices 由官方 YAMNet class map 依 exact `display_name` 動態解析，沒有 hard-code index 或 fuzzy substring fallback：

| Class name | Resolved index |
|---|---:|
| `Cheering` | 61 |
| `Applause` | 62 |
| `Crowd` | 64 |

Class map SHA-256：`cdf24d193e196d9e95912a2667051ae203e92a2ba09449218ccb40ef787c6df2`。

## 6. Development Results

以下欄位均為 `ROC-AUC / AP`：

| Match | RMS | YAMNet Cheering | YAMNet Crowd Combo | Embedding + LR OOF |
|---|---:|---:|---:|---:|
| `match_001` | 0.958390 / 0.918133 | 0.935323 / 0.895003 | 0.973315 / 0.950301 | **0.990050 / 0.982770** |
| `match_002` | 0.936419 / 0.833113 | 0.789621 / 0.599270 | 0.812062 / 0.595380 | 0.933614 / **0.904229** |
| `match_003` | 0.847273 / 0.852769 | 0.764444 / 0.761409 | 0.800808 / 0.830206 | **0.958384 / 0.967839** |
| `match_004` | 0.938289 / 0.900749 | 0.868163 / 0.746079 | 0.842450 / 0.652746 | **0.946704 / 0.907100** |
| **Macro mean** | **0.920093 / 0.876191** | **0.839388 / 0.750440** | **0.857159 / 0.757158** | **0.957188 / 0.940484** |

Macro mean 只平均四場 development matches，每場權重相同。Zero-shot methods 沒有使用 labels training，因此其 evaluation 不稱為 LOMO；只有 supervised LR reference 使用既有 LOMO OOF predictions。

### 6.1 Macro Differences

相較於 development macro 的 Embedding + LR：

| Comparison | ROC-AUC difference | AP difference |
|---|---:|---:|
| LR − RMS | +0.037095 | +0.064294 |
| LR − YAMNet Cheering | +0.117800 | +0.190044 |
| LR − YAMNet Crowd Combo | +0.100029 | +0.183326 |

這些差異是 descriptive comparisons，沒有執行 statistical-significance test。

## 7. External `match_005` Results

| Method | ROC-AUC | Average Precision |
|---|---:|---:|
| RMS | 0.854278 | 0.736602 |
| YAMNet Cheering | 0.683155 | 0.558662 |
| YAMNet Crowd Combo | 0.629679 | 0.455351 |
| Frozen Embedding + LR | **0.930481** | **0.862126** |

相較於 `match_005` 的 frozen Embedding + LR：

| Comparison | ROC-AUC difference | AP difference |
|---|---:|---:|
| LR − RMS | +0.076203 | +0.125524 |
| LR − YAMNet Cheering | +0.247326 | +0.303464 |
| LR − YAMNet Crowd Combo | +0.300802 | +0.406774 |

這是獨立 external result，未加入 development macro，也不是第五個 LOMO fold。

## 8. Interpretation

### 8.1 Supervised head 的增益

Embedding + LR 的 ROC-AUC 與 AP 均高於 YAMNet Cheering 和 Crowd Combo 的 4/4 development matches；在 untouched `match_005` 上差距進一步擴大。就目前資料而言，這支持 domain-specific supervised head 相對於直接使用 YAMNet 原生 AudioSet class scores，帶來實質的 discrimination improvement。

固定 Crowd Combo 並沒有一致優於單獨的 Cheering。它在 `match_001` 接近 LR，但在 `match_004` 與 `match_005` 低於 Cheering，因此不能把增加 `Applause` 與 `Crowd` 視為穩定改善。

### 8.2 Loudness proxy 警訊

RMS 的 development macro ROC-AUC 為 0.920093，與 LR 的 0.957188 相比差距只有約 0.037；其 macro AP 也達到 0.876191。`match_002` 的 RMS ROC-AUC 甚至略高於 LR OOF ROC-AUC，雖然 AP 仍較低。

因此 loudness 是目前 cheer labels 的重要 predictive proxy。這是後續 detector interpretation 與 cheer intensity 設計需要正視的警訊，尤其應區分「是否有 cheer」與「聲音有多大」。不過 LR 在 development macro AP 與 external AUC/AP 均維持較高結果，所以目前證據也不足以宣稱 LR 只是 RMS detector。

### 8.3 保守結論

在目前五場、每場 100 個固定 windows 的資料上，YAMNet mean-pooled embeddings 加 Logistic Regression，比 YAMNet 原生 Cheering/Crowd scores 呈現更穩定的 discrimination；同時，簡單 RMS 已具相當強的辨識力。這兩項觀察可以並存，且都不代表模型已被證明可泛化至所有羽球比賽。

## 9. Limitations

- Development 只有 4 場 matches，external validation 只有 1 場 match。
- 每場只有 100 個 segment-diverse sampled windows，不代表完整比賽的所有時間區段。
- 本輪只比較 threshold-free discrimination，沒有評估 operating threshold、calibration 或 segment-level aggregation。
- 沒有預先定義或執行 statistical-significance test。
- RMS 可能受到場館、轉播混音、麥克風增益與 hard clipping 影響；本輪沒有執行 clipping audit 或 source-level normalization experiment。
- YAMNet zero-shot 僅測試預先固定的 `Cheering` 與三-class Crowd Combo，不能推論所有可能 AudioSet combinations 的表現。
- External result 只有 `match_005`，不能單獨證明普遍泛化。

## 10. Reproduction

```powershell
uv run audio-highlight compare-zero-shot `
  --development-matches match_001 match_002 match_003 match_004 `
  --external-match match_005 `
  --output-dir artifacts/baseline_comparison/zero_shot_v1
```

此 command 讀取既有 canonical feature identities、blind labels、audio caches 與 frozen LR predictions。它不重新 fit classifier/scaler，也不修改既有 features 或 embeddings。

## 11. Reproducibility Anchors

| Artifact | SHA-256 |
|---|---|
| Frozen final model | `c5257098315fe51db163039cca95917dd482d1612e1f08b95d301a0dbf8f79f8` |
| Frozen model metadata | `d4ad73f3d73cfc86ddc33a1ff1beefead374d4f695e9c43776f0e237ff56cdf2` |
| Development LOMO predictions | `86fd5fe852dbfe687e7d4e58f939c1b06aa7e15e1e011d00a41d944cd16dd18e` |
| External `match_005` predictions | `a37769f56675259ad5c1b12424fd1e682e3bcd251c57fe5c9fc473dffcd5b0a7` |

每場 features、labels、sample manifests 與 audio caches 的 SHA-256 完整記錄於 `metadata.json`，本文件不重複抄錄全部 hashes。

## 12. Artifacts

- `artifacts/baseline_comparison/zero_shot_v1/predictions.csv`
- `artifacts/baseline_comparison/zero_shot_v1/metrics.json`
- `artifacts/baseline_comparison/zero_shot_v1/summary.csv`
- `artifacts/baseline_comparison/zero_shot_v1/metadata.json`

`predictions.csv` 含 500 筆 identity-aligned rows，保存 RMS、三個 YAMNet native class scores、Crowd Combo 與 frozen supervised LR probability。`metrics.json` 將 development 與 external results 分開；`summary.csv` 提供逐 match 與 development macro 的簡表；`metadata.json` 保存固定規則與完整 input provenance。
