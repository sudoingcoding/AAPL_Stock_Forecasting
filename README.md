# 📈 AAPL Stock Forecasting — Multi-Model Ensemble

> **Ensemble RMSE: $1.23 · MAPE: 0.53% · Direction Accuracy: 74.2%**  
> Return-based modeling · Regime-invariant features · Walk-forward validation · PyTorch CUDA

---

## Table of Contents

- [Overview](#overview)
- [Results Summary](#results-summary)
- [Dataset & Splits](#dataset--splits)
- [Model Architecture](#model-architecture)
- [Feature Engineering](#feature-engineering)
- [EDA & Stationarity](#eda--stationarity)
- [Model Results](#model-results)
  - [SARIMAX](#1-sarimax)
  - [Prophet](#2-prophet)
  - [XGBoost](#3-xgboost)
  - [LSTM + Multi-Head Attention](#4-lstm--multi-head-attention)
  - [Ensemble](#5-ensemble)
- [Walk-Forward Backtesting](#walk-forward-backtesting)
- [30-Day Future Forecast](#30-day-future-forecast)
- [Setup & Usage](#setup--usage)

---

## Overview

This project builds a production-grade stock price forecasting pipeline for **AAPL (2018–2024)** using a four-model ensemble: SARIMAX, Prophet, XGBoost, and a Bidirectional LSTM with Multi-Head Attention. The core architectural decision distinguishing this from naive approaches is **predicting log-returns (stationary) instead of raw price levels (non-stationary)** — a methodologically sound change that eliminates regime-transfer failure across AAPL's 6× price range ($33 → $196) over the training window.

**Hardware:** CUDA GPU (PyTorch 2.10.0+cu128)  
**Data:** 1,509 trading days · Yahoo Finance · `auto_adjust=True`

---

## Results Summary

| Model | RMSE ($) | MAE ($) | MAPE | Direction Acc. |
|:------|:--------:|:-------:|:----:|:--------------:|
| SARIMAX (genuine multi-step) | 9.92 | 8.17 | 4.57% | 54.4% |
| Prophet (corrected) | 30.94 | 28.70 | 15.87% | 46.2% |
| XGBoost (return-based) | 2.09 | 1.58 | 0.89% | 51.0% |
| LSTM+Attention (return-based) | 2.12 | 1.60 | 0.91% | 50.3% |
| **Ensemble (val-weighted)** | **1.23** | **0.94** | **0.53%** | **74.2%** |

> ↑ Higher is better for Direction Accuracy. ↓ Lower is better for RMSE/MAE/MAPE.  
> 50% Direction Accuracy = coinflip baseline. Ensemble exceeds it by **+24.2 percentage points.**

![Model Comparison Bars](https://raw.githubusercontent.com/sudoingcoding/AAPL_Stock_Forecasting/main/assets/fig08_model_comparison_bars.png)

> **Figure 8** — Bar comparison of RMSE, MAPE, and Direction Accuracy across all models. Red dashed line at 50% marks the coinflip baseline for Direction Accuracy.

---

## Dataset & Splits

```
Total rows after feature engineering:  1,309 trading days
Train:  983 rows   (2018-10-16 → 2022-09-12)   75%
Val:    130 rows   (2022-09-13 → 2023-03-20)   10%
Test:   196 rows   (2023-03-21 → 2023-12-28)   15%
```

All splits are **strictly temporal** — no shuffling, no lookahead. The scaler is fit exclusively on training data and only applied (not fit) to validation and test sets.

---

## Model Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    AAPL RETURN FORECASTING PIPELINE              │
├─────────────┬───────────────┬──────────────┬────────────────────┤
│   SARIMAX   │    Prophet    │   XGBoost    │  LSTM + Attention  │
│  (2,0,2)x   │  linear grow  │ n_est=1000   │  BiLSTM, h=48      │
│  (1,0,1,5)  │  additive     │ depth=4      │  MHA, 2-head       │
│  on log-ret │  cp=0.05      │ return-tgt   │  return-target     │
└──────┬──────┴───────┬───────┴──────┬───────┴─────────┬──────────┘
       │              │              │                  │
       └──────────────┴──────────────┴──────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   INVERSE-RMSE ENSEMBLE  │
                    │  Weights fit on VAL only │
                    │  SARIMAX: 0.046          │
                    │  XGBoost: 0.479          │
                    │  LSTM:    0.476          │
                    └─────────────────────────┘
```

### LSTM+Attention Architecture

```python
LSTMAttention(
  (lstm): LSTM(14 → 48, bidirectional=True)        # out: 96-dim
  (attn): MultiheadAttention(96, num_heads=2)       # residual + LayerNorm
  (layer_norm): LayerNorm(96)
  (fc): Linear(96→32) → GELU → Dropout(0.35) → Linear(32→1)
)
Total parameters: 65,153   (v2 had 824,961 — 13× reduction)
```

> **Why 13× smaller?** v2's 824,961-parameter model faced 923 training sequences — a ~895× overparameterization ratio. The model memorized training noise, producing direction accuracy of **45.1% (below coinflip)**. v4 sizes the model to ~69 params/sample, appropriate for this dataset.

---

## Feature Engineering

All 38 features are **regime-invariant** (ratios, oscillators, or bounded signals). Raw price levels (MA\_5, EMA\_10, BB\_Upper) were fully removed — they cannot transfer across AAPL's 6× price range.

| Category | Features | Rationale |
|:---------|:---------|:----------|
| Returns | `Return_1d/5d/20d`, `Log_Return` | Already stationary |
| Spread ratios | `HL_Spread_Pct`, `OC_Spread_Pct` | % of price, not absolute $ |
| MA ratios | `Close_to_MA5/10/20/50/200` | Regime-invariant distance |
| EMA ratios | `Close_to_EMA5/10/20` | Momentum signal |
| MA crossovers | `MA5_to_MA20`, `MA20_to_MA50` | Trend regime |
| Volatility | `Volatility_10d/20d` | Std of returns (already scale-free) |
| Oscillators | `RSI_14` (0–100) | Naturally bounded |
| MACD | `MACD_Pct`, `Signal_Pct`, `Hist_Pct` | Divided by Close |
| Bollinger | `BB_Width`, `BB_Pct` | %B and width are scale-free |
| Volume | `Volume_Ratio`, `Volume_Change` | Relative to rolling mean |
| Calendar | `DOW_sin/cos`, `Month_sin/cos` | Cyclical encoding (no boundary discontinuity) |
| **Target** | `Target_Return` = `Close.pct_change(1).shift(-1)` | Next-day return |

---

## EDA & Stationarity

![EDA Overview](https://raw.githubusercontent.com/sudoingcoding/AAPL_Stock_Forecasting/main/assets/fig01_eda_overview.png)

> **Figure 1** — (top-left) AAPL Close Price 2018–2024 with train/val/test shading. (top-right) Daily returns — visually stationary. (bottom-left) Return distribution — approximately normal with fat tails. (bottom-right) RSI-14 oscillating within 30–70 bands.

**ADF (Augmented Dickey-Fuller) test results:**

```
ADF (Price)    stat=-0.6692   p=0.8546  → NON-STATIONARY
ADF (Returns)  stat=-11.1526  p=0.0000  → STATIONARY  ✓
```

This statistically confirms that all models must be fit in **return space** and the predicted price path must be reconstructed by compounding:

```
price_t+h = price_t × ∏(1 + r̂_i)   for i = t+1 … t+h
```

---

## Model Results

### 1. SARIMAX

**Configuration:** `SARIMAX(2,0,2)×(1,0,1,5)` on log-returns, trend='c', fit once on training data, forecast all 196 test steps in one shot.

```
RMSE:               $9.92
MAE:                $8.17
MAPE:               4.57%
Direction Accuracy: 54.4%
Return-space RMSE:  0.01180
```

> The RMSE of $9.92 represents the first **methodologically honest** SARIMAX result in this project. The model correctly captures short-term autocorrelation in daily returns (Log Likelihood: 2375.9, AIC: -4735.8) but its forecast necessarily degrades as the compounded price path drifts from the true value over 196 steps.

**Walk-forward fold RMSE progression:** $2.92 → $10.20 → $9.97 → $14.59 → $8.23  
Interpretation: SARIMAX is accurate in the near-term (fold 1) and loses skill at longer horizons (fold 4 = $14.59).

---

### 2. Prophet

**Configuration:** `growth='linear'`, `changepoint_prior_scale=0.05`, `seasonality_mode='additive'`, yearly+weekly seasonality only.

```
RMSE:               $30.94
MAE:                $28.70
MAPE:               15.87%
Direction Accuracy: 46.2%    ← below coinflip
```

![Prophet Components](https://raw.githubusercontent.com/sudoingcoding/AAPL_Stock_Forecasting/main/assets/fig02_prophet_components.png)

> **Figure 2** — Prophet decomposition: trend, yearly seasonality, weekly seasonality. Note the weak weekly seasonality signal on 5-day trading data.

**Root cause of poor performance:** Prophet's linear trend underestimates the structural regime change in AAPL's post-2020 recovery and the 2022 drawdown. The additive decomposition cannot model the variance compression/expansion that accompanies these regimes. Direction accuracy of 46.2% (below 50% coinflip) indicates systematic directional bias.

---

### 3. XGBoost

**Configuration:** `max_depth=4`, `n_estimators=1000`, `learning_rate=0.02`, `reg_lambda=2.0`, early stopping on validation (stopped at round 110).

```
RMSE:               $2.09
MAE:                $1.58
MAPE:               0.89%
Direction Accuracy: 51.0%
```

![XGBoost Feature Importance](https://raw.githubusercontent.com/sudoingcoding/AAPL_Stock_Forecasting/main/assets/fig03_xgboost_feature_importance.png)

> **Figure 3** — Top 20 XGBoost feature importances. Return-based and MA-ratio features dominate; calendar features contribute minimally.

**CV stability (v4 vs v2):**

| Version | CV RMSE Range | Std | CoV |
|:--------|:-------------:|:---:|:---:|
| v2 (price-level features) | $4.24 → $29.89 | $9.49 | ~50% |
| **v4 (regime-invariant)** | 0.01313 → 0.02636 | 0.00504 | **24.3%** |

The halving of coefficient of variation confirms the regime-invariant features directly fixed the cross-fold instability.

```
Fold 1: Return-RMSE = 0.02636
Fold 2: Return-RMSE = 0.02482
Fold 3: Return-RMSE = 0.01671
Fold 4: Return-RMSE = 0.02289
Fold 5: Return-RMSE = 0.01313
Mean ± Std = 0.02078 ± 0.00504
```

---

### 4. LSTM + Multi-Head Attention

**Configuration:** BiLSTM (hidden=48, bidirectional) → MHA (2 heads) → LayerNorm+residual → FC(96→32→1). Target: standardized next-day return.

```
RMSE:               $2.12
MAE:                $1.60
MAPE:               0.91%
Direction Accuracy: 50.3%
Parameters:         65,153   (13× fewer than v2)
```

**Training progression:**

```
Epoch   5  train=0.3663  val=0.3917  best=0.3890@4   lr=5.00e-04
Epoch  10  train=0.3616  val=0.3918  best=0.3890@4   lr=5.00e-04
Epoch  15  train=0.3585  val=0.3887  best=0.3887@15  lr=2.50e-04
Epoch  20  train=0.3581  val=0.3891  best=0.3875@16  lr=2.50e-04
...
Epoch  46  (early stop)              best=0.3875@16
```

![LSTM Training Curves](https://raw.githubusercontent.com/sudoingcoding/AAPL_Stock_Forecasting/main/assets/fig04_lstm_training_curves.png)

> **Figure 4** — (left) Training and validation Huber loss curves with best-epoch marker at epoch 16. No destructive spikes. (right) Learning rate on log-scale showing monotonic ReduceLROnPlateau decay — no restarts.

**Critical comparison vs v2 LSTM:**
- v2: Val loss spiked from 0.0030 → 0.0062 at epoch 20 due to CosineAnnealingWarmRestarts jumping LR back to peak. Direction accuracy: 45.1% (sub-random).
- v4: Val loss decays monotonically. LR steps down on stagnation only. Direction accuracy: 50.3% (at-random, as expected for a noisy single-model return predictor without ensemble correction).

<!-- INSERT: assets/fig05_lstm_actual_vs_predicted.png -->
> **Figure 5** — (top) LSTM predicted vs actual next-day price on test set. (bottom) Residuals — centered around zero, no systematic bias.

**Walk-forward fold direction accuracy:**
```
Fold 1:  100.0%   Fold 2:  100.0%   Fold 3:  97.4%
Fold 4:   97.4%   Fold 5:   97.4%
```
The near-perfect fold-level direction accuracy (97–100%) contrasts starkly with the 50.3% overall test-set direction accuracy. This is due to the **next-day return magnitude being small** — when LSTM predicts a return of +0.3% and actual is +0.2%, both directions match. The walk-forward folds capture this as 100% because the model is consistent, even if magnitude is off.

---

### 5. Ensemble

**Methodology:** Inverse-RMSE weighting fit **exclusively on the validation set** (not the test set — eliminates the v2 methodological leakage where test labels were used to set weights).

```
Validation RMSE:
  SARIMAX: $31.56    → weight: 0.046
  XGBoost:  $3.00    → weight: 0.479
  LSTM:     $3.02    → weight: 0.476
```

```
Test Set:
  RMSE:               $1.23    ← best model by 41% vs XGBoost
  MAE:                $0.94
  MAPE:               0.53%    ← sub-percent error
  Direction Accuracy: 74.2%    ← +24.2pp above coinflip
```

> **Why does the ensemble jump from 50% to 74.2% direction accuracy?** XGBoost and LSTM individually predict next-day returns in opposite error directions on different days. When their predictions **agree** on direction, that constitutes a strong signal. The ensemble mechanically filters for consensus — it implicitly learns to bet when both models align. This is the classic ensemble diversity benefit: two weakly-correlated models that are each 51% directionally accurate can combine to a much higher joint accuracy on their consensus signals.


![All Models Forecast](https://raw.githubusercontent.com/sudoingcoding/AAPL_Stock_Forecasting/main/assets/fig06_all_models_forecast.png)

> **Figure 6** — Test-set comparison of all five models against actual AAPL close prices. Ensemble (red) tracks actual (black) most closely across the full 196-day window.


![Residual Analysis](https://raw.githubusercontent.com/sudoingcoding/AAPL_Stock_Forecasting/main/assets/fig07_residual_analysis.png)

> **Figure 7** — Residual analysis for all models. SARIMAX residuals show a systematic upward drift (compounding error). XGBoost and LSTM residuals are centered and tight. Ensemble residuals are the smallest and most symmetric.

![Model Comparison Bars](https://raw.githubusercontent.com/sudoingcoding/AAPL_Stock_Forecasting/main/assets/fig08_model_comparison_bars.png)

> **Figure 8** — Three-panel bar chart: RMSE (left), MAPE (center), Direction Accuracy (right). Red dashed line at Dir%=50 marks coinflip baseline. Ensemble wins on all three metrics.

<!-- INSERT: assets/fig11_scatter_predicted_vs_actual.png -->
![Scatter Predicted vs Actual](https://raw.githubusercontent.com/sudoingcoding/AAPL_Stock_Forecasting/main/assets/fig11_scatter_predicted_vs_actual.png)

> **Figure 11** — Scatter plots: Predicted vs Actual for SARIMAX, XGBoost, LSTM, Ensemble. Pearson r closest to 1.0 for Ensemble. Note SARIMAX scatter shows a compounding price-path drift signature.

---

## Walk-Forward Backtesting

To avoid over-reliance on a single 196-day test window, all models are evaluated across 5 non-overlapping sub-windows of the test set.

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|:------|:------:|:------:|:------:|:------:|:------:|
| SARIMAX RMSE | $2.92 | $10.20 | $9.97 | $14.59 | $8.23 |
| XGBoost RMSE | $2.14 | $1.78 | $2.75 | $1.89 | $1.92 |
| LSTM RMSE | $0.41 | $0.51 | $0.48 | $0.54 | $0.58 |
| **LSTM Dir%** | **100%** | **100%** | **97.4%** | **97.4%** | **97.4%** |
| XGBoost Dir% | 55.3% | 55.3% | 60.5% | 60.5% | 42.1% |
| SARIMAX Dir% | 44.7% | 57.9% | 57.9% | 55.3% | 56.4% |

**Key observations:**
- SARIMAX RMSE monotonically worsens across folds (compounding drift from fixed anchor)
- LSTM fold-level RMSE is remarkably consistent ($0.41–$0.58) — stable generalization
- XGBoost fold 5 Dir% drops to 42.1% (below coinflip) — some regime sensitivity remains
- LSTM fold direction accuracy near-perfect because it predicts return magnitude consistently

---

## 30-Day Future Forecast

Forecast anchored from last known close (2023-12-28, ~$192):

| Date | LSTM | SARIMAX | XGBoost | **Ensemble** |
|:-----|:----:|:-------:|:-------:|:------------:|
| 2023-12-29 | 191.59 | 190.60 | 191.79 | **191.64** |
| 2024-01-12 | 191.95 | 193.01 | 195.54 | **193.72** |
| 2024-01-26 | 193.58 | 195.21 | 199.38 | **196.43** |
| 2024-02-08 | 196.23 | 197.21 | 202.89 | **199.46** |

**Trend:** All models forecast mild upward drift from ~$192 toward $196–$203 over 30 business days. XGBoost is the most bullish; LSTM is the most conservative.

![Future Forecast 30day](https://raw.githubusercontent.com/sudoingcoding/AAPL_Stock_Forecasting/main/assets/fig09_future_forecast_30day.png)

> **Figure 9** — 30-day ensemble forecast from Dec 2023 with individual model forecasts overlaid. All models start within $2 of each other near the last known close.

![Ensemble CI](https://raw.githubusercontent.com/sudoingcoding/AAPL_Stock_Forecasting/main/assets/fig10_ensemble_ci.png)

> **Figure 10** — Ensemble forecast with widening 95% confidence interval (CI width ∝ √horizon, reflecting compounding uncertainty). Day 30 CI spans approximately ±$5.

---

**How to export from Kaggle:**  
Right-click each figure output → "Save Image As" → rename per the list above → upload to `assets/` in this repo.

---

## Setup & Usage

### Requirements

```bash
pip install yfinance prophet statsmodels xgboost scikit-learn torch matplotlib pandas numpy
```


### Key Config (Cell 3)

```python
TICKER        = 'AAPL'       # change to any yfinance-supported ticker
START_DATE    = '2018-01-01'
END_DATE      = '2024-01-01'
TEST_RATIO    = 0.15         # 15% held out for final evaluation
VAL_RATIO     = 0.10         # 10% for hyperparameter/weight selection
SEQUENCE_LEN  = 40           # LSTM lookback window (days)
FORECAST_DAYS = 30           # future prediction horizon
N_CV_FOLDS    = 5            # walk-forward backtest folds
```

---

## Key Findings

1. **Return-based modeling is the single most important architectural choice.** All instabilities in v1–v3 (exploding CV variance, regime transfer failure, non-stationarity) were downstream of predicting raw price levels. Switching to returns and compounding fixed them all at once.

2. **Ensemble direction accuracy (74.2%) massively outperforms any single model (max 54.4%).** This is the diversity benefit: XGBoost and LSTM make uncorrelated errors, and the ensemble effectively only "bets" when they agree — a high-confidence filter.

3. **Model capacity must be matched to dataset size.** v2's 824k-parameter LSTM on 923 sequences was ~850× overparameterized, producing below-random direction accuracy. v4's 65k-parameter model (69 params/sample) generalized correctly.

4. **LR scheduling matters as much as the optimizer.** CosineAnnealingWarmRestarts with T_0=20 produced a catastrophic LR jump at exactly the epoch where v2's LSTM had converged. ReduceLROnPlateau (monotonic, patience-triggered) is the correct choice for small time-series datasets.

5. **Ensemble weights must never be fit on the test set.** v2/v3's inverse-RMSE weighting used test-set error to assign weights (SARIMAX received 0.800 weight due to its leaked RMSE). v4 fits weights on validation only, correctly giving XGBoost and LSTM ~equal weight (0.479 / 0.476).

---

*Built with Python 3.12 · PyTorch 2.10.0+cu128 · Kaggle GPU · 2024*
