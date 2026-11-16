# Predicting Stock Price Direction from Market Micro-Movements and Macroeconomics

**Capstone Project — Berkeley ML/AI Professional Certificate**
**Module 20.1: Initial Report and Exploratory Data Analysis**

📓 **Notebook:** [notebooks/capstone_eda_baseline.ipynb](notebooks/capstone_eda_baseline.ipynb)

---

## Research Question

Is it possible to predict stock price movements from daily price/volume behaviour (technical indicators such as EMA,
RSI and MACD) combined with macroeconomic conditions? And can those predictions drive an algorithmic trading
strategy that earns a strong risk-adjusted return while **avoiding severe drawdowns**?

For this initial report the question is framed as a **binary classification**: *will a stock's return tomorrow be
positive?* The project's success target is **directional accuracy above 55%** on unseen data.

## Data Sources

| Source | Data | Frequency |
|---|---|---|
| [Yahoo Finance](https://finance.yahoo.com) (`yfinance`) | Adjusted OHLCV for SPY, AAPL, MSFT, JPM, XOM, JNJ, 2010–2025 | Daily |
| [FRED](https://fred.stlouisfed.org) | 10Y & 2Y Treasury yields, 10Y–2Y spread, VIX | Daily |
| [FRED](https://fred.stlouisfed.org) | Fed funds rate, CPI, unemployment rate | Monthly |

The tickers cover the broad market plus four sectors (technology, financials, energy, health care), which gives a
categorical dimension to the analysis. Financial-news sentiment (FinBERT) is planned for Module 24.

## Methodology

1. **Data cleaning** — type conversion, duplicate and invalid-bar removal, missing-value profiling.
   Daily macro gaps (bond holidays) are forward-filled. Monthly macro values are shifted to their **publication date**
   (~6 weeks after the reference month) before an as-of merge, to prevent look-ahead bias.
2. **Outlier analysis** — z-score and IQR analysis of returns and volume, and a check for data errors (e.g. >40% daily
   moves from unadjusted splits). Genuine extreme days (crashes) are kept because they are the drawdowns the strategy
   must survive. For the linear model, features are winsorised at training-set 1st/99th percentiles.
3. **Feature engineering** — log returns (1/5/21 day, lags), EMA-12/26 distance, MACD and histogram (scaled by
   price), RSI-14, 21-day realised volatility, Bollinger width/position, high-low range, volume z-score, macro levels
   and changes, VIX regime, yield-curve inversion flag, day of week. Target = sign of the next day's log return.
4. **Exploratory analysis** — growth and drawdown curves, return distributions (histogram + Q-Q), ADF stationarity
   tests, seasonal decomposition, ACF of returns and squared returns, volatility vs VIX, macro panel, rolling
   stock/yield correlation, Spearman correlation heatmap, and regime/categorical comparisons.
5. **Baseline model** — a naive "always up" classifier versus a **logistic regression** (scaled numeric features +
   one-hot ticker/weekday, `C` tuned with `TimeSeriesSplit`). Train on 2010–2021, test on 2022 onwards, with a 5-day
   embargo between them. Predictions are also turned into a long/flat strategy with 5 bps transaction costs.

### Evaluation Metrics and Rationale

| Metric | Why |
|---|---|
| **Directional accuracy** | The metric in the problem statement (target > 55%); the share of days the sign is called correctly. |
| **Balanced accuracy** | Up days are slightly more common than down days, so plain accuracy flatters an "always up" model. |
| **ROC-AUC** | Measures ranking skill independent of the 0.5 threshold; 0.5 = no skill. |
| **Sharpe ratio, max drawdown** | The research question is about risk-adjusted return and avoiding severe drawdowns, so predictions are also judged economically. |

## Summary of Findings

### Exploratory data analysis

- **Prices are non-stationary; returns are stationary.** ADF tests cannot reject a unit root for price levels but
  strongly reject it for log returns, so all modelling uses returns and scale-free indicators.
- **Returns are fat-tailed.** Excess kurtosis is far above zero and many more days exceed |z| > 3 than a normal
  distribution predicts. These extremes match VIX spikes (e.g. the March 2020 COVID crash). They are real events,
  not data errors.
- **Direction has little memory, but volatility has a lot.** Returns show near-zero autocorrelation, while squared
  returns are strongly autocorrelated (volatility clustering). Risk is more forecastable than direction, which
  points to volatility-based position sizing as a drawdown-control tool.
- **Market regimes change the relationships.** High-VIX regimes are followed by much higher realised volatility, and
  the correlation between stock returns and Treasury-yield changes switches sign over time (e.g. around 2022). A
  static model is unlikely to capture this, which supports the regime-aware Temporal Fusion Transformer planned next.
- **Individual features are only weakly correlated with next-day returns**, as expected in liquid markets. This is
  the central difficulty of the research question.

### Baseline model

<!-- Fill these in from the summary cell at the end of Section 7 after running the notebook -->

| Model (test set, 2022+) | Directional accuracy | Balanced accuracy | ROC-AUC |
|---|---|---|---|
| Naive (always up) | _from run_ | 0.500 | 0.500 |
| Logistic regression | _from run_ | _from run_ | _from run_ |

| Strategy (test set, 2022+) | CAGR | Sharpe | Max drawdown |
|---|---|---|---|
| Buy & hold, equal weight | _from run_ | _from run_ | _from run_ |
| Logistic long/flat (5 bps costs) | _from run_ | _from run_ | _from run_ |

**Interpretation:** _[e.g. whether the logistic model beat the naive baseline and the 55% target, whether its AUC is
meaningfully above 0.5, and whether staying in cash on predicted-down days reduced the maximum drawdown.]_

## Next Steps (Module 24)

1. Add FinBERT sentiment scores from financial-news headlines.
2. Replace the single split with walk-forward (rolling-origin) evaluation.
3. Try non-linear models (gradient boosting) and the Temporal Fusion Transformer for multi-horizon quantile forecasts.
4. Add volatility-targeted position sizing, stop-losses and portfolio-level drawdown limits, then the RL execution agent.

## Repository Structure

```
├── README.md
├── requirements.txt
├── notebooks/
│   └── capstone_eda_baseline.ipynb   # EDA, feature engineering, baseline model
├── data/                             # created by the notebook (git-ignored)
│   ├── raw/                          # cached Yahoo Finance and FRED downloads
│   └── processed/                    # model-ready dataset
├── images/                           # figures saved by the notebook
└── results/                          # baseline metric tables (CSV)
```

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/capstone_eda_baseline.ipynb
```

The first run downloads the data (internet access required) and caches it in `data/raw/`. Later runs use the cache.
Delete the cached CSVs to refresh the data.
