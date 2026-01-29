Regression-Based Intraday Trading Strategy

 Overview

  This project explores an intraday trading strategy based on lagged cross-asset correlations and regression on latent factors.
  The strategy is designed for groups of highly correlated stocks, where short-term lead–lag relationships may exist.
  
  The goal of the project is exploratory and methodological: to construct signals using econometric tools, to evaluate them via backtesting, and to understand the gap between predictive modeling and executable trading performance.
  This is more a learning and research-oriented project, not a production trading system.
  
 Data

•	Market: US equities

•	Universe size: ~40 stocks

•	Selection logic: stocks from the same industry (e.g. semiconductors), chosen to ensure strong cross-sectional correlation

•	Frequency: 5-minute bars

•	History: ~60 trading days

•	Source: Yahoo Finance (yfinance)

All price series are aligned using strict timestamp intersection.
No forward-filling is applied in order to avoid artificial synchronization.

The number of stocks and days for bcaktesting are limited because of the limited amount of free data available. I did not consider it necessary to implement a high-quality paid data source, as this is more of a learning project. However, it is possible to use much more data, which should provide a more accurate evaluation of the entire algorithm.

Strategy Logic

1. Lagged Correlation Analysis
   
 At each step, a rolling window of returns is used to compute lagged correlations:

•	Window length: WINDOW = 40 bars (~200 minutes)

•	Lag: LAG = 1 bar (5 minutes)

Rationale:

A window of 40 bars is long enough to estimate meaningful correlations, while still short enough to:

•	allow correlations to evolve intraday,

•	enable frequent changes in target and peer selection,

•	avoid enforcing long-term relationships in a short-horizon strategy.

For each asset:

•	the average absolute lagged correlation with other assets is computed,

•	the asset with the highest predictability is selected as the target,

•	its TOPN = 15 most correlated peers are used for factor construction.

2. Factor Construction (PCA)
   
To summarize the joint short-term behavior of the selected peer assets, a custom peer index is constructed using principal component analysis (PCA).

•	Number of peers: TOPN = 15

•	PCA estimation window: 280 bars (~23 hours)

•	Output: one time-series index representing common peer movement

Construction logic:

At each step, PCA is applied only to the returns of the selected peer assets (those with the strongest lagged correlations to the target).
The method is used solely as a dimensionality-reduction tool to compress multiple correlated return series into a single index capturing their dominant shared dynamics.
This constructed index is then:

•	smoothed using a short exponential moving average,

•	direction-stabilized over time to prevent sign flips,

•	and used as the sole explanatory variable in the regression model for the target asset.

Rationale:

Using a PCA-based index allows:

•	dimensionality reduction from multiple correlated peers to a single factor,

•	preservation of the dominant cross-sectional structure,

•	and a clear econometric interpretation in the regression stage.

The PCA window is intentionally much longer than the correlation window to:

•	produce a more stable index,

•	reduce sensitivity to transient correlations,

•	and decouple factor estimation from short-term signal generation.

3. Regression Model
 
An OLS regression is estimated:

Target Return ~ Hidden Factor

The model is trained on data up to time t−1 and evaluated out-of-sample at time t.

Explicitly monitored statistics:

•	coefficient estimate (beta)

•	t-statistic

•	p-value

•	residual volatility

These statistics are later used as filters, not as optimization targets.

4. Prediction & Normalization

The predicted return is computed as the change in factor exposure between consecutive bars:

predicted return ≈ beta × (factor_t − factor_{t−1})

Predictions are normalized by recent factor volatility:

•	rolling window: 50 bars

This normalization helps distinguish meaningful signals from noise across volatility regimes.

5. Signal Filters

A signal is allowed only if multiple conditions are met:

•	Statistical significance:
|t-stat| ≥ 2.8

•	Minimum predicted move:
|predicted return| ≥ 0.0007

•	Correlation stability:
current correlation strength must not collapse relative to recent history

•	Residual behavior:
signals are suppressed when residual volatility contracts, indicating reduced explanatory power

•	Noise filter:
residual standard deviation relative to target volatility must remain bounded

•	Signal persistence:
prediction direction must be stable for PERSIST = 2 consecutive evaluations

Rationale:

These filters aim to reduce:

•	unstable relationships,

•	spurious correlations,

•	short-lived noise-driven signals.

They are not tuned for profitability, but for statistical plausibility.

6. Volatility Regime Filter

Target asset volatility is evaluated relative to its own history:

•	low-volatility regime: lower 30% quantile

•	high-volatility regime: upper 80% quantile

Signals are suppressed in extreme volatility regimes where execution risk dominates predictability.

7. Trade Execution (Backtest)
    
•	Position type: long or short (based on signal sign)

•	Positions: single position at a time (no overlap)

•	Holding period: dynamically chosen between 5 and 8 bars
(~25–40 minutes), depending on current volatility

Dynamic holding logic:

•	lower volatility → longer hold

•	higher volatility → shorter hold

8. Risk Management & Costs
   
•	Stop-loss: 1.0 × volatility

•	Take-profit: 3.5 × volatility

•	Transaction cost: 0.0005 per trade (used as a proxy for slippage)

Transaction costs are deducted symmetrically at entry and exit and applied proportionally to returns.

Backtesting Framework

The backtest enforces:

•	strict chronological execution,

•	no overlapping trades,

•	explicit capital evolution from an initial balance.

Reported metrics include:

•	per-trade returns,

•	cumulative equity curve,

•	win rate,

•	Sharpe ratios (trade-based and annualized),

•	maximum drawdown,

•	rolling performance stability.

 Limitations

This project has several known limitations:

•	short data history (free data constraint),

•	Yahoo Finance data quality,

•	no detailed order-book or slippage modeling,

•	no survivorship bias adjustment,

•	potential overfitting due to repeated evaluation.

These limitations are acknowledged and treated as part of the learning process.
