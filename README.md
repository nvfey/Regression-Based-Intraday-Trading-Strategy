Regression-Based Intraday Trading Strategy

Overview

  This project explores an intraday trading strategy based on lagged cross-asset correlations and regression on latent factors.
  The strategy is designed for groups of highly correlated stocks, where short-term lead–lag relationships may exist.
  
  The goal of the project is exploratory and methodological: to construct signals using econometric tools, to evaluate them via backtesting, and to understand the gap between predictive modeling and executable trading performance.
    
  This is more a learning and research-oriented project, not a production trading system.

Data

  • Market: US equities
  
  • Universe size: ~40 stocks
  
  • Selection logic: stocks from the same industry (e.g. semiconductors), chosen to ensure strong cross-sectional correlation
  
  • Frequency: 5-minute bars
  
  • History: ~60 trading days
  
  • Source: Yahoo Finance (yfinance)
  
  All price series are aligned using strict timestamp intersection.
  No forward-filling is applied in order to avoid artificial synchronization. 
  
  The number of stocks and days for testing are limited because of the limited amount of free data available. I did not consider it necessary to implement a high-quality paid data source, as this is more of an learning project. However, it is possible to use much more data, which should provide a more accurate evaluation of the entire algorithm.

Strategy Logic
  1. Lagged Correlation Analysis
  
  At each step, a rolling window of returns is used to compute lagged correlations:
  
   • Window length: WINDOW = 40 bars (~200 minutes)
  
   • Lag: LAG = 1 bar (5 minutes)
  
  Rationale:
  
  A window of 40 bars is long enough to estimate meaningful correlations, while still short enough to:
  
   • allow correlations to evolve intraday,
  
   • enable frequent changes in target and peer selection,
  
   • avoid enforcing long-term relationships in a short-horizon strategy.
  
  For each asset:
  
   • the average absolute lagged correlation with other assets is computed,
  
   • the asset with the highest predictability is selected as the target,
  
   • its TOPN = 15 most correlated peers are used for factor construction.

   
