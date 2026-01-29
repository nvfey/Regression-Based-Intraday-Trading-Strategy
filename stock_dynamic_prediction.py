import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import statsmodels.api as sm
import matplotlib.pyplot as plt

universe = [
    "NVDA","AMD","INTC","QCOM","AVGO","TXN","MU","NXPI",
    "ADI","MRVL","ON","STM","SWKS","MCHP","LRCX","AMAT",
    "KLAC","ASML","TSM","TER","WDC","STX",
    "LSCC","COHR","KLIC","NVTS","LITE","AMKR","UCTT","MTSI",
    "VSH","DIOD","UMC","GFS","SNPS","CDNS","ARM","SMTC",
    "TSEM","RMBS"
]

# =========================================================
# 1. DOWNLOAD RAW DATA FROM YAHOO
# =========================================================
raw = yf.download(
    universe,
    period="60d",
    interval="5m",
    group_by="ticker",
    progress=False
)

# Extract CLOSE prices per symbol
close_dict = {}

for sym in universe:
    try:
        df = raw[sym]["Close"].dropna()
        df.index = pd.to_datetime(df.index)
        close_dict[sym] = df
        print(f"{sym}: {len(df)} bars")
    except Exception:
        print(f"{sym}: NOT AVAILABLE — skipped")

if not close_dict:
    raise SystemExit("No usable symbols loaded from Yahoo 5m data.")

# MERGE RAW INTO MATRIX
data = pd.DataFrame(close_dict)

# STRICT INTERSECTION ALIGNMENT
# Step 1: Start with timestamps from first symbol
symbols = list(data.columns)
common_ts = data[symbols[0]].dropna().index

# Step 2: Intersect iteratively with timestamps of all other symbols
for sym in symbols[1:]:
    ts = data[sym].dropna().index
    common_ts = common_ts.intersection(ts)

# Step 3: Filter data to intersection
data = data.loc[common_ts].sort_index()

print(f"\nAfter strict intersection: {data.shape[0]} timestamps remain")

# RETURNS AND VOLATILITY
returns = np.log(data / data.shift(1)).dropna()
volatility = returns.rolling(30).std().ffill()

print(f"Final aligned dataset: {returns.shape[0]} rows x {returns.shape[1]} assets")

# =========================================================
# PARAMETERS
# =========================================================
WINDOW = 40
PCA_WINDOW = 280    # stable PCA
LAG = 1
TOPN = 15

MIN_T = 2.8
MAX_P = 0.01
CORR_TH = 0.2

PERSIST = 2
MIN_PRED = 0.0007

COOLDOWN = 5
fee = 0.0005


# =========================================================
# 3. STORAGE
# =========================================================
trade_returns = []
trade_times = []

signals = []
signal_times = []

pred_ret_series = []      # store predicted returns at time t
pred_times = []           # index for predicted returns

corr_history = []        # for correlation stability
resid_history = []       # optional if needed later

in_position = False
i = WINDOW

# =========================================================
# 4. MAIN LOOP
# =========================================================
while i < len(returns) - 20:

    t_now = returns.index[i]
    if in_position:
        i += 1
        continue

    # ---------------------------
    # 4.1 Lagged correlation window
    # ---------------------------
    win = returns.iloc[i-WINDOW:i]
    shifted = win.shift(LAG)
    aligned = win

    mask = (~shifted.isna().any(axis=1)) & (~aligned.isna().any(axis=1))
    shifted = shifted[mask]
    aligned = aligned[mask]

    if len(aligned) < 20:
        i += 1
        continue

    X = shifted.to_numpy()
    Y = aligned.to_numpy()
    X -= X.mean(0)
    Y -= Y.mean(0)
    cov = (X.T @ Y) / (len(X)-1)
    corr = cov / np.outer(X.std(0), Y.std(0))

    lagcorr = pd.DataFrame(corr, index=win.columns, columns=win.columns)

    # ---------------------------
    # 4.2 Target and peers selection
    # ---------------------------th
    predictability = lagcorr.apply(lambda col: col.drop(col.name).abs().nlargest(TOPN).mean())
    target = predictability.idxmax()

    peers = lagcorr[target].drop(target).abs().nlargest(TOPN).index.tolist()

    # ---------------------------
    # 4.3 PCA (stable window)
    # ---------------------------
    pca_raw = returns.iloc[i - PCA_WINDOW:i]
    # Drop columns (symbols) with NaNs, not rows
    pca_slice = pca_raw.dropna(axis=1)

    # Prevent empty PCA window
    if len(pca_slice) < 5:
        i += 1
        continue

    # Ensure all peers are present
    if not all(p in pca_slice.columns for p in peers):
        i += 1
        continue

    pca = PCA(n_components=1)
    factor_raw = pca.fit_transform(pca_slice[peers]).flatten()

    factor = pd.Series(factor_raw, index=pca_slice.index).ewm(span=5).mean()

    # stabilize direction
    if 'prev_factor' in locals():

        corr = np.corrcoef(prev_factor[-50:], factor[-50:])[0, 1]

        if corr < 0:
            factor = -factor

    prev_factor = factor.copy()

    # -----------------------------
    # 4.4 Regression (train on t-1)
    # -----------------------------
    Xreg = sm.add_constant(factor.iloc[:-1])
    yreg = pca_slice[target].iloc[:-1]
    model = sm.OLS(yreg, Xreg).fit()

    beta = model.params.iloc[1]
    tval = model.tvalues.iloc[1]
    pval = model.pvalues.iloc[1]
    # ---------------------------------
    # 4.5 Out-of-sample prediction at t
    # ---------------------------------
    factor_t = factor.iloc[-1]
    factor_prev = factor.iloc[-2]
    pred_ret = beta * (factor_t - factor_prev)

    # normalize predicted value
    pred_std = (factor.diff().rolling(50).std().iloc[-1])
    if pred_std == 0 or pd.isna(pred_std):
        i += 1
        continue

    z_pred = pred_ret / pred_std

    # Store predicted return series
    pred_ret_series.append(pred_ret)
    pred_times.append(t_now)

    # ----------------------------------
    # 4.6 Filters
    # ----------------------------------
    sig = np.sign(z_pred)

    # correlation filter
    corr_strength = lagcorr[target].drop(target).abs().nlargest(TOPN).mean()
    corr_history.append(corr_strength)
    corr_hist_series = pd.Series(corr_history)

    # correlation stability = current strength / recent mean
    if len(corr_hist_series) >= 5:
        corr_stability = corr_strength / corr_hist_series.tail(5).mean()
    else:
        corr_stability = 1

    # Residual-volatility expansion
    if len(model.resid) >= 10:
        res = model.resid
        recent_vol = res.iloc[-5:].std()
        past_vol = res.iloc[-10:-5].std()

        if recent_vol < past_vol:
            sig = 0

    # Predictability stability filter
    if corr_stability < 0.80:
        sig = 0

    if len(corr_hist_series) >= 6:
        corr_slope = corr_hist_series.diff().tail(5).mean()

        # allow small dips, block collapse
        if corr_slope < -0.03:
            sig = 0

    # t-stat filter
    if abs(tval) < MIN_T:
        sig = 0

    # noise filter
    noise_ratio = model.resid.std() / yreg.std()
    if noise_ratio > 2:
        sig = 0

    # min predicted move
    if abs(pred_ret) < MIN_PRED:
        sig = 0

    # persistence of predictions
    signals.append(sig)
    signal_times.append(t_now)

    # volatility regime filter
    low_q = volatility.iloc[:i].quantile(0.3)
    high_q = volatility.iloc[:i].quantile(0.8)

    vol = volatility.at[t_now, target]
    if vol > high_q[target]:
        sig = 0

    if len(signals) >= PERSIST:
        if np.sign(pd.Series(signals[-PERSIST:])).nunique() > 1:
            sig = 0

    if sig == 0:
        i += 1
        continue


    MIN_HOLD_BARS = 5  # never hold fewer than 5 bars (25 min)
    MAX_HOLD_BARS = 8  # never hold more than 8 bars (40 min)
    VOL_LOW = 0.005  # 0.5% volatility → long hold
    VOL_HIGH = 0.02  # 2% volatility → short hold
    v = float(vol)
    if v <= VOL_LOW:
        hold_period = MAX_HOLD_BARS  # calm market → hold longest
    elif v >= VOL_HIGH:
        hold_period = MIN_HOLD_BARS  # high volatility → exit faster
    else:
        # Linearly interpolate between MIN and MAX
        ratio = (v - VOL_LOW) / (VOL_HIGH - VOL_LOW)
        hold_period = int(MAX_HOLD_BARS - ratio * (MAX_HOLD_BARS - MIN_HOLD_BARS))
    # ---------------------------
    # 4.7 TRADE (enter next bar)
    # ---------------------------
    entry_t = returns.index[i+1]
    entry_price = data.at[entry_t, target]
    direction = sig
    in_position = True

    idx0 = data.index.get_loc(entry_t)
    seg = data[target].iloc[idx0:idx0+hold_period+1]

    if len(seg) < hold_period + 1:
        break

    path = seg/entry_price - 1

    SL = 1.0 * vol
    TP = 3.5 * vol
    exit_ret = None

    for k in range(1, hold_period + 1):
        close_k = data[target].iloc[idx0 + k]
        position_ret = direction * (close_k / entry_price - 1)

        if position_ret >= TP:
            exit_ret = TP
            break
        if position_ret <= -SL:
            exit_ret = -SL
            break

    if exit_ret is None:
        exit_ret = data[target].iloc[idx0 + hold_period] / entry_price - 1

    final_ret = exit_ret * direction
    final_ret = final_ret - 2 * fee
    trade_returns.append(final_ret)
    trade_times.append(entry_t)

    in_position = False
    i += hold_period + COOLDOWN

# ===========
# 5. SUMMARY
# ===========
strategy = pd.Series(trade_returns, index=trade_times)
cum = (1 + strategy).cumprod()
signals_series = pd.Series(signals, index=signal_times)
pred_ret_series = pd.Series(pred_ret_series, index=pred_times, name="PredictedReturn")

# ==================
# BACKTEST SUMMARY
# ==================

strategy_ret = strategy

# ----- Win rate -----
wins = (strategy_ret > 0).sum()
losses = (strategy_ret < 0).sum()
winrate = wins / (wins + losses) if (wins + losses) > 0 else np.nan

# ----- Total return -----
total_ret = cum.iloc[-1] - 1

# ----- Capital curve -----
initial_capital = 100_000
capital_curve = [initial_capital]
for r in strategy_ret:
    capital_curve.append(capital_curve[-1] * (1 + r))
capital_series = pd.Series(capital_curve[1:], index=strategy_ret.index)

# ----- Sharpe per trade -----
if strategy_ret.std() > 0:
    sharpe_trade = strategy_ret.mean() / strategy_ret.std()
else:
    sharpe_trade = np.nan

# CAGR and annualized Sharpe
days = (strategy_ret.index[-1] - strategy_ret.index[0]).days
years = days / 365

if years > 0:
    cagr = (cum.iloc[-1])**(1/years) - 1
else:
    cagr = np.nan

# Volatility-based Sharpe using daily returns
capital_daily = capital_series.resample('1D').last().pct_change(fill_method=None).dropna()

if len(capital_daily) > 2:
    vol_annual = capital_daily.std() * np.sqrt(252)
    sharpe_from_cagr = cagr / vol_annual
else:
    sharpe_from_cagr = np.nan

# ----- Max Drawdown -----
rolling_max = cum.cummax()
max_dd = ((cum - rolling_max) / rolling_max).min()

# ----- Stability (fraction of rolling mean > 0) -----
stab = (strategy_ret.rolling(30).mean().dropna() > 0).mean()

# ==============
# PRINT SUMMARY
# ==============

print("\nBacktest Summary")
print(f"Trades: {len(strategy_ret)}")
print(f"Win rate: {winrate:.2%}")
print(f"Total return: {total_ret:.2%}")
print(f"Final capital: ${capital_series.iloc[-1]:,.2f}")
print(f"Sharpe per trade: {sharpe_trade:.2f}")
print(f"Sharpe (CAGR-based): {sharpe_from_cagr:.2f}")
print(f"Annualized return CAGR: {cagr:.2%}")
print(f"Max Drawdown: {max_dd:.2%}")
print(f"Stability: {stab:.2%}")

# ==============
# VISUALIZATION
# ==============

# ----- Equity curve -----
plt.figure(figsize=(12,5))
plt.plot(cum.index, cum)

for t, r in strategy.items():
    if r > 0:
        plt.annotate('▲', xy=(t, cum.loc[t]), color='green', ha='center')
    else:
        plt.annotate('▼', xy=(t, cum.loc[t]), color='red', ha='center')

plt.title("Trade Direction Markers on Equity Curve")
plt.grid(True)
plt.show()

# ----- Trade return histogram -----
plt.figure(figsize=(8,4))
plt.hist(strategy_ret, bins=50, edgecolor='k')
plt.title("Distribution of Per-Trade Returns")
plt.xlabel("Return")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# ----- Rolling Sharpe -----
rolling_sharpe = strategy_ret.rolling(20).mean() / strategy_ret.rolling(20).std()
plt.figure(figsize=(10,4))
plt.plot(rolling_sharpe.index, rolling_sharpe, label="Rolling Sharpe (20 trades)")
plt.axhline(0, color='red', linestyle='--')
plt.title("Rolling Sharpe Ratio")
plt.legend()
plt.grid(True)
plt.show()

# ----- Signal activity rate -----
if len(signals_series) > 100:
    plt.figure(figsize=(10,4))
    plt.plot(signals_series.index,
             (signals_series != 0).rolling(100).mean()*100,
             label="Active Trade Frequency (%)")
    plt.title("Signal Activity Rate (rolling 100 bars)")
    plt.legend()
    plt.grid(True)
    plt.show()
