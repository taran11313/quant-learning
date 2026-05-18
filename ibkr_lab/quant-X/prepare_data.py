"""
prepare_data.py
---------------
Pulls real OHLCV data from Yahoo Finance, computes 6 stationary features,
runs ADF stationarity tests, generates signals, and exports one JSON file
that the React dashboard reads.

Install dependencies (one time):
    pip install yfinance pandas numpy statsmodels

Run:
    python prepare_data.py

Output:
    trading_data.json  (place this in your React project's /public folder)
"""

import json
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import adfuller

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
TICKER   = "SPY"        # change to any ticker: AAPL, TSLA, QQQ, BTC-USD, etc.
PERIOD   = "2y"         # how far back to pull: 1y, 2y, 5y
INTERVAL = "1d"         # bar size: 1d, 1wk — use 1d for daily signals
# ──────────────────────────────────────────────────────────────────────────────


def fetch_data(ticker, period, interval):
    print(f"Fetching {ticker} ({period}, {interval} bars)...")
    df = yf.Ticker(ticker).history(period=period, interval=interval)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    df.index = df.index.strftime("%Y-%m-%d")
    print(f"  Got {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    return df


def adf_test(series, name):
    """
    Augmented Dickey-Fuller test.
    H0: series has a unit root (non-stationary).
    p < 0.05 → reject H0 → series IS stationary → safe to use as feature.
    """
    clean = series.dropna()
    result = adfuller(clean)
    stationary = bool(result[1] < 0.05)
    print(f"  {name:<25s} ADF={result[0]:7.3f}  p={result[1]:.4f}  {'✓ stationary' if stationary else '✗ NON-STATIONARY'}")
    return {
        "name":       name,
        "adf_stat":   round(float(result[0]), 4),
        "p_value":    round(float(result[1]), 4),
        "stationary": stationary,
    }


def build_features(df):
    close   = df["Close"]
    volume  = df["Volume"]
    returns = close.pct_change()

    # ── 6 STATIONARY FEATURES ─────────────────────────────────────────────────

    # 1. 1-day log return — mean-reverts around 0 across all regimes
    ret_1d = returns

    # 2. 5-day return — captures weekly momentum
    ret_5d = close.pct_change(5)

    # 3. Volatility ratio (short / long) — mean-reverts around 1
    #    > 1: volatility expanding (trending/momentum regime)
    #    < 1: volatility compressing (pre-breakout or mean-reversion setup)
    vol_5  = returns.rolling(5).std()
    vol_20 = returns.rolling(20).std()
    vol_ratio = vol_5 / vol_20

    # 4. Normalized momentum — return scaled by its own realized volatility
    #    = how many "vol units" did price move? Risk-adjusted momentum.
    momentum_norm = returns / (vol_20 + 1e-10)

    # 5. Volume z-score — volume relative to its 20-day distribution
    #    spike = institutional activity, trough = low conviction
    vol_mean = volume.rolling(20).mean()
    vol_std  = volume.rolling(20).std()
    volume_zscore = (volume - vol_mean) / (vol_std + 1e-10)

    # 6. SMA ratio — relative position of fast vs slow moving average
    #    0 = crossover point, + = bullish alignment, - = bearish alignment
    sma_5  = close.rolling(5).mean()
    sma_20 = close.rolling(20).mean()
    sma_ratio = sma_5 / sma_20 - 1

    features = pd.DataFrame({
        "return_1d":     ret_1d,
        "return_5d":     ret_5d,
        "vol_ratio":     vol_ratio,
        "momentum_norm": momentum_norm,
        "volume_zscore": volume_zscore,
        "sma_ratio":     sma_ratio,
    }, index=df.index)

    # ── TARGET: will tomorrow's return be positive? ────────────────────────────
    # Binary classification target — more stable than predicting raw return
    target = (returns.shift(-1) > 0).astype(int)

    features = features.dropna()
    target   = target.loc[features.index]

    return features, target


def generate_signals(features, target):
    """
    Rule-based signal combining all 6 features.
    Each feature votes +1 (bullish) or -1 (bearish).
    Final signal = sum of votes. Threshold at ±2 for strong signals.
    This is the framework a trained model replaces — but it gives
    you real, inspectable signals right now.
    """
    signals = pd.DataFrame(index=features.index)

    # Momentum vote: positive normalized momentum → bullish
    signals["vote_momentum"] = np.where(features["momentum_norm"] > 0, 1, -1)

    # Vol regime vote: vol compressing (ratio < 0.9) → potential breakout setup
    signals["vote_vol"] = np.where(features["vol_ratio"] < 0.9, 1,
                          np.where(features["vol_ratio"] > 1.1, -1, 0))

    # Trend vote: price above 20-day SMA → bullish
    signals["vote_trend"] = np.where(features["sma_ratio"] > 0, 1, -1)

    # Volume confirmation: high volume confirms direction
    signals["vote_volume"] = np.where(
        (features["volume_zscore"] > 1) & (features["momentum_norm"] > 0), 1,
        np.where((features["volume_zscore"] > 1) & (features["momentum_norm"] < 0), -1, 0)
    )

    # Short-term mean reversion: extreme negative 1d return → potential bounce
    signals["vote_reversion"] = np.where(features["return_1d"] < -0.02, 1,
                                 np.where(features["return_1d"] > 0.02, -1, 0))

    # Combined score: sum of all votes (-5 to +5)
    signals["score"] = signals[[c for c in signals.columns if c.startswith("vote_")]].sum(axis=1)

    # Final signal: strong threshold
    signals["signal"] = np.where(signals["score"] >= 2, "LONG",
                        np.where(signals["score"] <= -2, "SHORT", "NEUTRAL"))

    signals["actual_next_day"] = target
    return signals


def kelly_sizing(features, target):
    """
    Kelly criterion on the last 60 days (recent performance only).
    f* = (p * b - q) / b
    """
    recent_features = features.tail(60)
    recent_target   = target.loc[recent_features.index]

    # Use momentum_norm direction as the "model"
    pred_direction = (recent_features["momentum_norm"] > 0).astype(int)
    wins  = (pred_direction == recent_target).sum()
    total = len(recent_target)
    p     = wins / total
    q     = 1 - p

    # Approximate win/loss ratio from returns
    ret_1d = recent_features["return_1d"]
    avg_win  = ret_1d[ret_1d > 0].mean()
    avg_loss = abs(ret_1d[ret_1d < 0].mean())
    b = avg_win / (avg_loss + 1e-10)

    kelly_full = (p * b - q) / b
    kelly_half = max(0, min(0.25, kelly_full * 0.5))

    return {
        "win_rate":    round(float(p), 4),
        "win_loss_b":  round(float(b), 4),
        "kelly_full":  round(float(kelly_full), 4),
        "kelly_half":  round(float(kelly_half), 4),
        "max_position": "2% hard cap regardless of Kelly",
    }


def cumulative_returns(features, signals):
    """
    Compare strategy (using signal) vs buy-and-hold.
    Both start at 1.0.
    """
    ret_1d = features["return_1d"]
    sig    = signals["signal"].map({"LONG": 1, "SHORT": -1, "NEUTRAL": 0})

    strat_ret = sig * ret_1d * 0.1   # 10% position sizing (conservative)
    bh_ret    = ret_1d

    strat_cumret = (1 + strat_ret).cumprod()
    bh_cumret    = (1 + bh_ret).cumprod()

    # Sharpe ratio (annualized)
    excess = strat_ret - 0
    sharpe = float((excess.mean() / (excess.std() + 1e-10)) * np.sqrt(252))

    # Max drawdown
    rolling_max = strat_cumret.cummax()
    drawdown    = (strat_cumret - rolling_max) / (rolling_max + 1e-10)
    max_dd      = float(drawdown.min())

    return strat_cumret, bh_cumret, round(sharpe, 3), round(max_dd * 100, 2)


def main():
    # 1. Fetch
    df = fetch_data(TICKER, PERIOD, INTERVAL)

    # 2. Features
    print("\nBuilding features...")
    features, target = build_features(df)

    # 3. ADF tests
    print("\nStationarity tests:")
    raw_adf = adf_test(df["Close"].reindex(features.index), "raw_price (DO NOT USE)")
    adf_results = [raw_adf]
    for col in features.columns:
        adf_results.append(adf_test(features[col], col))

    # 4. Signals
    print("\nGenerating signals...")
    signals = generate_signals(features, target)

    # 5. Kelly
    kelly = kelly_sizing(features, target)
    print(f"\nKelly sizing (last 60 days):")
    print(f"  Win rate:   {kelly['win_rate']*100:.1f}%")
    print(f"  Kelly full: {kelly['kelly_full']*100:.1f}%")
    print(f"  Kelly half: {kelly['kelly_half']*100:.1f}%  ← deploy this")

    # 6. Returns
    strat_cumret, bh_cumret, sharpe, max_dd = cumulative_returns(features, signals)
    print(f"\nPerformance:")
    print(f"  Sharpe ratio: {sharpe}")
    print(f"  Max drawdown: {max_dd}%")

    # 7. Current signal (latest bar)
    latest      = signals.iloc[-1]
    latest_feat = features.iloc[-1]
    print(f"\nCurrent signal ({features.index[-1]}): {latest['signal']}  (score={latest['score']})")

    # 8. Build JSON payload
    # Limit chart data to last 252 bars (1 year) to keep file small
    tail = -252
    price_chart = [
        {"date": d, "close": round(float(df.loc[d, "Close"]), 2),
         "volume": int(df.loc[d, "Volume"])}
        for d in features.index[tail:]
        if d in df.index
    ]

    feature_chart = [
        {
            "date":          d,
            "return_1d":     round(float(features.loc[d, "return_1d"]) * 100, 4),
            "return_5d":     round(float(features.loc[d, "return_5d"]) * 100, 4),
            "vol_ratio":     round(float(features.loc[d, "vol_ratio"]), 4),
            "momentum_norm": round(float(features.loc[d, "momentum_norm"]), 4),
            "volume_zscore": round(float(features.loc[d, "volume_zscore"]), 4),
            "sma_ratio":     round(float(features.loc[d, "sma_ratio"]) * 100, 4),
            "signal":        signals.loc[d, "signal"],
            "score":         int(signals.loc[d, "score"]),
        }
        for d in features.index[tail:]
    ]

    returns_chart = [
        {
            "date":     d,
            "strategy": round(float(strat_cumret.loc[d]), 4),
            "buyHold":  round(float(bh_cumret.loc[d]), 4),
        }
        for d in features.index[tail:]
        if d in strat_cumret.index
    ]

    signal_votes_latest = {
        k: int(v) for k, v in latest.items()
        if k.startswith("vote_")
    }

    output = {
        "meta": {
            "ticker":       TICKER,
            "period":       PERIOD,
            "interval":     INTERVAL,
            "total_bars":   len(features),
            "from":         features.index[0],
            "to":           features.index[-1],
            "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "current_signal": {
            "date":         features.index[-1],
            "signal":       str(latest["signal"]),
            "score":        int(latest["score"]),
            "votes":        signal_votes_latest,
            "features": {
                "return_1d":     round(float(latest_feat["return_1d"]) * 100, 4),
                "return_5d":     round(float(latest_feat["return_5d"]) * 100, 4),
                "vol_ratio":     round(float(latest_feat["vol_ratio"]), 4),
                "momentum_norm": round(float(latest_feat["momentum_norm"]), 4),
                "volume_zscore": round(float(latest_feat["volume_zscore"]), 4),
                "sma_ratio":     round(float(latest_feat["sma_ratio"]) * 100, 4),
            }
        },
        "kelly":         kelly,
        "performance": {
            "sharpe":       sharpe,
            "max_drawdown": max_dd,
            "accuracy_proxy": round(float(kelly["win_rate"]) * 100, 1),
        },
        "stationarity":  adf_results,
        "price_chart":   price_chart,
        "feature_chart": feature_chart,
        "returns_chart": returns_chart,
    }

    out_path = "trading_data.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nExported → {out_path}  ({len(json.dumps(output)) // 1024} KB)")
    print("Next step: copy trading_data.json into your React project's /public folder.")


if __name__ == "__main__":
    main()
