"""
backtest.py
-----------
Configurable historical backtest engine with Stop Loss logic.
"""

import pandas as pd
import yfinance as yf
import numpy as np

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
TICKER = "IREN"
PERIOD = "5y"

# Strategy Toggle: "MEAN_REVERSION_LONG" or "MOMENTUM_SHORT"
# As per instructions, we are shifting to a momentum continuation strategy for IREN
STRATEGY_TYPE = "MOMENTUM_SHORT"

# Thresholds configured to increase sample size (50-100 signals)
if STRATEGY_TYPE == "MOMENTUM_SHORT":
    RET_THRESH = -0.06         # Drop of > 6% (relaxed from 10%)
    VOL_RATIO_THRESH = 1.5     # Volatility expanding (relaxed from 1.5)
    VOL_Z_THRESH = -0.2        # Volume Z-score > -0.2 (relaxed from 1.0)
else:
    # Relaxed Mean Reversion Long
    RET_THRESH = -0.04
    VOL_RATIO_THRESH = 0.9     
    VOL_Z_THRESH = -0.2        

# Step 3: Implement Stop Loss
# Hard rule: If price moves against the signal close, cut the trade immediately.
USE_STOP_LOSS = True
# ──────────────────────────────────────────────────────────────────────────────

def run_historical_backtest(ticker, period):
    print(f"Fetching historical OHLCV data for {ticker} ({period})...")
    df = yf.Ticker(ticker).history(period=period, interval="1d")
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    df.index = df.index.strftime("%Y-%m-%d")

    close = df["Close"]
    volume = df["Volume"]
    returns = close.pct_change()

    # Features
    ret_1d = returns
    vol_5 = returns.rolling(5).std()
    vol_20 = returns.rolling(20).std()
    vol_ratio = vol_5 / (vol_20 + 1e-10)

    vol_mean = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std()
    volume_zscore = (volume - vol_mean) / (vol_std + 1e-10)

    data = pd.DataFrame({
        "open": df["Open"],
        "high": df["High"],
        "low": df["Low"],
        "close": df["Close"],
        "return_1d": ret_1d,
        "vol_ratio": vol_ratio,
        "volume_zscore": volume_zscore,
    }, index=df.index).dropna()

    # Shift next day's price data to current day for easy evaluation
    data["next_open"] = data["open"].shift(-1)
    data["next_high"] = data["high"].shift(-1)
    data["next_low"] = data["low"].shift(-1)
    data["next_close"] = data["close"].shift(-1)
    
    data = data.dropna()

    # Apply conditions based on Strategy Type
    if STRATEGY_TYPE == "MOMENTUM_SHORT":
        # Expecting continuation of the drop
        condition = (
            (data["return_1d"] < RET_THRESH) &
            (data["vol_ratio"] > VOL_RATIO_THRESH) &
            (data["volume_zscore"] > VOL_Z_THRESH)
        )
    else:
        # Expecting mean reversion bounce
        condition = (
            (data["return_1d"] < RET_THRESH) &
            (data["vol_ratio"] < VOL_RATIO_THRESH) &
            (data["volume_zscore"] < VOL_Z_THRESH)
        )

    matches = data[condition].copy()
    total_matches = len(matches)

    if total_matches == 0:
        print(f"\nNo signals found for {STRATEGY_TYPE} with current thresholds.")
        return

    # Evaluate each match with Stop Loss logic
    actual_returns = []
    wins = 0

    for idx, row in matches.iterrows():
        signal_close = row["close"]
        next_open = row["next_open"]
        next_high = row["next_high"]
        next_low = row["next_low"]
        next_close = row["next_close"]
        
        trade_return = 0.0

        if STRATEGY_TYPE == "MEAN_REVERSION_LONG":
            # Target: price goes UP
            if USE_STOP_LOSS:
                stop_price = signal_close 
                if next_open <= stop_price:
                    # Gapped down below stop, exit at open
                    trade_return = (next_open - signal_close) / signal_close
                elif next_low <= stop_price:
                    # Dropped below stop intraday, exit at exactly the stop price
                    trade_return = (stop_price - signal_close) / signal_close
                else:
                    # Survived the day, exit at close
                    trade_return = (next_close - signal_close) / signal_close
            else:
                trade_return = (next_close - signal_close) / signal_close
                
            if trade_return > 0:
                wins += 1

        elif STRATEGY_TYPE == "MOMENTUM_SHORT":
            # Target: price goes DOWN
            if USE_STOP_LOSS:
                stop_price = signal_close
                if next_open >= stop_price:
                    # Gapped up above stop, exit at open
                    trade_return = (signal_close - next_open) / signal_close
                elif next_high >= stop_price:
                    # Rallied above stop intraday, exit at exactly the stop price
                    trade_return = (signal_close - stop_price) / signal_close
                else:
                    # Survived the day, exit at close
                    trade_return = (signal_close - next_close) / signal_close
            else:
                trade_return = (signal_close - next_close) / signal_close
                
            if trade_return > 0:
                wins += 1
                
        actual_returns.append(trade_return)

    matches["actual_trade_return"] = actual_returns
    
    win_rate = (wins / total_matches) * 100
    avg_return = np.mean(actual_returns) * 100

    print(f"\n" + "="*70)
    print(f" BACKTEST RESULTS: {STRATEGY_TYPE} on {ticker} ({period})")
    print(f" Thresholds: Drop > {abs(RET_THRESH*100):.1f}%, Vol Ratio {'>' if STRATEGY_TYPE=='MOMENTUM_SHORT' else '<'} {VOL_RATIO_THRESH}, Vol Z {'>' if STRATEGY_TYPE=='MOMENTUM_SHORT' else '<'} {VOL_Z_THRESH}")
    print(f" Stop Loss Enabled: {USE_STOP_LOSS}")
    print("="*70)
    print(f"Total trading days analyzed: {len(data)}")
    print(f"Signals generated:           {total_matches}")
    print(f"Winning trades:              {wins} ({win_rate:.1f}%)")
    print(f"Average Trade Return:        {avg_return:+.2f}%\n")

    print("Detailed Historical Trades (Last 15):")
    print(f"{'Date':<12} | {'1d Drop':<10} | {'Vol Ratio':<10} | {'Vol Z-score':<12} | {'Trade Ret':<12}")
    print("-" * 65)
    for idx, row in matches.tail(15).iterrows():
        ret_str = f"{row['return_1d']*100:.2f}%"
        trade_ret_str = f"{row['actual_trade_return']*100:+.2f}%"
        print(f"{idx:<12} | {ret_str:<10} | {row['vol_ratio']:<10.2f} | {row['volume_zscore']:<12.2f} | {trade_ret_str:<12}")


def backtest_dip_long_momentum(df, features):
    """
    Tests: Buy the dip when there is a massive, high-volume crash.
    """
    df_backtest = features[['return_1d', 'vol_ratio', 'volume_zscore']].copy()
    price_series = df['Close']
    
    # Target: Massive drops with high volume (capitulation)
    cond1 = df_backtest['return_1d'] < -0.06  # Drop > 6%
    cond2 = df_backtest['vol_ratio'] > 1.2    # Volatility expanding
    cond3 = df_backtest['volume_zscore'] > 1.0 # Massive volume spike
    
    signal_days = df_backtest[cond1 & cond2 & cond3]
    print(f"High Volume Capitulation Signals found: {len(signal_days)}")
    
    # Check next day return (LONG trade - buying the dip)
    next_day_returns = price_series.pct_change().shift(-1).loc[signal_days.index]
    short_wins = (next_day_returns < 0).sum()
    
    # A winning trade is when the next day's return is POSITIVE
    wins = (next_day_returns > 0).sum()
    total = len(next_day_returns)
    win_rate = wins / total if total > 0 else 0
    
    print(f"long Wins (Next day positive): {wins}")
    print(f"short Wins (Next day negative): {short_wins}")

    print(f"long Win Rate: {(next_day_returns > 0).mean():.2%}")
    print(f"short Win Rate: {(next_day_returns < 0).mean():.2%}")
    # print(f"Average Next-Day Return: {next_day_returns.mean():.4%}")
    return win_rate



if __name__ == "__main__":
    # Fetch data and generate features so we can pass them to the function
    print(f"Fetching data for {TICKER} ({PERIOD})...")
    df = yf.Ticker(TICKER).history(period=PERIOD, interval="1d")
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    df.index = df.index.strftime("%Y-%m-%d")

    # Compute features required by the function
    returns = df["Close"].pct_change()
    vol_5 = returns.rolling(5).std()
    vol_20 = returns.rolling(20).std()
    
    vol_mean = df["Volume"].rolling(20).mean()
    vol_std = df["Volume"].rolling(20).std()

    features = pd.DataFrame({
        "return_1d": returns,
        "vol_ratio": vol_5 / (vol_20 + 1e-10),
        "volume_zscore": (df["Volume"] - vol_mean) / (vol_std + 1e-10)
    }, index=df.index).dropna()
    
    # Align df to match the dropped NA rows in features
    df = df.loc[features.index]

    print(f"\n--- Running Dip Long Momentum Backtest on {TICKER} ---")
    backtest_dip_long_momentum(df, features)
