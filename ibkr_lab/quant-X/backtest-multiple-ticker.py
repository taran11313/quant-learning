import yfinance as yf
import pandas as pd
import numpy as np

# ── 1. Define the ticker basket ──────────────────────────────────────────────
TICKERS = ["IREN", "MARA", "RIOT", "CLSK", "HUT", "CIFR"]

# ── 2. Define the backtest logic (reusable function) ────────────────────────
def backtest_dip_long(ticker, period="5y", interval="1d"):
    """
    Runs the dip-buy backtest on a single ticker and returns the trade results.
    """
    # ── Fetch data ────────────────────────────────────────────────────────────
    df = yf.Ticker(ticker).history(period=period, interval=interval)
    df = df[["Close", "Volume"]].copy()
    df.dropna(inplace=True)
    
    # ── Build features ────────────────────────────────────────────────────────
    df['return_1d'] = df['Close'].pct_change()
    df['vol_5'] = df['return_1d'].rolling(5).std()
    df['vol_20'] = df['return_1d'].rolling(20).std()
    df['vol_ratio'] = df['vol_5'] / df['vol_20']
    df['vol_mean'] = df['Volume'].rolling(20).mean()
    df['vol_std'] = df['Volume'].rolling(20).std()
    df['volume_zscore'] = (df['Volume'] - df['vol_mean']) / df['vol_std']
    df.dropna(inplace=True)
    
    # ── Apply signal conditions ──────────────────────────────────────────────
    cond1 = df['return_1d'] < -0.08   # Drop > 8%
    cond2 = df['vol_ratio'] > 1.4     # Volatility expanding
    cond3 = df['volume_zscore'] > 1.5 # Volume spike (>1.5 std)
    
    signal_df = df[cond1 & cond2 & cond3].copy()
    
    if len(signal_df) == 0:
        return pd.DataFrame()  # No signals for this ticker
    
    # ── Compute next day returns ─────────────────────────────────────────────
    signal_df['next_return'] = df['Close'].pct_change().shift(-1).loc[signal_df.index]
    
    # ── Store results ─────────────────────────────────────────────────────────
    signal_df['ticker'] = ticker
    signal_df['win'] = signal_df['next_return'] > 0
    signal_df['loss'] = signal_df['next_return'] < 0
    return signal_df[['ticker', 'return_1d', 'vol_ratio', 'volume_zscore', 'next_return', 'win', 'loss']]

# ── 3. Loop through all tickers and pool the results ─────────────────────────
all_trades = pd.DataFrame()

for ticker in TICKERS:
    print(f"Processing {ticker}...")
    trades = backtest_dip_long(ticker)
    all_trades = pd.concat([all_trades, trades])
    print(f"  → Found {len(trades)} signals for {ticker}")

# ── 4. Final aggregated statistics ─────────────────────────────────────────────
total_signals = len(all_trades)
long_wins = all_trades['win'].sum()
short_wins = all_trades['loss'].sum()
long_win_rate = long_wins / total_signals if total_signals > 0 else 0
short_win_rate = short_wins / total_signals if total_signals > 0 else 0
avg_return = all_trades['next_return'].mean()

print("\n" + "="*60)
print("POOLED BACKTEST RESULTS (All Mining Tickers)")
print("="*60)
print(f"Total signals across all tickers: {total_signals}")
print(f"LONG Wins (Next day positive):     {long_wins}")
print(f"SHORT Wins (Next day negative):    {short_wins}")
print(f"LONG Win Rate:                     {long_win_rate:.2%}")
print(f"SHORT Win Rate:                    {short_win_rate:.2%}")
print(f"Average Next-Day Return:           {avg_return:.4%}")
print("="*60)

# 1. Calculate the PnL for LONG and SHORT strategies
all_trades['long_pnl'] = all_trades['next_return']  # Profit if you go LONG
all_trades['short_pnl'] = -all_trades['next_return'] # Profit if you go SHORT

# 2. Calculate average PnL for each strategy across all tickers
avg_long_pnl = all_trades['long_pnl'].mean()
avg_short_pnl = all_trades['short_pnl'].mean()

print("\n" + "="*60)
print("EXPECTED VALUE (EV) PER TRADE")
print("="*60)
print(f"Average LONG PnL (per trade):   {avg_long_pnl:.4%}")
print(f"Average SHORT PnL (per trade):  {avg_short_pnl:.4%}")
print("="*60)

if avg_short_pnl > avg_long_pnl:
    print(f"🎯 Recommendation: Strategy has a positive SHORT bias. Consider a Short-Biased strategy (i.e., size your shorts larger than your longs).")
    print(f"   Expected PnL per trade: {avg_short_pnl:.4%}")
else:
    print(f"🎯 Recommendation: Strategy has a positive LONG bias. Consider a Long-Biased strategy.")
    print(f"   Expected PnL per trade: {avg_long_pnl:.4%}")





# ── 6. Main execution function ─────────────────────────────────────────────────
# def main():
#     print("Starting Pooled Backtest...")
    
#     # Define your ticker basket
#     TICKERS = ["IREN", "MARA", "RIOT", "CLSK", "HUT", "CIFR"]
    
#     all_trades = pd.DataFrame()
    
#     for ticker in TICKERS:
#         print(f"Processing {ticker}...")
#         trades = backtest_dip_long(ticker)
#         all_trades = pd.concat([all_trades, trades])
#         print(f"  → Found {len(trades)} signals for {ticker}")
    
#     total_signals = len(all_trades)
#     if total_signals == 0:
#         print("No signals found across any ticker.")
#         return
    
#     long_wins = all_trades['win'].sum()
#     short_wins = all_trades['loss'].sum()
#     long_win_rate = long_wins / total_signals
#     short_win_rate = short_wins / total_signals
#     avg_return = all_trades['next_return'].mean()
    
#     print("\n" + "="*60)
#     print("POOLED BACKTEST RESULTS (All Mining Tickers)")
#     print("="*60)
#     print(f"Total signals across all tickers: {total_signals}")
#     print(f"LONG Wins (Next day positive):     {long_wins}")
#     print(f"SHORT Wins (Next day negative):    {short_wins}")
#     print(f"LONG Win Rate:                     {long_win_rate:.2%}")
#     print(f"SHORT Win Rate:                    {short_win_rate:.2%}")
#     print(f"Average Next-Day Return:           {avg_return:.4%}")
#     print("="*60)
    
#     # Optional: Save to CSV
#     # all_trades.to_csv("pooled_trades.csv", index=True)

# # ── 7. Call the main function ──────────────────────────────────────────────────
# if __name__ == "__main__":
#     main()    