"""
prepare_data.py
---------------
Pulls real OHLCV data from Yahoo Finance, computes 6 stationary features,
runs ADF stationarity tests, trains an LSTM on sequential data,
generates predictions, and exports one JSON file for the React dashboard.

Install dependencies (one time):
    pip install yfinance pandas numpy statsmodels torch scikit-learn

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
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
TICKER   = "IREN"        # change to any ticker: AAPL, TSLA, QQQ, BTC-USD, etc.
PERIOD   = "2y"         # how far back to pull: 1y, 2y, 5y
INTERVAL = "1d"         # bar size: 1d, 1wk — use 1d for daily signals
LOOKBACK = 20           # number of past days to feed into LSTM
HIDDEN_SIZE = 64        # LSTM hidden units
NUM_LAYERS = 2          # LSTM layers
DROPOUT = 0.2           # dropout rate
EPOCHS = 100            # max training epochs
LEARNING_RATE = 0.001   # optimizer learning rate
PATIENCE = 10           # early stopping patience
BATCH_SIZE = 16         # training batch size
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
    vol_5  = returns.rolling(5).std()
    vol_20 = returns.rolling(20).std()
    vol_ratio = vol_5 / vol_20

    # 4. Normalized momentum — return scaled by its own realized volatility
    momentum_norm = returns / (vol_20 + 1e-10)

    # 5. Volume z-score — volume relative to its 20-day distribution
    vol_mean = volume.rolling(20).mean()
    vol_std  = volume.rolling(20).std()
    volume_zscore = (volume - vol_mean) / (vol_std + 1e-10)

    # 6. SMA ratio — relative position of fast vs slow moving average
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
    target = (returns.shift(-1) > 0).astype(int)

    features = features.dropna()
    target   = target.loc[features.index]

    return features, target


def prepare_sequences(features, target, lookback=LOOKBACK):
    """
    Convert features and target into sequences for LSTM.
    Each sample: (lookback, num_features) → next day's direction.
    """
    X, y = [], []
    feature_array = features.values
    target_array = target.values
    for i in range(len(feature_array) - lookback):
        X.append(feature_array[i:i+lookback])
        y.append(target_array[i+lookback])
    return np.array(X), np.array(y)


# ── LSTM MODEL DEFINITION (from the article) ────────────────────────────────
class TradingLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super(TradingLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        # Take the output from the last time step
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return self.sigmoid(out).squeeze()


# ── TRAINING LOOP WITH EARLY STOPPING ────────────────────────────────────────
def train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE, patience=PATIENCE):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float('inf')
    best_weights = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}  |  Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}")

    model.load_state_dict(best_weights)
    return model


def generate_predictions(features, target, lookback=LOOKBACK):
    """
    Train LSTM and generate predictions for the entire dataset.
    Returns: predictions (probability of upward move), signal (LONG/SHORT/NEUTRAL),
             and the trained model.
    """
    # ── Prepare sequences ────────────────────────────────────────────────────
    X, y = prepare_sequences(features, target, lookback)

    # ── Split: 60% train, 20% validation, 20% test ─────────────────────────
    n = len(X)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    # ── Scale features ──────────────────────────────────────────────────────
    # Note: We scale each feature independently across the time dimension
    # We fit scaler on training data only, then transform val/test
    scaler = StandardScaler()
    # Reshape to (n_samples * lookback, n_features) for fitting
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    scaler.fit(X_train_flat)

    def scale_sequences(X_data):
        shape = X_data.shape
        flat = X_data.reshape(-1, shape[-1])
        scaled = scaler.transform(flat)
        return scaled.reshape(shape)

    X_train = scale_sequences(X_train)
    X_val = scale_sequences(X_val)
    X_test = scale_sequences(X_test)

    # ── Convert to PyTorch tensors ──────────────────────────────────────────
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test)

    # ── Create DataLoaders ──────────────────────────────────────────────────
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ── Initialize and train model ──────────────────────────────────────────
    input_size = X_train.shape[-1]
    model = TradingLSTM(input_size=input_size)
    print("\nTraining LSTM...")
    model = train_model(model, train_loader, val_loader)

    # ── Evaluate on test set ────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_t)
        test_labels = y_test_t
        test_accuracy = ((test_preds > 0.5).float() == test_labels).float().mean().item()
        print(f"  Test accuracy: {test_accuracy:.4f}")

    # ── Generate predictions for ALL sequences (including train/val) ──────
    # We'll predict on the full dataset using the trained model
    X_full_t = torch.FloatTensor(scale_sequences(X))
    model.eval()
    with torch.no_grad():
        full_preds = model(X_full_t).numpy()

    # ── Convert to signals ──────────────────────────────────────────────────
    # We need to align predictions with the original index
    # The first `lookback` days have no prediction (no sequence)
    # We insert NaNs for those days
    pred_series = np.full(len(features), np.nan)
    pred_series[lookback:] = full_preds

    # Binary signal: > 0.6 → LONG, < 0.4 → SHORT, else NEUTRAL
    signal_series = np.full(len(features), "NEUTRAL", dtype=object)
    signal_series[pred_series > 0.6] = "LONG"
    signal_series[pred_series < 0.4] = "SHORT"

    # Score: scaled prediction to -5..5 range for consistency with original output
    score_series = (pred_series - 0.5) * 10  # 0.5 → 0, 1.0 → 5, 0.0 → -5
    score_series = np.clip(score_series, -5, 5)

    return pred_series, signal_series, score_series, model, scaler


def kelly_sizing(features, target, predictions):
    """
    Kelly criterion on the last 60 days (recent performance only).
    f* = (p * b - q) / b
    """
    recent_features = features.tail(60)
    recent_target   = target.loc[recent_features.index]
    recent_preds    = predictions[-60:] if len(predictions) >= 60 else predictions

    # Use prediction > 0.5 as the "model" signal
    pred_direction = (recent_preds > 0.5).astype(int)
    wins  = (pred_direction == recent_target.values).sum()
    total = len(recent_target)
    p     = wins / total if total > 0 else 0.5
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


def cumulative_returns(features, signals, predictions):
    """
    Compare strategy (using LSTM signal) vs buy-and-hold.
    Both start at 1.0.
    """
    ret_1d = features["return_1d"]
    sig    = signals.map({"LONG": 1, "SHORT": -1, "NEUTRAL": 0})

    # Use prediction probability for position sizing (more nuanced than binary)
    # For simplicity, use binary signal with 10% position size
    strat_ret = sig * ret_1d * 0.1
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

    # 4. LSTM predictions
    print("\nPreparing sequences and training LSTM...")
    pred_probs, signal_series, score_series, model, scaler = generate_predictions(features, target)

    # Create signals DataFrame
    signals = pd.DataFrame({
        "signal": signal_series,
        "score": score_series,
        "pred_prob": pred_probs,
    }, index=features.index)

    # 5. Kelly
    kelly = kelly_sizing(features, target, pred_probs)
    print(f"\nKelly sizing (last 60 days):")
    print(f"  Win rate:   {kelly['win_rate']*100:.1f}%")
    print(f"  Kelly full: {kelly['kelly_full']*100:.1f}%")
    print(f"  Kelly half: {kelly['kelly_half']*100:.1f}%  ← deploy this")

    # 6. Returns
    strat_cumret, bh_cumret, sharpe, max_dd = cumulative_returns(features, signals["signal"], pred_probs)
    print(f"\nPerformance:")
    print(f"  Sharpe ratio: {sharpe}")
    print(f"  Max drawdown: {max_dd}%")

    # 7. Current signal (latest bar)
    latest_idx = -1
    latest_signal = signals.iloc[latest_idx]
    latest_feat = features.iloc[latest_idx]
    print(f"\nCurrent signal ({features.index[latest_idx]}): {latest_signal['signal']}  (score={latest_signal['score']:.1f})")

    # 8. Build JSON payload
    tail = -252  # last 252 bars (1 year)
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
            "signal":        str(signals.loc[d, "signal"]),
            "score":         int(signals.loc[d, "score"]),
            "pred_prob":     round(float(signals.loc[d, "pred_prob"]), 4),
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

    # For the current signal "votes", we can show feature contributions as pseudo-votes
    # based on each feature's directional impact
    current_feat_values = latest_feat.to_dict()
    vote_map = {}
    # Simple rule: if feature > 0 → bullish, < 0 → bearish (for most features)
    for feat_name, val in current_feat_values.items():
        if val > 0.1:
            vote_map[f"vote_{feat_name}"] = 1
        elif val < -0.1:
            vote_map[f"vote_{feat_name}"] = -1
        else:
            vote_map[f"vote_{feat_name}"] = 0

    output = {
        "meta": {
            "ticker":       TICKER,
            "period":       PERIOD,
            "interval":     INTERVAL,
            "total_bars":   len(features),
            "from":         features.index[0],
            "to":           features.index[-1],
            "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model":        "LSTM (PyTorch)",
            "lookback":     LOOKBACK,
            "test_accuracy": round(float(test_accuracy) if 'test_accuracy' in locals() else 0.0, 4),
        },
        "current_signal": {
            "date":         features.index[latest_idx],
            "signal":       str(latest_signal["signal"]),
            "score":        int(latest_signal["score"]),
            "pred_prob":    round(float(latest_signal["pred_prob"]), 4),
            "votes":        vote_map,  # pseudo-votes from feature values
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
