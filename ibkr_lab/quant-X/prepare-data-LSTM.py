"""
prepare-data-LSTM.py
---------------
Pulls real OHLCV data from Yahoo Finance, computes 6 stationary features,
runs ADF stationarity tests, trains an LSTM on sequential data,
generates predictions, and exports one JSON file for the React dashboard.

Install dependencies (one time):
    pip install yfinance pandas numpy statsmodels torch scikit-learn

Run:
    python prepare-data-LSTM.py

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
PERIOD   = "2y"          # how far back to pull: 1y, 2y, 5y
INTERVAL = "1d"          # bar size: 1d, 1wk — use 1d for daily signals
LOOKBACK = 20            # number of past days to feed into LSTM
HIDDEN_SIZE = 64         # LSTM hidden units
NUM_LAYERS = 2           # LSTM layers
DROPOUT = 0.2            # dropout rate
EPOCHS = 100             # max training epochs
LEARNING_RATE = 0.001    # optimizer learning rate
PATIENCE = 10            # early stopping patience
BATCH_SIZE = 16          # training batch size

# TARGET CONFIGURATION:
# "binary"      -> Predict Up (1) or Down (0)
# "regression"  -> Predict the actual next-day percentage return
# "multi_class" -> Predict LONG (2), NEUTRAL (1), or SHORT (0) based on thresholds
# "quantile"    -> Predict top 20% (2), middle 60% (1), or bottom 20% (0)
TARGET_MODE = "multi_class"
MULTI_CLASS_THRESHOLD = 0.005 # +/- 0.5% return threshold for NEUTRAL
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
    clean = series.dropna()
    result = adfuller(clean)
    stationary = bool(result[1] < 0.05)
    print(f"  {name:<25s} ADF={result[0]:7.3f}  p={result[1]:.4f}  {'[PASS] stationary' if stationary else '[FAIL] NON-STATIONARY'}")
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

    # ── TARGET CONFIGURATION ───────────────────────────────────────────────────
    future_return = returns.shift(-1)
    
    if TARGET_MODE == "binary":
        target = (future_return > 0).astype(int)
    elif TARGET_MODE == "regression":
        target = future_return
    elif TARGET_MODE == "multi_class":
        target = pd.Series(1, index=future_return.index)
        target[future_return > MULTI_CLASS_THRESHOLD] = 2
        target[future_return < -MULTI_CLASS_THRESHOLD] = 0
    elif TARGET_MODE == "quantile":
        target = pd.Series(1, index=future_return.index)
        valid_returns = future_return.dropna()
        if len(valid_returns) > 0:
            top_q = valid_returns.quantile(0.80)
            bot_q = valid_returns.quantile(0.20)
            target[future_return >= top_q] = 2
            target[future_return <= bot_q] = 0
    else:
        raise ValueError("Invalid TARGET_MODE")

    features = features.dropna()
    target   = target.loc[features.index]
    
    # Final alignment to remove any remaining NaNs (from shift)
    valid_idx = features.index.intersection(target.dropna().index)
    features = features.loc[valid_idx]
    target = target.loc[valid_idx]

    return features, target


def prepare_sequences(features, target, lookback=LOOKBACK):
    X, y = [], []
    feature_array = features.values
    target_array = target.values
    for i in range(len(feature_array) - lookback):
        X.append(feature_array[i:i+lookback])
        y.append(target_array[i+lookback])
    return np.array(X), np.array(y)


# ── LSTM MODEL DEFINITION ───────────────────────────────────────────────────
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
        
        if TARGET_MODE == "regression":
            self.fc = nn.Linear(hidden_size, 1)
        elif TARGET_MODE in ["multi_class", "quantile"]:
            self.fc = nn.Linear(hidden_size, 3)
        else: # binary
            self.fc = nn.Linear(hidden_size, 1)
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        
        if TARGET_MODE == "binary":
            return self.sigmoid(out).squeeze()
        elif TARGET_MODE in ["multi_class", "quantile"]:
            return out # Logits for CrossEntropyLoss
        else: # regression
            return out.squeeze()


# ── TRAINING LOOP WITH EARLY STOPPING ────────────────────────────────────────
def train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE, patience=PATIENCE):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    if TARGET_MODE == "regression":
        criterion = nn.MSELoss()
    elif TARGET_MODE in ["multi_class", "quantile"]:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCELoss()
        
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float('inf')
    best_weights = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            if TARGET_MODE in ["multi_class", "quantile"]:
                y_batch = y_batch.long()
                
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                if TARGET_MODE in ["multi_class", "quantile"]:
                    y_batch = y_batch.long()
                    
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
    X, y = prepare_sequences(features, target, lookback)

    n = len(X)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    scaler = StandardScaler()
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

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    input_size = X_train.shape[-1]
    model = TradingLSTM(input_size=input_size)
    print("\nTraining LSTM...")
    model = train_model(model, train_loader, val_loader)

    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_t)
        if TARGET_MODE == "binary":
            test_accuracy = ((test_preds > 0.5).float() == y_test_t).float().mean().item()
            print(f"  Test accuracy: {test_accuracy:.4f}")
        elif TARGET_MODE in ["multi_class", "quantile"]:
            _, predicted_classes = torch.max(test_preds, 1)
            test_accuracy = (predicted_classes == y_test_t.long()).float().mean().item()
            print(f"  Test accuracy: {test_accuracy:.4f}")
        else: # regression
            mse = nn.MSELoss()(test_preds, y_test_t).item()
            print(f"  Test MSE: {mse:.6f}")
            test_accuracy = mse

    X_full_t = torch.FloatTensor(scale_sequences(X))
    model.eval()
    with torch.no_grad():
        if TARGET_MODE == "binary":
            full_preds = model(X_full_t).numpy()
        elif TARGET_MODE in ["multi_class", "quantile"]:
            logits = model(X_full_t)
            probs = torch.softmax(logits, dim=1).numpy()
            predicted_classes = np.argmax(probs, axis=1)
        else: # regression
            full_preds = model(X_full_t).numpy()

    signal_series = np.full(len(features), "NEUTRAL", dtype=object)
    score_series = np.full(len(features), 0.0)

    if TARGET_MODE == "binary":
        pred_series = np.full(len(features), 0.5)
        pred_series[lookback:] = full_preds
        signal_series[pred_series > 0.6] = "LONG"
        signal_series[pred_series < 0.4] = "SHORT"
        score_series = np.clip((pred_series - 0.5) * 10, -5, 5)
    elif TARGET_MODE in ["multi_class", "quantile"]:
        prob_long = np.full(len(features), 0.5)
        prob_short = np.full(len(features), 0.5)
        prob_long[lookback:] = probs[:, 2]
        prob_short[lookback:] = probs[:, 0]
        
        full_classes = np.full(len(features), 1)
        full_classes[lookback:] = predicted_classes
        
        signal_series[full_classes == 2] = "LONG"
        signal_series[full_classes == 0] = "SHORT"
        score_series = np.clip((prob_long - prob_short) * 5, -5, 5)
        pred_series = prob_long # Export prob_long for JSON
    elif TARGET_MODE == "regression":
        pred_series = np.full(len(features), 0.0)
        pred_series[lookback:] = full_preds
        signal_series[pred_series > MULTI_CLASS_THRESHOLD] = "LONG"
        signal_series[pred_series < -MULTI_CLASS_THRESHOLD] = "SHORT"
        score_series = np.clip(pred_series * 100, -5, 5)

    return pred_series, signal_series, score_series, model, scaler, test_accuracy


def kelly_sizing(features, signals):
    recent_features = features.tail(60)
    recent_signals = signals.tail(60)
    
    # Only calculate over days we actually took a trade
    active_mask = recent_signals != "NEUTRAL"
    if not active_mask.any():
        return {
            "win_rate": 0.0, "win_loss_b": 0.0, 
            "kelly_full": 0.0, "kelly_half": 0.0,
            "max_position": "No active trades"
        }
        
    active_returns = recent_features["return_1d"][active_mask]
    active_sigs = recent_signals[active_mask].map({"LONG": 1, "SHORT": -1})
    
    trade_returns = active_returns * active_sigs
    wins = (trade_returns > 0).sum()
    total = len(trade_returns)
    
    p = wins / total if total > 0 else 0.5
    q = 1 - p
    
    avg_win = trade_returns[trade_returns > 0].mean() if (trade_returns > 0).any() else 0
    avg_loss = abs(trade_returns[trade_returns < 0].mean()) if (trade_returns < 0).any() else 0
    
    b = avg_win / (avg_loss + 1e-10)
    
    if b > 0:
        kelly_full = (p * b - q) / b
    else:
        kelly_full = 0
        
    kelly_half = max(0, min(0.25, kelly_full * 0.5))

    return {
        "win_rate":    round(float(p), 4),
        "win_loss_b":  round(float(b), 4),
        "kelly_full":  round(float(kelly_full), 4),
        "kelly_half":  round(float(kelly_half), 4),
        "max_position": "2% hard cap regardless of Kelly",
    }


def cumulative_returns(features, signals):
    ret_1d = features["return_1d"]
    sig    = signals.map({"LONG": 1, "SHORT": -1, "NEUTRAL": 0}).fillna(0)

    # For simplicity, use binary signal with 10% position size
    strat_ret = sig * ret_1d * 0.1
    bh_ret    = ret_1d

    strat_cumret = (1 + strat_ret).cumprod()
    bh_cumret    = (1 + bh_ret).cumprod()

    excess = strat_ret - 0
    sharpe = float((excess.mean() / (excess.std() + 1e-10)) * np.sqrt(252))

    rolling_max = strat_cumret.cummax()
    drawdown    = (strat_cumret - rolling_max) / (rolling_max + 1e-10)
    max_dd      = float(drawdown.min())

    return strat_cumret, bh_cumret, round(sharpe, 3), round(max_dd * 100, 2)


def main():
    df = fetch_data(TICKER, PERIOD, INTERVAL)

    print("\nBuilding features...")
    features, target = build_features(df)

    print("\nStationarity tests:")
    raw_adf = adf_test(df["Close"].reindex(features.index), "raw_price (DO NOT USE)")
    adf_results = [raw_adf]
    for col in features.columns:
        adf_results.append(adf_test(features[col], col))

    print(f"\nPreparing sequences and training LSTM ({TARGET_MODE} mode)...")
    pred_probs, signal_series, score_series, model, scaler, test_acc = generate_predictions(features, target)

    signals = pd.DataFrame({
        "signal": signal_series,
        "score": score_series,
        "pred_prob": pred_probs,
    }, index=features.index)

    kelly = kelly_sizing(features, signals["signal"])
    print(f"\nKelly sizing (last 60 days):")
    print(f"  Win rate:   {kelly['win_rate']*100:.1f}%")
    print(f"  Kelly full: {kelly['kelly_full']*100:.1f}%")
    print(f"  Kelly half: {kelly['kelly_half']*100:.1f}%  <-- deploy this")

    strat_cumret, bh_cumret, sharpe, max_dd = cumulative_returns(features, signals["signal"])
    print(f"\nPerformance:")
    print(f"  Sharpe ratio: {sharpe}")
    print(f"  Max drawdown: {max_dd}%")

    latest_idx = -1
    latest_signal = signals.iloc[latest_idx]
    latest_feat = features.iloc[latest_idx]
    print(f"\nCurrent signal ({features.index[latest_idx]}): {latest_signal['signal']}  (score={latest_signal['score']:.1f})")

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

    current_feat_values = latest_feat.to_dict()
    vote_map = {}
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
            "target_mode":  TARGET_MODE,
            "test_accuracy": round(float(test_acc), 4),
        },
        "current_signal": {
            "date":         features.index[latest_idx],
            "signal":       str(latest_signal["signal"]),
            "score":        int(latest_signal["score"]),
            "pred_prob":    round(float(latest_signal["pred_prob"]), 4),
            "votes":        vote_map,
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

    print(f"\nExported -> {out_path}  ({len(json.dumps(output)) // 1024} KB)")
    print("Next step: copy trading_data.json into your React project's /public folder.")


if __name__ == "__main__":
    main()
