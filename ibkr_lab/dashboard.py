import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from ib_insync import *
import time
from datetime import datetime
from fredapi import Fred


# ===== DATA FETCH FUNCTIONS =====
def get_ib_data(ticker, duration='180 D'):
    try:
        ib = IB()
        ib.connect('127.0.0.1', 7497, clientId=1)
        contract = Stock(ticker, 'SMART', 'USD')
        bars = ib.reqHistoricalData(contract, endDateTime='', durationStr=duration,
                                     barSizeSetting='1 day', whatToShow='TRADES', useRTH=True)
        ib.disconnect()
        df = util.df(bars).set_index('date')
        return df
    except Exception as e:
        return None

# ===== MAIN DASHBOARD =====
st.set_page_config(layout="wide", page_title="DeepSeek Quant Dashboard")

st.title("Live Market Architecture")

# Sidebar controls
with st.sidebar:
    st.header("Data Controls")
    if st.button("Force Data Refresh"):
        st.cache_data.clear()

# 1. MACRO PANEL (Top row)
col1, col2, col3 = st.columns(3)
with col1:
    # Fetch DXY from yfinance with fallback handling for empty data
    import yfinance as yf
    dxy = yf.Ticker("DX-Y.NYB")
    dxy_hist = dxy.history(period="5d", interval="1d")
    if dxy_hist.empty or 'Close' not in dxy_hist.columns:
        st.metric("DXY", "N/A")
    else:
        dxy_price = dxy_hist['Close'].dropna().iloc[-1]
        st.metric("DXY", f"{dxy_price:.2f}")
with col2:
    # You can add Fed Reverse Repo here via FRED
    fred = Fred(api_key='ac61e5f95d3391de3307e345cf4fff3f')
    rrp = fred.get_series('RRPONTSYD')
    st.metric("Fed RRP", rrp.dropna().iloc[-1])

# 2. CROSS-ASSET CORRELATION HEATMAP (Middle row)
st.subheader("60-Day Rolling Correlation Web")

# Fetch your three tickers
tickers = ['CIFR', 'NVDA', 'BTC']
prices = {}
for t in tickers:
    df = get_ib_data(t)
    if df is not None:
        prices[t] = df['close']

if len(prices) == 3:
    # Merge and align dates
    combined_df = pd.DataFrame(prices).dropna()

    # Calculate log returns and rolling corrs
    returns = np.log(combined_df / combined_df.shift(1))
    rolling_corr = returns.rolling(window=60).corr()
    if isinstance(rolling_corr.index, pd.MultiIndex):
        last_date = rolling_corr.index.get_level_values(0).max()
        last_corr = rolling_corr.loc[last_date]
    else:
        last_corr = rolling_corr.iloc[-1]

    # Create Heatmap
    fig = go.Figure(data=go.Heatmap(
        z=last_corr.values,
        text=last_corr.round(2).values,
        texttemplate="%{text}",
        textfont={"size": 12},
        x=last_corr.columns,
        y=last_corr.index,
        colorscale='RdBu_r',
        zmin=-1, zmax=1
    ))
    fig.update_layout(title="Correlation Matrix (Last Day)")
    st.plotly_chart(fig, use_container_width=True)

    # Show regime alert
    if last_corr.loc['CIFR', 'NVDA'] < 0.3:
        st.error("⚠️ REGIME ALERT: CIFR-NVDA correlation < 0.3. AI Narrative is dead. Trade on BTC.")
    if last_corr.loc['CIFR', 'BTC'] < 0.1:
        st.warning("⚠️ REGIME ALERT: CIFR-BTC correlation < 0.1. CIFR is decoupled from Macro.")

# 3. ORDER FLOW TAPE (Bottom row)
st.subheader("Live Order Flow (Level 2 Proxy)")
# You can add real-time bar logic here
