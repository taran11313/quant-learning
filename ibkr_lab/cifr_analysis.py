import nest_asyncio
nest_asyncio.apply()

from ib_insync import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    # 1. Connect to IBKR (paper trading port 7497, live 7496)
    ib = IB()
    ib.connect('127.0.0.1', 7497, clientId=2)

    # 2. Define contracts
    cifr = Stock('CIFR', 'SMART', 'USD')
    btc_fut = Crypto('BTC', 'PAXOS', 'USD')  # IB's spot BTC
    nvda = Stock('NVDA', 'SMART', 'USD')

    # 3. Fetch 2 years of daily bars
    contracts = [cifr, nvda]
    bars_dict = {}
    for contract in contracts:
        bars = ib.reqHistoricalData(
            contract, endDateTime='', durationStr='2 Y',
            barSizeSetting='1 day', whatToShow='TRADES', useRTH=True, formatDate=2
        )
        bars_dict[contract.symbol] = util.df(bars).set_index('date')['close']

    # Fetch BTC data
    btc_bars = ib.reqHistoricalData(
        btc_fut, endDateTime='', durationStr='2 Y',
        barSizeSetting='1 day', whatToShow='MIDPOINT', useRTH=False, formatDate=2
    )
    btc_prices = util.df(btc_bars).set_index('date')['close']

    # 4. Align and calculate returns
    df = pd.DataFrame({
        'CIFR': bars_dict['CIFR'],
        'NVDA': bars_dict['NVDA'],
        'BTC': btc_prices
    }).dropna()
    returns = df.pct_change().dropna()

    # 5. Rolling 60-day correlations
    rolling_corr_cifr_btc = returns['CIFR'].rolling(60).corr(returns['BTC'])
    rolling_corr_cifr_nvda = returns['CIFR'].rolling(60).corr(returns['NVDA'])

    # 6. Plot
    plt.figure(figsize=(14,6))
    plt.plot(rolling_corr_cifr_btc.index, rolling_corr_cifr_btc, label='CIFR-BTC 60d Corr')
    plt.plot(rolling_corr_cifr_nvda.index, rolling_corr_cifr_nvda, label='CIFR-NVDA 60d Corr')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.title('CIFR Rolling Correlation: Bitcoin vs. AI Factor (NVDA)')
    plt.ylabel('Correlation Coefficient')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 7. Disconnect
    ib.disconnect()

# This single line fixes the event loop problem
if __name__ == '__main__':
    IB.run(main())

