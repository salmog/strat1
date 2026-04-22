import vectorbt as vbt
import pandas as pd
import numpy as np

# 1. Fetch Data
symbol = "AAPL"
data = vbt.YFData.download(symbol, period="5y")
ohlcv = data.get()
close = ohlcv['Close']
low = ohlcv['Low']
high = ohlcv['High']

# 2. Indicators (Testing RSI < 30 AND RSI < 40)
sma200 = vbt.MA.run(close, window=200).ma
sma50 = vbt.MA.run(close, window=50).ma
rsi = vbt.RSI.run(close, window=14).rsi
atr = vbt.ATR.run(high, low, close, window=14).atr

# 3. Parameter Sweep: Entry Logic for RSI 30 and 40
# We create a DataFrame of entries for both thresholds
entries_rsi30 = (close > sma200) & (low <= sma50) & (rsi < 30)
entries_rsi40 = (close > sma200) & (low <= sma50) & (rsi < 40)

# Combine them into a multi-index DataFrame for vectorbt
entries = pd.concat([entries_rsi30, entries_rsi40], axis=1, keys=['RSI < 30', 'RSI < 40'])

# Match dimensions for 'close' and 'atr' to align with our entries DataFrame
close_aligned = pd.concat([close, close], axis=1, keys=['RSI < 30', 'RSI < 40'])
atr_aligned = pd.concat([atr, atr], axis=1, keys=['RSI < 30', 'RSI < 40'])

# 4. True Risk-Based Position Sizing
starting_capital = 10000
risk_per_trade = 0.02 # 2% of capital
risk_amount = starting_capital * risk_per_trade # $200 risk per trade

# Stop loss distance in dollars
stop_distance_dollars = 2 * atr_aligned 

# Calculate exact number of shares to buy so that a hit stop = $200 loss
# Use np.floor to avoid buying fractional shares
size_in_shares = np.floor(risk_amount / stop_distance_dollars)

# Replace infinities/NaNs with 0 just in case ATR is 0
size_in_shares = size_in_shares.replace([np.inf, -np.inf], 0).fillna(0)

# 5. Run Backtest
pf = vbt.Portfolio.from_signals(
    close=close_aligned,
    entries=entries,
    exits=None, 
    sl_trail=(2 * atr_aligned) / close_aligned, # 2 ATR trailing stop
    init_cash=starting_capital,
    fees=0.001, 
    size=size_in_shares, # Use our calculated shares
    size_type='amount'   # Tell vbt we are passing exact quantities
)

# 6. Compare Results
print(pf.stats())

# You can plot a specific run, e.g., the 'RSI < 40' run:
# pf['RSI < 40'].plot().show()
