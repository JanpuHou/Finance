import mplfinance as mpf
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Fetch historical stock data
symbol = '2883.tw'
data = yf.download(symbol, start='2024-09-08', end='2024-11-09', progress=False)

# Display the first few rows of the dataset
print(data.head())

# Candlestick chart
mpf.plot(data,type='candle',title='2883.tw')