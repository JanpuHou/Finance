import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

symbol = '2883.tw'
# Read the CSV file
data = pd.read_csv("2883_data.csv")

import plotly.graph_objects as go

# Candlestick chart
candlestick = go.Figure(data=[go.Candlestick(x=data.index,
                                              open=data['Open'],
                                              high=data['High'],
                                              low=data['Low'],
                                              close=data['Close'])])

candlestick.update_layout(title=f'{symbol} Candlestick Chart',
                          xaxis_title='Date',
                          yaxis_title='Stock Price',
                          xaxis_rangeslider_visible=False)

candlestick.show()