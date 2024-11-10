import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

symbol = '2883.tw'
# Read the CSV file
data = pd.read_csv("2883_data.csv")

# Moving Average Plot
plt.figure(figsize=(12, 6))
data['Close'].plot(label=f'{symbol} Closing Price', linewidth=2)
data['Close'].rolling(window=30).mean().plot(label=f'{symbol} 30-Day Avg', linestyle='--', color='blue')
data['Close'].rolling(window=90).mean().plot(label=f'{symbol} 90-Day Avg', linestyle='--', color='red')
plt.title(f'{symbol} Closing Prices with 30-Day/60-Day Moving Average')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.show()