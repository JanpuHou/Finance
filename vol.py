import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

symbol = '2883.tw'
# Read the CSV file
data = pd.read_csv("2883_data.csv")

# Volume Plot
plt.figure(figsize=(12, 6))
plt.bar(data.index, data['Volume'], color='green', alpha=0.7)
plt.title(f'{symbol} Trading Volume Over Time')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.show()