import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

symbol = '2883.tw'
# Read the CSV file
data = pd.read_csv("2883_data.csv")

# Correlation Heatmap
correlation_matrix = data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title(f'Correlation Heatmap for {symbol} Financial Metrics')
plt.show()