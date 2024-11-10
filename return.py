import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

symbol = '2883.tw'
# Read the CSV file
data = pd.read_csv("2883_data.csv")

# Seaborn style set
sns.set(style="whitegrid")

# Distribution of Daily Returns
plt.figure(figsize=(12, 6))
sns.histplot(data['Close'].pct_change().dropna(), bins=30, kde=True, color='blue')
plt.title(f'Distribution of {symbol} Daily Returns')
plt.xlabel('Daily Return')
plt.ylabel('Frequency')
plt.show()