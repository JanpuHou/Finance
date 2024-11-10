import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Fetch historical stock data
symbol = '2883.tw'
data = yf.download(symbol, start='2023-11-08', end='2024-11-08', progress=False)

# Display the first few rows of the dataset
print(data.head())
data.to_csv(f'2883_data.csv')
# Read the CSV file
data = pd.read_csv("2883_data.csv")

plt.plot(data['Close'])
plt.show()