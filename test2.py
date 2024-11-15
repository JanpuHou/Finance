import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv("2883_data.csv")
print(data.head())
plt.plot(data['Close'])
plt.show()

# We slice the data frame to get the column we want and normalize the data.

from sklearn.preprocessing import MinMaxScaler
price = data[['Close']]
scaler = MinMaxScaler(feature_range=(-1, 1))
price['Close'] = scaler.fit_transform(price['Close'].values.reshape(-1,1))

# Now we split the data into train and test sets. 
# Before doing so, we must define the window width of the analysis. 
# The use of prior time steps to predict the next time step is called the sliding window method.

def split_data(stock, lookback):
    data_raw = stock.to_numpy() # convert to numpy array
    data = []
    
    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - lookback): 
        data.append(data_raw[index: index + lookback])
    
    data = np.array(data);
    test_set_size = int(np.round(0.2*data.shape[0]));
    train_set_size = data.shape[0] - (test_set_size);
    
    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]
    
    x_test = data[train_set_size:,:-1]
    y_test = data[train_set_size:,-1,:]
    
    return [x_train, y_train, x_test, y_test]
lookback = 20 # choose sequence length
x_train, y_train, x_test, y_test = split_data(price, lookback)

print('shape of x_train, y_train, x_test, y_test')
print (x_train.shape)
print (y_train.shape)
print (x_test.shape)
print (y_test.shape)