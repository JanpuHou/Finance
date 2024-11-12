import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import yfinance as yf

# Fetch historical stock data
symbol = '2883.tw'
data = yf.download(symbol, start='2024-10-16', end='2024-11-12', progress=False)

# Display the first few rows of the dataset
print(data)
plt.plot(data['Close'])
plt.show()

# We slice the data frame to get the column we want and normalize the data.

from sklearn.preprocessing import MinMaxScaler
price = data[['Close']]
scaler = MinMaxScaler(feature_range=(-1, 1))
price['Close'] = scaler.fit_transform(price['Close'].values.reshape(-1,1))

    
lookback = 20 # choose sequence length

x=price['Close']
print(x)
#x=[ 0.60317462,0.57142846,0.69841247,0.68253969,0.873016,0.80952369,0.95238107,1.,0.80952369,0.80952369,0.77777754,0.746032,0.76190477,0.73015862,0.65079354, 0.68253969,0.68253969,0.66666692,0.746032  ]
x_test = np.array(x)
#print(x_test.shape)

y_test = np.expand_dims(x_test, axis=1)
#print(y_test)
#print(y_test.shape)

y_final = np.expand_dims(y_test, axis=0)
#print(y_final)
print(y_final.shape)
print(y_final)
# Then we transform them into tensors, 
# which is the basic structure for building a PyTorch model.

import torch
import torch.nn as nn

x_final = torch.from_numpy(y_final).type(torch.Tensor)

# We define some common values for both models regarding the layers.

model = torch.jit.load('model_scripted.pt')
model.eval()

import math, time
from sklearn.metrics import mean_squared_error

# make predictions
y_test_pred = model(x_final)

# invert predictions
y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())


print (y_test_pred)