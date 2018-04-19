a# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

###########################################################
filepath = 'TBrain_Round2_DataSet_20180331/tetfp.csv'
history = pd.read_csv(filepath, encoding='big5-hkscs')

# Extracting/Filtering the training dataset by stock_id
stock_id = 50
dataset = history.loc[history['代碼'] == stock_id]

# Taking 收盤價 as a predictor
dataset = dataset.iloc[:, 6:7].values
dataset_pct = dataset.pct_change().fillna(0)*100
###########################################################
# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
dataset = sc.fit_transform(dataset_pct.values)
###########################################################
# _series_to_supervised
series = dataset
n_in = 60
n_out = 1
supervised = []
#for i in range(n_in, len(data)-n_out):
for i in range(n_in, len(series)-n_out+1):
    supervised.append(series[i - n_in:i+n_out, 0])
###########################################################
# __normalize_windows
from pandas import DataFrame

window_data = supervised
df = DataFrame(window_data)
#df = df.pct_change(axis=1).fillna(0) * 100
df2 = df.div(df[0], axis=0)-1
Xy = df.values
###########################################################
# Visualising the results
plt.plot(training_scaled_close_price, color = 'blue', label = 'Predicted Open Price')
#plt.plot(dataset_pct.values, color = 'red', label = 'Real Open Price')
plt.title('Open Price Prediction')
plt.xlabel('Time')
plt.ylabel('ETF 0050 Stock Price')
plt.legend()
plt.show()
