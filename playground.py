# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

###########################################################
from util import load_csv
filepath = 'TBrain_Round2_DataSet_20180331/tetfp.csv'
history = load_csv(filepath)

# Extracting/Filtering the training dataset by stock_id
# Taking 收盤價 as a predictor
from util import query_close_price
dataset = query_close_price(history, 50)

###########################################################
# Visualising the stock price
from util import plot_stock_price
plot_stock_price(dataset, last_ndays=240)

# Calculating percentage changes from starting point
df2 = pd.DataFrame(dataset)
df2 = df2.div(df2[0][0], axis=0) - 1
plot_stock_price(df2.values, last_ndays=240)

# Calculating percentage changes from starting point
# + Feature Scaling
from sklearn.preprocessing import MinMaxScaler
df3 = pd.DataFrame(dataset)
df3 = df3.div(df3[0][0], axis=0) - 1
sc = MinMaxScaler(feature_range = (0, 1))
scaled_price = sc.fit_transform(df3.values)
plot_stock_price(scaled_price, last_ndays=240)

# Calculating percentage change
df = pd.DataFrame(dataset)
df = df.pct_change().fillna(0)
plot_stock_price(df.values, last_ndays=240)

###########################################################
# series_to_supervised
from util import series_to_supervised
supervised = series_to_supervised(dataset, n_in=50, n_out=5)

###########################################################
# Visualising the stock price
from util import plot_stock_price
plot_stock_price(dataset, first_ndays=55)
plot_stock_price(supervised[0].reshape(55, 1))

###########################################################
# normalize_windows
from util import normalize_windows
Xy = normalize_windows(supervised)

###########################################################
# Visualising the stock price
from util import plot_stock_price
plot_stock_price(df2.values, first_ndays=55)
plot_stock_price(Xy[0].reshape(55, 1))

# + Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
sc.fit(Xy.reshape(Xy.shape[0]*Xy.shape[1], 1))
scaled_Xy = sc.transform(Xy)

###########################################################
# train_test_split
from util import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Xy, test_samples=240, num_forecasts=5)

###########################################################
# Visualising the stock price
from util import plot_stock_price
plot_stock_price(supervised[0].reshape(55, 1), first_ndays=50)
plot_stock_price(X_train[0])

plot_stock_price(supervised[0].reshape(55, 1), last_ndays=5)
plot_stock_price(y_train[0])

plot_stock_price(supervised[-1].reshape(55, 1), first_ndays=50)
plot_stock_price(X_test[-1])

plot_stock_price(supervised[-1].reshape(55, 1), last_ndays=5)
plot_stock_price(y_test[-1])

###########################################################
# Visualising the stock price
from util import plot_stock_price
plot_stock_price(df2.values, first_ndays=55)
plot_stock_price(Xy[0].reshape(55, 1))
plot_stock_price(scaled_price[0].reshape(55, 1))

###########################################################
# Creating model
from model import create_stateless_lstm_model
layers= 1
output_dim = 50
optimizer = 'adam'
regressor = create_stateless_lstm_model(
    input_shape=(X_train.shape[1], X_train.shape[2]),
    layers=layers,
    output_dim=output_dim, 
    optimizer=optimizer)
###########################################################
# Visualising the results
plt.plot(training_scaled_close_price, color = 'blue', label = 'Predicted Open Price')
#plt.plot(dataset_pct.values, color = 'red', label = 'Real Open Price')
plt.title('Open Price Prediction')
plt.xlabel('Time')
plt.ylabel('ETF 0050 Stock Price')
plt.legend()
plt.show()
