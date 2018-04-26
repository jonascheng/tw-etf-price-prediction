# -*- coding: utf-8 -*-
# 1. Stateless LSTM
# 2. Lookback 50 days and forecast 5 days
# 3. Last 240 days for testing set
# 4. Normalized
# 5. Open price
"""
Spyder Editor

This is a temporary script file.
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

stock_id = '0050'

###########################################################
from util import load_csv
filepath = 'TBrain_Round2_DataSet_20180331/tetfp.csv'
history = load_csv(filepath)

# Extracting/Filtering the training dataset by stock_id
# Taking 收盤價 as a predictor
from util import query_close_price, query_open_price, query_avg_price, query_volume
dataset = query_close_price(history, int(stock_id))
dataset_open = query_open_price(history, int(stock_id))
dataset_avg = query_avg_price(history, int(stock_id))
dataset_vol = query_volume(history, int(stock_id))

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))
scaled_dataset_vol = sc.fit_transform(dataset_vol)

###########################################################
# Visualising the stock price
last_ndays = 240
from util import plot_stock_price
plot_stock_price(dataset, last_ndays=last_ndays)
plot_stock_price(dataset_open, last_ndays=last_ndays)
plot_stock_price(dataset_avg, last_ndays=last_ndays)
plot_stock_price(dataset_vol, last_ndays=last_ndays)
plot_stock_price(scaled_dataset_vol, last_ndays=last_ndays)

###########################################################
# series_to_supervised
from util import series_to_supervised
supervised = series_to_supervised(dataset, n_in=50, n_out=5)
supervised_open = series_to_supervised(dataset_open, n_in=50, n_out=5)
supervised_avg = series_to_supervised(dataset_avg, n_in=50, n_out=5)

###########################################################
# Visualising the stock price
from util import plot_stock_price
plot_stock_price(dataset, first_ndays=55)
plot_stock_price(supervised[0].transpose())
plot_stock_price(dataset_open, first_ndays=55)
plot_stock_price(supervised_open[0].transpose())
plot_stock_price(dataset_avg, first_ndays=55)
plot_stock_price(supervised_avg[0].transpose())

###########################################################
# normalize_windows
import copy
from util import normalize_windows
ori_Xy = copy.deepcopy(supervised)
Xy = normalize_windows(supervised)
F1 = normalize_windows(supervised_open)
F2 = normalize_windows(supervised_avg)

###########################################################
# Visualising the stock price
from util import plot_stock_price
plot_stock_price(Xy[0].transpose())

###########################################################
# train_test_split
from util import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Xy, test_samples=240, num_forecasts=5)

###########################################################
# Adding more features
F1_train, F1_test, _, _ = train_test_split(F1, test_samples=240, num_forecasts=5)
F2_train, F2_test, _, _ = train_test_split(F2, test_samples=240, num_forecasts=5)

X_train = np.append(X_train, F1_train, axis=2)
X_train = np.append(X_train, F2_train, axis=2)
X_test = np.append(X_test, F1_test, axis=2)
X_test = np.append(X_test, F2_test, axis=2)

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
# Creating stateless model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import TimeDistributed
from keras import metrics

input_shape = (X_train.shape[1], X_train.shape[2])
output = y_train.shape[1]

regressor = Sequential()
regressor.add(
    LSTM(units=50,              # Positive integer, dimensionality of the output space.
    return_sequences=False,  # Boolean. Whether to return the last output in the output sequence, or the full sequence.
    input_shape=input_shape,
    stateful=False))
regressor.add(Dropout(0.2))

regressor.add(Dense(units=output))
# Adding Linear Activation function
activation = 'linear'
regressor.add(Activation(activation))
# Compiling the RNN
regressor.compile(
    optimizer='adam',
    metrics=[metrics.mse],
    loss='mean_squared_error')

print(regressor.summary())

###########################################################
# Fitting the RNN to the Training set
for i in range(10):
    regressor.fit(
        X_train, 
        y_train, 
        epochs=1, 
        batch_size=1,
        validation_data=(X_test, y_test),
        shuffle=False)
    regressor.reset_states()

    real_price = y_train
    predicted_price = regressor.predict(X_train, batch_size=1)
    #real_price = y_test
    #predicted_price = regressor.predict(X_test, batch_size=1)
    
    real_price = np.concatenate((real_price[0], np.array(real_price)[1:, -1]))
    predicted_price = np.concatenate((predicted_price[0], np.array(predicted_price)[1:, -1]))
    np.save('real_price.npy', real_price)
    np.save('predicted_price_0.npy', predicted_price)
    
    plt.plot(predicted_price, color = 'blue', label = 'Predicted Price')
    plt.plot(real_price, color = 'red', label = 'Real Price')
    plt.title('Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('ETF Stock Price')
    plt.legend()
    plt.show()

###########################################################
# Visualising the results
real_price = y_test
predicted_price = regressor.predict(X_test, batch_size=1)

real_price = np.concatenate((real_price[0], np.array(real_price)[1:, -1]))
predicted_price = np.concatenate((predicted_price[0], np.array(predicted_price)[1:, -1]))


plt.plot(predicted_price[-5:], color = 'blue', label = 'Predicted Price')
plt.plot(real_price[-5:], color = 'red', label = 'Real Price')
plt.title('Price Prediction')
plt.xlabel('Time')
plt.ylabel('ETF Stock Price')
plt.legend()
plt.show()

###########################################################
# Creating model
#from model import create_stateless_lstm_model
#layers= 1
#output_dim = 50
#optimizer = 'adam'
#regressor = create_stateless_lstm_model(
#    X_train,
#    y_train,
#    layers=layers,
#    output_dim=output_dim, 
#    optimizer=optimizer)
