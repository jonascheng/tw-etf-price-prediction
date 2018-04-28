# -*- coding: utf-8 -*-
# 1. Stateful LSTM
# 2. Lookback 5 days and forecast 5 days
# 3. Last 240 days for testing set
# 4. Normalized
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
from util import query_close_price
dataset = query_close_price(history, int(stock_id))

###########################################################
# Visualising the stock price
from util import plot_stock_price
plot_stock_price(dataset, last_ndays=240)

# Feature Scaling
#from sklearn.preprocessing import MinMaxScaler
#sc = MinMaxScaler(feature_range = (0, 1))
#scaled_price = sc.fit_transform(dataset)

###########################################################
# series_to_supervised
from util import series_to_supervised
supervised = series_to_supervised(dataset, n_in=5, n_out=5)

###########################################################
# Visualising the stock price
from util import plot_stock_price
plot_stock_price(dataset, first_ndays=10)
plot_stock_price(supervised[0].transpose())

###########################################################
# normalize_windows
import copy
from util import normalize_windows
ori_Xy = copy.deepcopy(supervised)
Xy = normalize_windows(supervised)

###########################################################
# Visualising the stock price
from util import plot_stock_price
plot_stock_price(Xy[0].transpose())

###########################################################
# train_test_split
from util import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Xy, test_samples=240, num_forecasts=5)

# reshape for stateful LSTM
y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))
y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1], 1))

###########################################################
# Visualising the stock price
from util import plot_stock_price
plot_stock_price(supervised[0].reshape(10, 1), first_ndays=5)
plot_stock_price(X_train[0])

plot_stock_price(supervised[0].reshape(10, 1), last_ndays=5)
plot_stock_price(y_train[0])

plot_stock_price(supervised[-1].reshape(10, 1), first_ndays=5)
plot_stock_price(X_test[-1])

plot_stock_price(supervised[-1].reshape(10, 1), last_ndays=5)
plot_stock_price(y_test[-1])

###########################################################
# Creating stateful model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import TimeDistributed
from keras import metrics

batch_input_shape = (1, X_train.shape[1], 1)

regressor = Sequential()
regressor.add(
    LSTM(units=50,              # Positive integer, dimensionality of the output space.
    return_sequences=True,  # Boolean. Whether to return the last output in the output sequence, or the full sequence.
    batch_input_shape=batch_input_shape,
    stateful=True))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
#regressor.add(
#    LSTM(units=50, 
#    return_sequences = True,
#    stateful=True))
#regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
#regressor.add(
#    LSTM(units=50, 
#    return_sequences = True,
#    stateful=True))
#regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
#regressor.add(
#    LSTM(units=50, 
#    return_sequences = True,
#    stateful=True))
#regressor.add(Dropout(0.2))

regressor.add(
    TimeDistributed(Dense(1)))
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


plt.plot(predicted_price, color = 'blue', label = 'Predicted Price')
plt.plot(real_price, color = 'red', label = 'Real Price')
plt.title('Price Prediction')
plt.xlabel('Time')
plt.ylabel('ETF Stock Price')
plt.legend()
plt.show()
