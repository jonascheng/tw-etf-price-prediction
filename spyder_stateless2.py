# -*- coding: utf-8 -*-
# 1. Stateless LSTM
# 2. Percentage change
# 3. Lookback 50 days and forecast 5 days
# 4. Last 240 days for testing set
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
import settings
from util import load_csv
filepath = '{}/tetfp.csv'.format(settings.DATASET_PATH)
history = load_csv(filepath)

# Extracting/Filtering the training dataset by stock_id
# Taking 收盤價 as a predictor
from util import query_close_price
dataset = query_close_price(history, int(stock_id))

###########################################################
# Visualising the stock price
from util import plot_stock_price
plot_stock_price(dataset, last_ndays=240)

###########################################################
# Calculating percentage change
df = pd.DataFrame(dataset)
df = df.pct_change().fillna(0)

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
scaled_price = sc.fit_transform(df.values)

###########################################################
# Visualising the stock price
from util import plot_stock_price
plot_stock_price(df.values, last_ndays=240)
plot_stock_price(scaled_price, last_ndays=240)

###########################################################
# series_to_supervised
from util import series_to_supervised
supervised = series_to_supervised(scaled_price, n_in=50, n_out=5)

###########################################################
# Visualising the stock price
from util import plot_stock_price
plot_stock_price(df.values, first_ndays=55)
plot_stock_price(supervised[0].transpose())

Xy = np.array(supervised)

###########################################################
# Visualising the stock price
from util import plot_stock_price
plot_stock_price(Xy[0].transpose())

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
# Creating stateless model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import TimeDistributed
from keras import metrics

input_shape = (X_train.shape[1], 1)
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
        #validation_data=(X_test, y_test),
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


plt.plot(predicted_price[-30:], color = 'blue', label = 'Predicted Price')
plt.plot(real_price[-30:], color = 'red', label = 'Real Price')
plt.title('Price Prediction')
plt.xlabel('Time')
plt.ylabel('ETF Stock Price')
plt.legend()
plt.show()
