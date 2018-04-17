# -*- coding: utf-8 -*-

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset_etf_history = pd.read_csv('TBrain_Round2_DataSet_20180331/tetfp.csv',encoding='big5-hkscs')
dataset_stock_history = pd.read_csv('TBrain_Round2_DataSet_20180331/tsharep.csv',encoding='big5-hkscs')

# Listing unique stock codes
unique_etf_stock_codes = dataset_etf_history.代碼.unique()
unique_stock_codes = dataset_stock_history.代碼.unique()

# Extracting the training dataset for 元大台灣50
testing_set = True
dataset_etf_history = dataset_etf_history.loc[dataset_etf_history['代碼'] == 50]
# dataset_etf_adjusted_history = dataset_etf_adjusted_history.loc[dataset_etf_adjusted_history['代碼'] == 50]

if testing_set is True:
    # reserve the last 5 working date
    training_dataset = dataset_etf_history[:-5]
    testing_dataset = dataset_etf_history[-5:]
else:
    # take all dataset as training one
    training_dataset = dataset_etf_history

# Taking 收盤價 as a predictor
training_close_price = training_dataset.iloc[:, 6:7].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))
training_scaled_close_price = sc.fit_transform(training_close_price)

# Creating a data structure with 60 timesteps look-back and 1 output
look_back = 60
X_train = []
y_train = []
for i in range(look_back, len(training_scaled_close_price)):
    X_train.append(training_scaled_close_price[i-look_back:i, 0])
    y_train.append(training_scaled_close_price[i, 0])
# Converting array of list to numpy array
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(
        X_train, 
        (X_train.shape[0],  # batch_size which is number of observations
         X_train.shape[1],  # timesteps which is look_back,
         1                  # input_dim which is number of predictors
         ))

# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Activation

# Initialising the RNN
optimizer = 'rmsprop'     # Recommended optimizer for RNN
activation = 'linear'     # Linear activation
output_space = 50
dropout = 0.2
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(
        units = output_space,                 # Positive integer, dimensionality of the output space.
        return_sequences = True,    # Boolean. Whether to return the last output in the output sequence, or the full sequence.
        input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(dropout))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = output_space, return_sequences = True))
regressor.add(Dropout(dropout))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = output_space, return_sequences = True))
regressor.add(Dropout(dropout))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = output_space))
regressor.add(Dropout(dropout))

# Add Dense layer to aggregate the data from the prediction vector into a single value
regressor.add(Dense(units = 1))

# Add Linear Activation function
regressor.add(Activation(activation))

# Compiling the RNN
regressor.compile(optimizer = optimizer, loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

# Part 3 - Making the predictions and visualising the results

if testing_set is True:
    # Getting the real stock price of the last 5 working date
    real_price = testing_dataset.iloc[:, 6:7].values

    # Getting the predicted stock price of the last 5 working date
    dataset_total = dataset_etf_history[len(dataset_etf_history) - 5 - look_back:]

    # Taking 收盤價 as a input    
    inputs = dataset_total.iloc[:, 6:7].values
    inputs = inputs.reshape(-1,1)
    inputs = sc.transform(inputs)
    X_test = []
    for i in range(look_back, len(inputs)):
        X_test.append(inputs[i-60:i, 0])
    # Converting array of list to numpy array
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_price = regressor.predict(X_test)
    predicted_price = sc.inverse_transform(predicted_price)

    # Visualising the results
    plt.plot(real_price, color = 'red', label = 'Real Price')
    plt.plot(predicted_price, color = 'blue', label = 'Predicted Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('ETF 0050 Stock Price')
    plt.legend()
    plt.show()
