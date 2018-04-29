# -*- coding: utf-8 -*-

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset_etf_history = pd.read_csv('TBrain_Round2_DataSet_20180427/tetfp.csv',encoding='big5-hkscs')
dataset_etf_adjusted_history = pd.read_csv('TBrain_Round2_DataSet_20180427/taetfp.csv',encoding='big5-hkscs')
dataset_stock_history = pd.read_csv('TBrain_Round2_DataSet_20180427/tsharep.csv',encoding='big5-hkscs')
dataset_stock_adjusted_history = pd.read_csv('TBrain_Round2_DataSet_20180427/tasharep.csv',encoding='big5-hkscs')

# Listing unique stock codes
unique_etf_stock_codes = dataset_etf_history.代碼.unique()
unique_stock_codes = dataset_stock_history.代碼.unique()

# Extracting the training dataset for 元大台灣50
testing_set = True
dataset_etf_history = dataset_etf_history.loc[dataset_etf_history['代碼'] == 50]
dataset_etf_adjusted_history = dataset_etf_adjusted_history.loc[dataset_etf_adjusted_history['代碼'] == 50]

if testing_set is True:
    # reserve the last 5 working date
    training_dataset = dataset_etf_history[:-5]
    testing_dataset = dataset_etf_history[-5:]
else:
    # take all dataset as training one
    training_dataset = dataset_etf_history

# Taking 開盤價 as a predictor
training_open_price = training_dataset.iloc[:, 3:4].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))
training_scaled_open_price = sc.fit_transform(training_open_price)

# Creating a data structure with 60 timesteps look-back and 1 output
look_back = 60
X_train = []
y_train = []
for i in range(look_back, len(training_scaled_open_price)):
    X_train.append(training_scaled_open_price[i-look_back:i, 0])
    y_train.append(training_scaled_open_price[i, 0])
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

# Initialising the RNN
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

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

# Part 3 - Making the predictions and visualising the results

if testing_set is True:
    # Getting the real stock price of the last 5 working date
    real_open_price = testing_dataset.iloc[:, 3:4].values

    # Getting the predicted stock price of the last 5 working date
    dataset_total = dataset_etf_history[len(dataset_etf_history) - 5 - look_back:]

    # Taking 開盤價 as a input    
    inputs = dataset_total.iloc[:, 3:4].values
    inputs = inputs.reshape(-1,1)
    inputs = sc.transform(inputs)
    X_test = []
    for i in range(look_back, len(inputs)):
        X_test.append(inputs[i-60:i, 0])
    # Converting array of list to numpy array
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_open_price = regressor.predict(X_test)
    predicted_open_price = sc.inverse_transform(predicted_open_price)

    # Visualising the results
    plt.plot(real_open_price, color = 'red', label = 'Real Open Price')
    plt.plot(predicted_open_price, color = 'blue', label = 'Predicted Open Price')
    plt.title('Open Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('ETF 0050 Stock Price')
    plt.legend()
    plt.show()
