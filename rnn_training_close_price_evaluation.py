# -*- coding: utf-8 -*-

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Tuned params and variables
batch_size = 32         # Batch size
nb_epoch = 100          # Epoch
look_back = 60          # number of sequence data
optimizer = 'rmsprop'   # Recommended optimizer for RNN
activation = 'linear'   # Linear activation
output_dim = 50         # Output dimension
dropout = 0.2           # Dropout rate

# Importing the dataset
dataset_etf_history = pd.read_csv('TBrain_Round2_DataSet_20180331/tetfp.csv',encoding='big5-hkscs')
dataset_stock_history = pd.read_csv('TBrain_Round2_DataSet_20180331/tsharep.csv',encoding='big5-hkscs')

# Listing unique stock codes
unique_etf_stock_codes = dataset_etf_history.代碼.unique()
unique_stock_codes = dataset_stock_history.代碼.unique()

# Extracting the training dataset for 元大台灣50
dataset_etf_history = dataset_etf_history.loc[dataset_etf_history['代碼'] == 50]

# Taking 收盤價 as a predictor
dataset_close_price = dataset_etf_history.iloc[:, 6:7].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))
dataset_close_price = sc.fit_transform(dataset_close_price)

# =============================================================================
# Spliting dataset into training and testing sets
from sklearn.model_selection import train_test_split

X = []
y = []
for i in range(look_back, len(dataset_close_price)):
    X.append(dataset_close_price[i-look_back:i, 0])
    y.append(dataset_close_price[i, 0:1])
# Converting array of list to numpy array
X, y = np.array(X), np.array(y)

# Select 10% of the data for testing and 90% for training.
# Shuffle the data in order to train in random order.
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=0) 

# Select the last 5 working date for testing and the others for training.
X_train = X[:-5,:]
y_train = y[:-5,:]
X_test = X[-5:,:]
y_test = y[-5:,:]
    
# Reshape the inputs from 1 dimenstion to 3 dimension
X_train = np.reshape(
        X_train, 
        (X_train.shape[0],  # batch_size which is number of observations
         X_train.shape[1],  # timesteps which is look_back,
         1                  # input_dim which is number of predictors
         ))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
# =============================================================================


# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Activation

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(
        # Positive integer, dimensionality of the output space.
        units = output_dim,     
        # Boolean. Whether to return the last output in the output sequence, or the full sequence.
        return_sequences = True,
        input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(dropout))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = output_dim, return_sequences = True))
regressor.add(Dropout(dropout))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = output_dim, return_sequences = True))
regressor.add(Dropout(dropout))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = output_dim))
regressor.add(Dropout(dropout))

# Add Dense layer to aggregate the data from the prediction vector into a single value
regressor.add(Dense(units = 1))

# Add Linear Activation function
regressor.add(Activation(activation))

# Compiling the RNN
regressor.compile(optimizer=optimizer, metrics=['accuracy'], loss='mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs=nb_epoch, batch_size=batch_size, validation_split=0.05)

# Part 3 - Evaluating the model
import math

mse_score = regressor.evaluate(X_test, y_test)
#rmse_score = math.sqrt(mse_score)
print('Mean squared error (MSE): {}'.format(mse_score))
#print('Root Mean squared error (RMSE): {}'.format(mse_score))

# Part 4 - Making the predictions and visualising the results

# Getting the real stock price of the last 5 working date
real_price = sc.inverse_transform(y_test)

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
