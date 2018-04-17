# -*- coding: utf-8 -*-

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Activation


class Model():
    def __init__(self):
        pass

    def rnn(self, input_shape, output_dim, optimizer):
        # Initialising the RNN
        regressor = Sequential()

        # Adding the first LSTM layer
        regressor.add(LSTM(units=output_dim, return_sequences=False, input_shape=input_shape))

        # Adding Dense layer to aggregate the data from the prediction vector into a single value
        regressor.add(Dense(units=1))

        # Adding Linear Activation function
        activation = 'linear'
        regressor.add(Activation(activation))

        # Compiling the RNN
        regressor.compile(
            optimizer=optimizer,
            metrics=['accuracy'],
            loss='mean_squared_error')

        return regressor
