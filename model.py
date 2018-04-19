# -*- coding: utf-8 -*-

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Activation
from keras import metrics


class Model():
    def __init__(self):
        pass

    def rnn(self, input_shape, layers, output_dim, optimizer):
        # Initialising the RNN
        regressor = Sequential()

        # Adding the first LSTM layer
        print('Adding the first LSTM layers')
        return_sequences = False if layers == 1 else True
        regressor.add(
            LSTM(units=output_dim,  # Positive integer, dimensionality of the output space.
            return_sequences=return_sequences, # # Boolean. Whether to return the last output in the output sequence, or the full sequence.
            input_shape=input_shape))

        # Adding additional LSTM layers
        for i in range(layers-1, 0, -1):
            print('Adding additional LSTM layers {}'.format(i))
            return_sequences = False if i == 1 else True            
            regressor.add(
                LSTM(units=output_dim, 
                return_sequences=return_sequences))

        # Adding Dense layer to aggregate the data from the prediction vector into a single value
        regressor.add(Dense(units=1))

        # Adding Linear Activation function
        activation = 'linear'
        regressor.add(Activation(activation))

        # Compiling the RNN
        regressor.compile(
            optimizer=optimizer,
            metrics=[metrics.mse],
            loss='mean_squared_error')

        return regressor
