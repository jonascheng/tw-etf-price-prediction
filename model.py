# -*- coding: utf-8 -*-

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import TimeDistributed
from keras import metrics


def create_stateless_lstm_model(X, y, layers, output_dim, optimizer, dropout=0):
    print('Creating model with layers {}, output_dim {}, optimizer {}, dropout {}'.format(
        layers, 
        output_dim, 
        optimizer,
        dropout))

    stateful = False
    input_shape = (X.shape[1], 1)
    output = y.shape[1]
    print('input_shape {}, output {}'.format(input_shape, output))

    # Initialising the RNN
    regressor = Sequential()

    # Adding the first LSTM layer
    print('Adding the first LSTM layers')
    return_sequences = False if layers == 1 else True
    regressor.add(
        LSTM(units=output_dim,              # Positive integer, dimensionality of the output space.
        return_sequences=return_sequences,  # Boolean. Whether to return the last output in the output sequence, or the full sequence.
        input_shape=input_shape,
        stateful=stateful))
    if dropout > 0:
        regressor.add(Dropout(dropout))

    # Adding additional LSTM layers
    for i in range(layers-1, 0, -1):
        print('Adding additional LSTM layers {}'.format(i))
        return_sequences = False if i == 1 else True
        regressor.add(
            LSTM(units=output_dim, 
            return_sequences=return_sequences,
            stateful=stateful))
        if dropout > 0:
            regressor.add(Dropout(dropout))

    # Adding Dense layer to aggregate the data from the prediction vector into a single value
    regressor.add(Dense(units=output))

    # Adding Linear Activation function
    activation = 'linear'
    regressor.add(Activation(activation))

    # Compiling the RNN
    regressor.compile(
        optimizer=optimizer,
        metrics=[metrics.mse],
        loss='mean_squared_error')

    print(regressor.summary())
    return regressor
