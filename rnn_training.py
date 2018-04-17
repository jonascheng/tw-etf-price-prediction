# -*- coding: utf-8 -*-

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from hyperopt import Trials, tpe
from hyperas import optim


'''
Data providing function:

Make sure to have every relevant import statement included here and return data as
used in model function below. This function is separated from model() so that hyperopt
won't reload data for each evaluation run.
'''
def data():
    from dataloader import DataLoader
    loader = DataLoader('TBrain_Round2_DataSet_20180331/tetfp.csv', normalize=True)
    X_train, y_train, X_test, y_test = loader.data(50)
    return X_train, y_train, X_test, y_test


'''
Model providing function:

Create Keras model with double curly brackets dropped-in as needed.
Return value has to be a valid python dictionary with two customary keys:
    - loss: Specify a numeric evaluation metric to be minimized
    - status: Just use STATUS_OK and see hyperopt documentation if not feasible
The last one is optional, though recommended, namely:
    - model: specify the model just created so that we can later use it again.
'''
def model(X_train, y_train, X_test, y_test):
    # Importing the Keras libraries and packages
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers import Dropout
    from keras.layers import Activation    
    # Importing the hyperopt libraries and packages
    from hyperopt import STATUS_OK
    from hyperas.distributions import choice, uniform, conditional

    # Initialising the RNN
    regressor = Sequential()
    
    # Adding the first LSTM layer
    output_dim = {{choice([50, 60, 90])}}
    regressor.add(LSTM(units=output_dim, return_sequences=False, input_shape=(X_train.shape[1], 1)))
    
    # Adding Dense layer to aggregate the data from the prediction vector into a single value
    regressor.add(Dense(units=1))
    
    # Adding Linear Activation function
    activation = 'linear'
    regressor.add(Activation(activation))
    
    # Compiling the RNN
    optimizer = {{choice(['rmsprop', 'adam', 'sgd'])}}
    regressor.compile(
        optimizer=optimizer,
        metrics=['accuracy'],
        loss='mean_squared_error')
    
    # Fitting the RNN to the Training set
    nb_epoch = {{choice([1, 10, 100])}}
    batch_size = {{choice([1, 32])}}
    regressor.fit(
        X_train, 
        y_train, 
        epochs=nb_epoch, 
        batch_size=batch_size, 
        validation_data=(X_test, y_test))

    # Evaluating the model    
    score, acc = regressor.evaluate(X_test, y_test)
    print('Test score: {}, accuracy: {}'.format(score, acc))
    
    return {'loss': -acc, 'status': STATUS_OK, 'model': regressor}

best_run, best_model = optim.minimize(model=model, data=data, algo=tpe.suggest, max_evals=5, trials=Trials())
_, _, X_test, y_test = data()
print('Evalutation of best performing model:')
print(best_model.evaluate(X_test, y_test))
print("Best performing model chosen hyper-parameters:")
print(best_run)

best_model.save('rnn_etf_50.h5')

