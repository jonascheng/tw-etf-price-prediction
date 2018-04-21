# -*- coding: utf-8 -*-

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from hyperopt import Trials, tpe
from hyperas import optim


def data():
    """
    Data providing function:

    Make sure to have every relevant import statement included here and return data as
    used in model function below. This function is separated from model() so that hyperopt
    won't reload data for each evaluation run.
    """
    import dataloader

    # Normalized
    loader = dataloader.DataLoader()
    X_train, y_train, X_test, y_test = loader.data_last_ndays_for_test(50, ndays=240)
    return X_train, y_train, X_test, y_test


def model(X_train, y_train, X_test, y_test):
    """
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """
    # Importing the Keras libraries and packages
    from keras.callbacks import EarlyStopping
    # Importing the hyperopt libraries and packages
    from hyperopt import STATUS_OK
    from hyperas.distributions import choice, uniform

    from model import create_stateless_lstm_model

    nb_epoch = {{choice([1, 10, 100])}}
    batch_size = {{choice([1, 32])}}

    layers = {{choice([1, 2, 3])}}
    output_dim = {{choice([50, 60, 90])}}
    optimizer = {{choice(['rmsprop', 'adam', 'sgd'])}}
    dropout = {{choice([0, 0.2, 0.4])}}

    regressor = create_stateless_lstm_model(
        X_train,
        y_train,
        layers=layers, 
        output_dim=output_dim, 
        optimizer=optimizer,
        dropout=dropout)
    
    # Defining early stopping criteria
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.0000001, patience=10, verbose=1, mode='min')

    # Collecting callback list
    callbacks_list = [earlyStopping]

    # Fitting the RNN to the Training set
    regressor.fit(
        X_train, 
        y_train, 
        epochs=nb_epoch, 
        batch_size=batch_size, 
        validation_data=(X_test, y_test),
        callbacks=callbacks_list)

    # Evaluating the model    
    score, mse = regressor.evaluate(X_test, y_test)
    print('Test score: {}, mse: {}'.format(score, mse))

    return {'loss': mse, 'status': STATUS_OK, 'model': regressor}

best_run, best_model = optim.minimize(model=model, data=data, algo=tpe.suggest, max_evals=5, trials=Trials())
_, _, X_test, y_test = data()
print('Evalutation of best performing model:')
print(best_model.evaluate(X_test, y_test))
print("Best performing model chosen hyper-parameters:")
print(best_run)

best_model.save('rnn_etf_50.h5')
