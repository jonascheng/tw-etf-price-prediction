# -*- coding: utf-8 -*-
# 1. DataForStatelessModel
# 2. Open, High, Low, Average, and Close Price
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from hyperopt import Trials, tpe
from hyperas import optim

import dataloader
from util import visualize_model


def data():
    """
    Data providing function:

    Make sure to have every relevant import statement included here and return data as
    used in model function below. This function is separated from model() so that hyperopt
    won't reload data for each evaluation run.
    """
    # workaround to dynamically change stock_id
    with open('.processing_stock_id', 'r') as file:
        stock_id = file.read()
    print('Loading dataset for {}'.format(stock_id))
    import dataloader
    loader = dataloader.DataForStatelessModelMoreFeatures()
    if stock_id in ['00690', '00692', '00701', '00713']:
        ndays = 30
    else:
        ndays = 240
    X_train, y_train, X_test, y_test = loader.data_last_ndays_for_test(int(stock_id), ndays=ndays)
    return X_train, y_train, X_test, y_test

"""
# 20180429 Open/Close/Avg
nb_epoch = {{choice([1, 10, 50])}} => 50
batch_size = 32
layers = {{choice([2, 3, 4])}} => 2, 3
output_dim = {{choice([50, 60, 90])}} => 50, 60
optimizer = {{choice(['rmsprop', 'sgd', 'adam'])}}
dropout = 0.2
"""

"""
# 20180506 Open/Close/Avg/High/Low
nb_epoch = {{choice([50, 100, 125])}} => 100, 125
batch_size = {{choice([32, 128])}} => all
layers = {{choice([2, 3, 4])}} => 2, 3
output_dim = {{choice([50, 60, 70, 256])}} => 50, 60, 256
optimizer = {{choice(['rmsprop', 'adam'])}} => all
dropout = {{choice([0.2, 0.3])}} => all
"""

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
    from hyperas.distributions import choice

    from model import create_stateless_lstm_model
    from util import SavePredictionCallback

    nb_epoch = {{choice([50, 100, 125, 150])}}
    batch_size = {{choice([32, 128])}}
    layers = {{choice([1, 2, 3, 4])}}
    output_dim = {{choice([40, 50, 60, 256])}}
    optimizer = {{choice(['rmsprop', 'sgd', 'adam'])}}
    dropout = {{choice([0.1, 0.2, 0.3, 0.4])}}

    regressor = create_stateless_lstm_model(
        X_train,
        y_train,
        layers=layers,
        output_dim=output_dim,
        optimizer=optimizer,
        dropout=dropout)

    # Defining early stopping criteria
    # earlyStopping = EarlyStopping(monitor='loss', min_delta=0.00001, patience=10, verbose=1, mode='min')

    # Defining intermediate prediction result
    predicted_prefix = 'stateless_mf_etf_predicted_epoch{}_batch{}_layers{}_output{}_dropout{}_opt{}'.format(nb_epoch, batch_size, layers, output_dim, dropout, optimizer)
    savePrediction = SavePredictionCallback(predicted_prefix, X_test)

    # Collecting callback list
    # callbacks_list = [earlyStopping, savePrediction]
    callbacks_list = [savePrediction]

    # Fitting the RNN to the Training set
    real_price = y_test
    real_price = np.concatenate((real_price[0], np.array(real_price)[1:, -1]))
    # np.save('stateless_real_price.npy', real_price)
    regressor.fit(
        X_train,
        y_train,
        epochs=nb_epoch,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=callbacks_list,
        shuffle=False)

    # Evaluating the model
    score, mse = regressor.evaluate(X_test, y_test, batch_size=batch_size)
    print('Test score: {}, mse: {}'.format(score, mse))

    return {'loss': mse, 'status': STATUS_OK, 'model': regressor}


def start_training(stock_id, trained_model):
    # workaround to dynamically change stock_id
    with open('.processing_stock_id', 'w') as file:
        file.write(str(stock_id))
    best_run, best_model = optim.minimize(
        model=model,
        data=data,
        algo=tpe.suggest,
        max_evals=5,
        trials=Trials())
    _, _, X_test, y_test = data()
    print('Evalutation of best performing model for stock id {}:'.format(stock_id))
    print(best_model.evaluate(X_test, y_test))
    print('Best performing model for stock id {} chosen hyper-parameters:'.format(stock_id))
    print(best_run)
    best_model.save(trained_model)
    # plotting best performing mode
    if stock_id in ['00690', '00692', '00701', '00713']:
        ndays = 30
    else:
        ndays = 240
    plot_prefix = 'stateless_mf_etf_stock_{}'.format(stock_id)
    for key, value in best_run.items():
        plot_prefix = '_'.join((plot_prefix, str(key), str(value)))
    loader = dataloader.DataForStatelessModelMoreFeatures()
    visualize_model(loader, best_model, stock_id, ndays, plot_prefix)

if __name__ == '__main__':
    stock_id = 6201
    start_training(stock_id, 'stateless_etf_{}.h5'.format(stock_id))