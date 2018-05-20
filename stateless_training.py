# -*- coding: utf-8 -*-
# 1. DataForStatelessModel
# 2. Close Price
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from argparse import ArgumentParser

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
    loader = dataloader.DataForStatelessModel(int(stock_id))
    if stock_id in ['00690', '00692', '00701', '00713']:
        ndays = 30
    else:
        ndays = 240
    X_train, y_train, X_test, y_test = loader.data_last_ndays_for_test(int(stock_id), ndays=ndays)
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
    from util import SavePredictionCallback

    nb_epoch = 200
    batch_size = 128
    layers = {{choice([4, 5])}}
    output_dim = {{choice([128, 256])}}
    optimizer = 'adam'
    dropout = 0.2

    regressor = create_stateless_lstm_model(
        X_train,
        y_train,
        layers=layers,
        output_dim=output_dim,
        optimizer=optimizer,
        dropout=dropout)

    # Defining early stopping criteria
    earlyStopping = EarlyStopping(monitor='loss', min_delta=0.000001, patience=25, verbose=1, mode='min')

    # Defining intermediate prediction result
    predicted_prefix = 'stateless_predicted_epoch{}_batch{}_layers{}_output{}_dropout{}_{}'.format(nb_epoch, batch_size, layers, output_dim, dropout, optimizer)
    savePrediction = SavePredictionCallback(predicted_prefix, X_test)

    # Collecting callback list
    callbacks_list = [earlyStopping, savePrediction]

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
    plot_prefix = 'stateless_etf_stock_{}'.format(stock_id)
    for key, value in best_run.items():
        plot_prefix = '_'.join((plot_prefix, str(key), str(value)))
    loader = dataloader.DataForStatelessModel(int(stock_id))
    visualize_model(loader, best_model, stock_id, ndays, plot_prefix)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('stock_id', help='stock id')
    parser.add_argument('trained_model', help='output model name')
    args = parser.parse_args()
    print('start training process for...')
    print('stock id: {}'.format(args.stock_id))
    print('output model name: {}'.format(args.trained_model))
    start_training(args.stock_id, args.trained_model)
