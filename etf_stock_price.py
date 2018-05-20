# -*- coding: utf-8 -*-

# Importing the libraries
import sys
import numpy as np
import pandas as pd
import subprocess

from keras.models import load_model

from dataloader import DataForStatelessModel, DataForStatelessModelMoreFeatures
from util import get_model_name

# ETF list
stock_ids = [
    '0050', '0051', '0052', '0053', '0054',
    '0055', '0056', '0057', '0058', '0059',
    '006201', '006203', '006204', '006208', '00690',
    '00692', '00701', '00713']
assert(len(stock_ids)==18)
# stock_ids = ['0052', '0054', '006201', '006208', '00690', '00692', '00701', '00713']


def convert_ud(pct_change):
    if pct_change > 0:
        return 1
    elif pct_change < 0:
        return -1
    else:
        return 0


def predict():
    predictions = []
    for stock_id in stock_ids:
        print('Predicting stock {}...'.format(stock_id))

        pred = {}

        loader = DataForStatelessModel(int(stock_id))
        # Querying the last day of stock price
        last_price = loader.data_last_price(int(stock_id))
        X_ori_test, X_test = loader.data_for_prediction(int(stock_id))

        regressor = load_model(get_model_name(stock_id))

        # Normalized prediction
        predicted_price = regressor.predict(X_test).transpose()

        # Inversed transform prediction
        predicted_price = loader.inverse_transform_prediction(predicted_price)

        # Calculating Up/Down trend
        prices = np.concatenate((last_price, predicted_price))
        ud = pd.DataFrame(prices).round(2).pct_change().dropna().values

        pred.update({
            'ETFid': stock_id,
            'Mon_ud': convert_ud(ud[0][0]),
            'Mon_cprice': '{0:.2f}'.format(predicted_price[0][0]),
            'Tue_ud': convert_ud(ud[1][0]),
            'Tue_cprice': '{0:.2f}'.format(predicted_price[1][0]),
            'Wed_ud': convert_ud(ud[2][0]),
            'Wed_cprice': '{0:.2f}'.format(predicted_price[2][0]),
            'Thu_ud': convert_ud(ud[3][0]),
            'Thu_cprice': '{0:.2f}'.format(predicted_price[3][0]),
            'Fri_ud': convert_ud(ud[4][0]),
            'Fri_cprice': '{0:.2f}'.format(predicted_price[4][0])
        })

        predictions.append(pred)
        print('Predicting stock {} completed'.format(stock_id))

    df = pd.DataFrame(predictions)
    df.to_csv(
        'etf_predictions.csv',
        index=False,
        columns=['ETFid', 'Mon_ud', 'Mon_cprice', 'Tue_ud', 'Tue_cprice', 'Wed_ud', 'Wed_cprice', 'Thu_ud', 'Thu_cprice', 'Fri_ud', 'Fri_cprice'])


def train():
    for stock_id in stock_ids:
        print('Training stock {}...'.format(stock_id))
        cmd = 'python stateless_training_more_features.py {} {}'.format(stock_id, get_model_name(stock_id))
        fhandle = open('etf_stock_price_train_{}.txt'.format(stock_id), 'w')
        proc = subprocess.Popen(
            cmd,
            shell=True,
            close_fds=True,
            stderr=fhandle,
            stdout=fhandle)
        proc.wait()
        fhandle.close()
        print('Training stock {} completed'.format(stock_id))


if __name__ == '__main__':
    if (len(sys.argv) > 0):
        if sys.argv[1] == 'predict':
            predict()
        else:
            train()
