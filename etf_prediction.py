# -*- coding: utf-8 -*-

# Importing the libraries
import numpy as np
import pandas as pd

from keras.models import load_model

from dataloader import DataLoader

# ETF list
# stock_ids = ['0050','0051','0052','0053','0054','0055','0056','0057','0058','0059','006201','006203','006204','006208','00690','00692','00701','00713']
stock_ids = ['0050']
# assert(len(stock_ids)==18)


def convert_up(pct_change):
    if pct_change > 0:
        return 1
    elif pct_change < 0:
        return -1
    else:
        return 0


# Prediction
predictions = []
for stock_id in stock_ids:
    pred = {}

    loader = DataLoader()
    last_price = loader.data_last_price(int(stock_id))
    X_test = loader.data_for_prediction(int(stock_id))

    regressor = load_model('rnn_etf_{}.h5'.format(int(stock_id)))

    # Normalized prediction
    predicted_price = regressor.predict(X_test).transpose()

    # Inversed transform prediction
    predicted_price = loader.inverse_transform_prediction(predicted_price)

    # Calculating Up/Down trend
    prices = np.concatenate((last_price, predicted_price))
    ud = pd.DataFrame(prices).pct_change().dropna().values
    
    pred.update({
        'ETFid': stock_id,
        'Mon_ud': convert_up(ud[0][0]),
        'Mon_cprice': '{0:.2f}'.format(predicted_price[0][0]),
        'Tue_ud': convert_up(ud[1][0]),
        'Tue_cprice': '{0:.2f}'.format(predicted_price[1][0]),
        'Wed_ud': convert_up(ud[2][0]),
        'Wed_cprice': '{0:.2f}'.format(predicted_price[2][0]),
        'Thu_ud': convert_up(ud[3][0]),
        'Thu_cprice': '{0:.2f}'.format(predicted_price[3][0]),
        'Fri_ud': convert_up(ud[4][0]),
        'Fri_cprice': '{0:.2f}'.format(predicted_price[4][0])
        })

    predictions.append(pred)

df = pd.DataFrame(predictions)
df.to_csv(
    'predictions.csv',
    index=False,
    columns=['ETFid','Mon_ud','Mon_cprice','Tue_ud','Tue_cprice','Wed_ud','Wed_cprice','Thu_ud','Thu_cprice','Fri_ud','Fri_cprice'])
