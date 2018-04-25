# -*- coding: utf-8 -*-

# Importing the libraries
from keras.models import load_model

from dataloader import DataLoader
from visualizer import Visualizer
from util import get_model_name

stock_id = '0050'

loader = DataLoader()
last_price = loader.data_last_price(int(stock_id))
X_train, y_train, X_test, y_test = loader.data_last_ndays_for_test(int(stock_id), ndays=5)
X_ori_train, y_ori_train, X_ori_test, y_ori_test = loader.ori_data()

# regressor = load_model(get_model_name(stock_id))
regressor = load_model('rnn_etf_{}.h5'.format(int(stock_id)))

# Normalized prediction
real_price = y_test
predicted_price = regressor.predict(X_test)

real_price = np.concatenate((real_price[0], np.array(real_price)[1:, -1]))
predicted_price = np.concatenate((predicted_price[0], np.array(predicted_price)[1:, -1]))

print('Normalized prediction\n{}'.format(predicted_price))
Visualizer.show(real_price, predicted_price, 'Normalized Stock Price Prediction')

# Inversed transform prediction
real_price2 = y_ori_test.transpose()
predicted_price2 = loader.inverse_transform_prediction(predicted_price)
print('Inversed prediction\n{}'.format(predicted_price2))
Visualizer.show(real_price2, predicted_price2, 'Stock Price Prediction')
