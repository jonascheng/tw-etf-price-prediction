# -*- coding: utf-8 -*-

# Importing the libraries
from keras.models import load_model

from dataloader import DataLoader
from visualizer import Visualizer

loader = DataLoader()
loader.data_last_ndays_for_test(50, ndays=5)
X_train, y_train, X_test, y_test = loader.data_last_ndays_for_test(50, ndays=1)
X_ori_train, y_ori_train, X_ori_test, y_ori_test = loader.ori_data()

regressor = load_model('rnn_etf_50.h5')

# Normalized prediction
real_price = y_test.transpose()
predicted_price = regressor.predict(X_test).transpose()
print('Normalized prediction\n{}'.format(predicted_price))
Visualizer.show(real_price, predicted_price, 'Normalized Stock Price Prediction')

# Inversed transform prediction
real_price2 = y_ori_test.transpose()
predicted_price2 = loader.inverse_transform_prediction(predicted_price)
print('Inversed prediction\n{}'.format(predicted_price2))
Visualizer.show(real_price2, predicted_price2, 'Stock Price Prediction')
