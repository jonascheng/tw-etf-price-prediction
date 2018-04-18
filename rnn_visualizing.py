# -*- coding: utf-8 -*-

# Importing the libraries
from keras.models import load_model

from dataloader import DataLoader, DataLoader2
from visualizer import Visualizer

loader = DataLoader('TBrain_Round2_DataSet_20180331/tetfp.csv')
#loader = DataLoader2('TBrain_Round2_DataSet_20180331/tetfp.csv')
loader.data_last_ndays_for_test(50, ndays=5)
X_train, y_train, X_test, y_test = loader.data_last_ndays_for_test(50, ndays=10)
X_ori_train, y_ori_train, X_ori_test, y_ori_test = loader.ori_data()

regressor = load_model('rnn_etf_50.h5')

# Normalized prediction
real_price = y_test
predicted_price = regressor.predict(X_test)
Visualizer.show(real_price, predicted_price, 'Normalized Stock Price Prediction')

# Inversed transform prediction
real_price2 = y_ori_test
predicted_price2 = loader.inverse_transform_prediction(predicted_price)
Visualizer.show(real_price2, predicted_price2, 'Stock Price Prediction')
