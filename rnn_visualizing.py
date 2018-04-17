# -*- coding: utf-8 -*-

# Importing the libraries
from keras.models import load_model

from dataloader import DataLoader
from visualizer import Visualizer

loader = DataLoader('TBrain_Round2_DataSet_20180331/tetfp.csv', normalize=True)

X_train, y_train, X_test, y_test = loader.data_last_5_for_test(50)
X_ori_train, y_ori_train, X_ori_test, y_ori_test = loader.ori_data()

regressor = load_model('rnn_etf_50.h5')

# Normalized prediction
real_price = y_test
predicted_price = regressor.predict(X_test)
Visualizer.show(real_price, predicted_price, 'Normalized Stock Price Prediction')

# Inversed transform prediction
real_price = y_ori_test
predicted_price = loader.inverse_transform_prediction(predicted_price)
Visualizer.show(real_price, predicted_price, 'Stock Price Prediction')
