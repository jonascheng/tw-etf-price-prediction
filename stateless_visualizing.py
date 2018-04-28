# -*- coding: utf-8 -*-

# Importing the libraries
import numpy as np
from keras.models import load_model

from dataloader import DataForStatelessModel, DataForStatelessModelMoreFeatures
from util import get_model_name, plot_real_predicted_stock_price

stock_id = '0050'
ndays = 240

#loader = DataForStatelessModel()
loader = DataForStatelessModelMoreFeatures()
last_price = loader.data_last_price(int(stock_id))
X_train, y_train, X_test, y_test = loader.data_last_ndays_for_test(int(stock_id), ndays=ndays)
X_ori_train, y_ori_train = loader.ori_train_data()
X_ori_test, y_ori_test = loader.ori_test_data()

# regressor = load_model(get_model_name(stock_id))
regressor = load_model('stateless_etf_{}.h5'.format(int(stock_id)))

# Normalized prediction
real_price = y_test
predicted_price = regressor.predict(X_test)
predicted_price1 = predicted_price
predicted_price2 = predicted_price

if ndays > 1:
    real_price = np.concatenate((real_price[0], np.array(real_price)[1:, -1]))
    predicted_price1 = np.concatenate((predicted_price1[0], np.array(predicted_price1)[1:, -1]))
else:
    real_price = real_price.transpose()
    predicted_price1 = predicted_price1.transpose()
    
#print('Normalized prediction\n{}'.format(predicted_price1))
plot_real_predicted_stock_price(real_price, predicted_price1, 'Normalized Stock Price Prediction')

# Inversed transform prediction
real_price2 = y_ori_test
predicted_price2 = loader.inverse_transform_prediction(predicted_price)

if ndays > 1:
    real_price2 = np.concatenate((real_price2[0], np.array(real_price2)[1:, -1]))
    predicted_price2 = np.concatenate((predicted_price2[0], np.array(predicted_price2)[1:, -1]))
else:
    real_price2 = real_price2.transpose()
    predicted_price2 = predicted_price2.transpose()

#print('Inversed prediction\n{}'.format(predicted_price2))
plot_real_predicted_stock_price(real_price2, predicted_price2, 'Stock Price Prediction')
