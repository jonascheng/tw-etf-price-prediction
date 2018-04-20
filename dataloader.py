# -*- coding: utf-8 -*-

# Importing the libraries
import copy
import numpy as np
import pandas as pd

from abc import ABCMeta
from sklearn.preprocessing import MinMaxScaler

from util import load_csv, query_close_price, series_to_supervised, normalize_windows, train_test_split


class DataLoaderBase(metaclass=ABCMeta):
    # Tuned params and variables
    # number of sequence data
    look_back = 50
    
    def __init__(self, filepath):
        # Importing the dataset
        self.history = load_csv(filepath)


class DataLoader(DataLoaderBase):
    def __init__(self, filepath):
        super(DataLoader, self).__init__(filepath)
        
    def __data(self, stock_id, ndays):
        # Taking 收盤價 as a predictor
        dataset = query_close_price(self.history, stock_id)
        # Transforming time series dataset into supervised dataset
        supervised = series_to_supervised(dataset, n_in=self.look_back, n_out=1)
        # Normalize dataset if needed
        ori_Xy = copy.deepcopy(supervised)
        Xy = normalize_windows(supervised)
        # Converting array of list to numpy array
        ori_Xy = np.array(ori_Xy)
        Xy = np.array(Xy)
        # Feature Scaling in feature range (0, 1)
        # self.sc = MinMaxScaler(feature_range = (0, 1))
        # self.sc.fit(Xy.reshape(Xy.shape[0]*Xy.shape[1], 1))
        # Xy = self.sc.transform(Xy)
        # Spliting dataset into training and testing sets
        self.X_ori_train, self.X_ori_test, self.y_ori_train, self.y_ori_test = train_test_split(ori_Xy, test_samples=ndays, num_forecasts=1)
        X_train, X_test, y_train, y_test = train_test_split(Xy, test_samples=ndays, num_forecasts=1)

        return X_train, y_train, X_test, y_test

    def data_last_ndays_for_test(self, stock_id, ndays):
        return self.__data(stock_id, ndays)

    def data(self, stock_id):
        return self.__data(stock_id, ndays=0)

    def ori_data(self):
        return self.X_ori_train, self.y_ori_train, self.X_ori_test, self.y_ori_test

    def inverse_transform_prediction(self, prediction):
        # prediction = self.sc.inverse_transform(prediction)
        prediction = (prediction + 1) * self.X_ori_test[:, 0]
        return prediction
