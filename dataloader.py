# -*- coding: utf-8 -*-

# Importing the libraries
import copy
import numpy as np
import pandas as pd

from abc import ABCMeta
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
# Spliting dataset into training and testing sets
from sklearn.model_selection import train_test_split


class DataLoaderBase(metaclass=ABCMeta):
    def __init__(self, filepath):
        # Tuned params and variables
        # number of sequence data
        self.look_back = 50
        # Importing the dataset
        self.history = pd.read_csv(filepath, encoding='big5-hkscs')
        print('historical data is loaded from {}'.format(filepath))

    def _series_to_supervised(self, series, n_in, n_out=1):
        """
        Frame a time series as a supervised learning dataset.
        Arguments:
            series: Sequence of observations as a NumPy array.
            n_in: Number of observations as input (X).
            n_out: Number of observations as output (y).
        Returns:
            NumPy array of series for supervised learning.
        """
        assert type(series) is np.ndarray, 'unexpected type of series: {}'.format(type(series))
        assert(series.shape[0] > n_in + n_out)
        assert(series.shape[1] == 1)
        # Composing time sequence dataset with timesteps + target
        supervised = []
        for i in range(n_in, len(series) - n_out + 1):
            supervised.append(series[i - n_in:i + n_out, 0])
        return supervised

    def _train_test_split(self, Xy, ndays):
        """
        Split supervised learning dataset into train and test sets
        Arguments:
            Xy: Two dimensions of sequence observations as a NumPy array.
            ndays: 
                If ndays > 0, reserve the last ndays (axis 0) of sequence observations as test set.
                If ndays = 0, random reserve 20% of sequence observations as test set.
        """
        assert type(Xy) is np.ndarray, 'unexpected type of series: {}'.format(type(Xy))
        assert(Xy.shape[0] > ndays)
        # Spliting dataset into training and testing sets    
        if ndays > 0:
            # Historical price for regression
            X = Xy[:, :-1]
            # Target price for regression
            y = Xy[:, -1:]
            # Select the last ndays working date for testing and the others for training.
            X_train = X[:-ndays, :]
            y_train = y[:-ndays, :]
            X_test = X[-ndays:, :]
            y_test = y[-ndays:, :]
        else:
            X = Xy[:, :-1]
            y = Xy[:, -1:]
            # Select 20% of the data for testing and 80% for training.
            # Shuffle the data in order to train in random order.
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0) 

        # Reshape the inputs from 1 dimenstion to 3 dimension
        # X_train.shape[0]: batch_size which is number of observations
        # X_train.shape[1]: timesteps which is look_back
        # 1: input_dim which is number of predictors
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        return X_train, X_test, y_train, y_test


class DataLoader2(DataLoaderBase):
    def __init__(self, filepath):
        super(DataLoader2, self).__init__(filepath)

    def __normalize_windows(self, window_data):
        # Normalize dataset to improve the convergence
        # Normalize each value to reflect the percentage changes from previous point
        df = DataFrame(window_data)
        df = df.pct_change(axis=1).fillna(0)
        return df.values

    def __data(self, stock_id, ndays):
        # Extracting/Filtering the training dataset by stock_id
        dataset = self.history.loc[self.history['代碼'] == stock_id]
        # Taking 收盤價 as a predictor
        dataset = dataset.iloc[:, 6:7].values
        # Transforming time series dataset into supervised dataset
        supervised = self._series_to_supervised(dataset, self.look_back)
        # Normalize dataset if needed
        ori_Xy = copy.deepcopy(supervised)
        Xy = self.__normalize_windows(supervised)
        # Converting array of list to numpy array
        ori_Xy = np.array(ori_Xy)
        Xy = np.array(Xy)
        # Spliting dataset into training and testing sets
        self.X_ori_train, self.X_ori_test, self.y_ori_train, self.y_ori_test = self._train_test_split(ori_Xy, ndays)
        X_train, X_test, y_train, y_test = self._train_test_split(Xy, ndays)

        return X_train, y_train, X_test, y_test

    def data_last_ndays_for_test(self, stock_id, ndays):
        return self.__data(stock_id, ndays)

    def data(self, stock_id):
        return self.__data(stock_id, ndays=0)

    def ori_data(self):
        return self.X_ori_train, self.y_ori_train, self.X_ori_test, self.y_ori_test

    def inverse_transform_prediction(self, prediction):
        prediction = (prediction + 1) * self.X_ori_test[:, -1]
        return prediction


class DataLoader(DataLoaderBase):
    def __init__(self, filepath):
        super(DataLoader, self).__init__(filepath)

    def __normalize_windows(self, window_data):
        # Normalize dataset to improve the convergence
        # Normalize each value to reflect the percentage changes from starting point
        df = DataFrame(window_data)
        df = df.div(df[0], axis=0) - 1
        return df.values
        
    def __data(self, stock_id, ndays):
        # Extracting/Filtering the training dataset by stock_id
        dataset = self.history.loc[self.history['代碼'] == stock_id]
        # Taking 收盤價 as a predictor
        dataset = dataset.iloc[:, 6:7].values
        # Transforming time series dataset into supervised dataset
        supervised = self._series_to_supervised(dataset, self.look_back)
        # Normalize dataset if needed
        ori_Xy = copy.deepcopy(supervised)
        Xy = self.__normalize_windows(supervised)
        # Converting array of list to numpy array
        ori_Xy = np.array(ori_Xy)
        Xy = np.array(Xy)
        # Spliting dataset into training and testing sets
        self.X_ori_train, self.X_ori_test, self.y_ori_train, self.y_ori_test = self._train_test_split(ori_Xy, ndays)
        X_train, X_test, y_train, y_test = self._train_test_split(Xy, ndays)

        return X_train, y_train, X_test, y_test

    def data_last_ndays_for_test(self, stock_id, ndays):
        return self.__data(stock_id, ndays)

    def data(self, stock_id):
        return self.__data(stock_id, ndays=0)

    def ori_data(self):
        return self.X_ori_train, self.y_ori_train, self.X_ori_test, self.y_ori_test

    def inverse_transform_prediction(self, prediction):
        prediction = (prediction + 1) * self.X_ori_test[:, 0]
        return prediction
