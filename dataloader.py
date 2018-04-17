# -*- coding: utf-8 -*-

# Importing the libraries
import copy
import numpy as np
import pandas as pd

# Spliting dataset into training and testing sets
from sklearn.model_selection import train_test_split


class DataLoader():
    def __init__(self, filepath, normalize=True):
        # Tuned params and variables
        # number of sequence data
        self.look_back = 60
        self.normalize = normalize
        # Importing the dataset
        self.history = pd.read_csv(filepath, encoding='big5-hkscs')
        print('historical data is loaded from {}'.format(filepath))

    def __normalize_windows(self, window_data):
        if self.normalize is True:
            # Normalize dataset to improve the convergence
            # Normalize each value to reflect the percentage changes from starting point
            normalized_data = []
            for window in window_data:
                normalized_window = [((float(p) / float(window[0])) - 1) for p in window]
                normalized_data.append(normalized_window)            
            window_data = normalized_data
        return window_data

    def __train_test_split(self, Xy, ndays):
        # Spliting dataset into training and testing sets    
        if ndays > 0:
            # Historical price for regression
            X = Xy[:, :-1]
            # Target price for regression
            y = Xy[:, -1:]
            # Select the last 5 working date for testing and the others for training.
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

    def __data(self, stock_id, ndays):
        # Extracting/Filtering the training dataset by stock_id
        dataset = self.history.loc[self.history['代碼'] == stock_id]
        # Taking 收盤價 as a predictor
        dataset = dataset.iloc[:, 6:7].values
        # Composing time sequence dataset with timesteps + target
        timesequence = []
        for i in range(self.look_back, len(dataset)):
            timesequence.append(dataset[i - self.look_back:i + 1, 0])
        # Normalize dataset if needed
        ori_Xy = copy.deepcopy(timesequence)
        Xy = self.__normalize_windows(timesequence)
        # Converting array of list to numpy array
        ori_Xy = np.array(ori_Xy)
        Xy = np.array(Xy)
        # Spliting dataset into training and testing sets
        self.X_ori_train, self.X_ori_test, self.y_ori_train, self.y_ori_test = self.__train_test_split(ori_Xy, ndays)
        X_train, X_test, y_train, y_test = self.__train_test_split(Xy, ndays)

        return X_train, y_train, X_test, y_test

    def data_last_ndays_for_test(self, stock_id, ndays):
        return self.__data(stock_id, ndays)

    def data(self, stock_id):
        return self.__data(stock_id, ndays=0)

    def ori_data(self):
        return self.X_ori_train, self.y_ori_train, self.X_ori_test, self.y_ori_test

    def inverse_transform_prediction(self, prediction):
        if self.normalize is True:
            prediction = (prediction + 1) * self.X_ori_test[:, 0]
        return prediction
