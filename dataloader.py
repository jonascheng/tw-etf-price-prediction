# -*- coding: utf-8 -*-

# Importing the libraries
import copy
import numpy as np
import pandas as pd

import abc
from sklearn.preprocessing import MinMaxScaler

import settings
from util import load_csv, series_to_supervised, normalize_windows, train_test_split, predict_split
from util import query_close_price, query_open_price, query_volume, query_high_price, query_low_price
from util import query_avg_price


class DataLoader(abc.ABC):
    def __init__(self):
        # Importing the dataset
        filepath = '{}/tetfp.csv'.format(settings.DATASET_PATH)
        self.history = load_csv(filepath)


class DataForStatelessModel(DataLoader):
    # Tuned params and variables
    # number of sequence data
    look_back = 50
    look_forward = 5

    def __init__(self):
        super(DataForStatelessModel, self).__init__()

    def __data(self, stock_id, ndays):
        # Taking 收盤價 as a predictor
        dataset = query_close_price(self.history, stock_id)
        # Transforming time series dataset into supervised dataset
        supervised = series_to_supervised(dataset, n_in=self.look_back, n_out=self.look_forward)
        # Normalize dataset if needed
        ori_Xy = copy.deepcopy(supervised)
        Xy = normalize_windows(supervised)
        # Converting array of list to numpy array
        ori_Xy = np.array(ori_Xy)
        Xy = np.array(Xy)
        # Spliting dataset into training and testing sets
        self.X_ori_train, self.X_ori_test, self.y_ori_train, self.y_ori_test = train_test_split(ori_Xy, test_samples=ndays, num_forecasts=self.look_forward)
        X_train, X_test, y_train, y_test = train_test_split(Xy, test_samples=ndays, num_forecasts=self.look_forward)

        return X_train, y_train, X_test, y_test

    def data_last_price(self, stock_id):
        # Taking 收盤價 as a predictor
        dataset = query_close_price(self.history, stock_id)
        return dataset[-1:]

    def data_for_prediction(self, stock_id):
        # Taking 收盤價 as a predictor
        dataset = query_close_price(self.history, stock_id)
        # Transforming time series dataset into supervised dataset
        supervised = series_to_supervised(dataset, n_in=self.look_back, n_out=0)
        # Normalize dataset if needed
        ori_Xy = copy.deepcopy(supervised)
        Xy = normalize_windows(supervised)
        # Converting array of list to numpy array
        ori_Xy = np.array(ori_Xy)
        Xy = np.array(Xy)

        # Spliting dataset into predicting sets
        self.X_ori_test = predict_split(ori_Xy)
        X_test = predict_split(Xy)

        return self.X_ori_test, X_test

    def data_last_ndays_for_test(self, stock_id, ndays):
        return self.__data(stock_id, ndays)

    def data(self, stock_id):
        return self.__data(stock_id, ndays=0)

    def ori_train_data(self):
        return self.X_ori_train, self.y_ori_train

    def ori_test_data(self):
        return self.X_ori_test, self.y_ori_test

    def inverse_transform_prediction(self, prediction):
        prediction = (prediction + 1) * self.X_ori_test[:, 0]
        return prediction


class DataForStatelessModelMoreFeatures(DataLoader):
    # Tuned params and variables
    # number of sequence data
    look_back = 50
    look_forward = 5

    def __init__(self):
        super(DataForStatelessModelMoreFeatures, self).__init__()

    def __set_look_back(self, stock_id):        
        if stock_id in [57, 58]:
            self.look_back = 110
        elif stock_id in [50, 51, 53, 6203, 6204]:
            self.look_back = 100
        elif stock_id in [54, 59]:
            self.look_back = 90
        elif stock_id in [55]:
            self.look_back = 80
        elif stock_id in [56, 6201]:
            self.look_back = 70
        elif stock_id in [690]:
            self.look_back = 18
        elif stock_id in [701]:
            self.look_back = 13
        elif stock_id in [713]:
            self.look_back = 6
        else:
            # 52, 6208, 692
            self.look_back = 50
        print('set look back to {} for stock {}'.format(self.look_back, stock_id))

    def __data(self, stock_id, ndays):
        self.__set_look_back(stock_id)

        # Taking 收盤價 開盤價 高低均價 成交量 as a predictor
        dataset = query_close_price(self.history, stock_id)
        dataset_open = query_open_price(self.history, int(stock_id))
        dataset_high = query_high_price(self.history, int(stock_id))
        dataset_low = query_low_price(self.history, int(stock_id))
        dataset_avg = query_avg_price(self.history, int(stock_id))
        dataset_vol = query_volume(self.history, int(stock_id))
        # Transforming time series dataset into supervised dataset
        supervised = series_to_supervised(dataset, n_in=self.look_back, n_out=self.look_forward)
        supervised_open = series_to_supervised(dataset_open, n_in=self.look_back, n_out=self.look_forward)
        supervised_high = series_to_supervised(dataset_high, n_in=self.look_back, n_out=self.look_forward)
        supervised_low = series_to_supervised(dataset_low, n_in=self.look_back, n_out=self.look_forward)
        supervised_avg = series_to_supervised(dataset_avg, n_in=self.look_back, n_out=self.look_forward)
        supervised_vol = series_to_supervised(dataset_vol, n_in=self.look_back, n_out=self.look_forward)
        # Normalize dataset if needed
        ori_Xy = copy.deepcopy(supervised)
        Xy = normalize_windows(supervised)
        feature_open = normalize_windows(supervised_open)
        feature_high = normalize_windows(supervised_high)
        feature_low = normalize_windows(supervised_low)
        feature_avg = normalize_windows(supervised_avg)
        # feature_vol = normalize_windows(supervised_vol)
        # Feature Scaling for volume
        from sklearn.preprocessing import MinMaxScaler
        sc = MinMaxScaler(feature_range=(0, 1))
        feature_vol = sc.fit_transform(supervised_vol)
        # Converting array of list to numpy array
        ori_Xy = np.array(ori_Xy)
        Xy = np.array(Xy)
        # Spliting dataset into training and testing sets
        self.X_ori_train, self.X_ori_test, self.y_ori_train, self.y_ori_test = train_test_split(ori_Xy, test_samples=ndays, num_forecasts=self.look_forward)
        X_train, X_test, y_train, y_test = train_test_split(Xy, test_samples=ndays, num_forecasts=self.look_forward)
        # Adding more features
        feature_open_train, feature_open_test, _, _ = train_test_split(feature_open, test_samples=ndays, num_forecasts=5)
        feature_high_train, feature_high_test, _, _ = train_test_split(feature_high, test_samples=ndays, num_forecasts=5)
        feature_low_train, feature_low_test, _, _ = train_test_split(feature_low, test_samples=ndays, num_forecasts=5)
        feature_avg_train, feature_avg_test, _, _ = train_test_split(feature_avg, test_samples=ndays, num_forecasts=5)
        feature_vol_train, feature_vol_test, _, _ = train_test_split(feature_vol, test_samples=ndays, num_forecasts=5)
        X_train = np.append(X_train, feature_open_train, axis=2)
        X_train = np.append(X_train, feature_high_train, axis=2)
        X_train = np.append(X_train, feature_low_train, axis=2)
        X_train = np.append(X_train, feature_avg_train, axis=2)
        # X_train = np.append(X_train, feature_vol_train, axis=2)
        X_test = np.append(X_test, feature_open_test, axis=2)
        X_test = np.append(X_test, feature_high_test, axis=2)
        X_test = np.append(X_test, feature_low_test, axis=2)
        X_test = np.append(X_test, feature_avg_test, axis=2)
        # X_test = np.append(X_test, feature_vol_test, axis=2)

        return X_train, y_train, X_test, y_test

    def data_last_price(self, stock_id):
        # Taking 收盤價 as a predictor
        dataset = query_close_price(self.history, stock_id)
        return dataset[-1:]

    def data_for_prediction(self, stock_id):
        self.__set_look_back(stock_id)

        # Taking 收盤價 開盤價 高低均價 成交量 as a predictor
        dataset = query_close_price(self.history, stock_id)
        dataset_open = query_open_price(self.history, int(stock_id))
        dataset_high = query_high_price(self.history, int(stock_id))
        dataset_low = query_low_price(self.history, int(stock_id))
        dataset_avg = query_avg_price(self.history, int(stock_id))
        dataset_vol = query_volume(self.history, int(stock_id))
        # Transforming time series dataset into supervised dataset
        supervised = series_to_supervised(dataset, n_in=self.look_back, n_out=0)
        supervised_open = series_to_supervised(dataset_open, n_in=self.look_back, n_out=0)
        supervised_high = series_to_supervised(dataset_high, n_in=self.look_back, n_out=0)
        supervised_low = series_to_supervised(dataset_low, n_in=self.look_back, n_out=0)
        supervised_avg = series_to_supervised(dataset_avg, n_in=self.look_back, n_out=0)
        supervised_vol = series_to_supervised(dataset_vol, n_in=self.look_back, n_out=0)
        # Normalize dataset if needed
        ori_Xy = copy.deepcopy(supervised)
        Xy = normalize_windows(supervised)
        feature_open = normalize_windows(supervised_open)
        feature_high = normalize_windows(supervised_high)
        feature_low = normalize_windows(supervised_low)
        feature_avg = normalize_windows(supervised_avg)
        # feature_vol = normalize_windows(supervised_vol)
        # Feature Scaling for volume
        from sklearn.preprocessing import MinMaxScaler
        sc = MinMaxScaler(feature_range=(0, 1))
        feature_vol = sc.fit_transform(supervised_vol)
        # Converting array of list to numpy array
        ori_Xy = np.array(ori_Xy)
        Xy = np.array(Xy)
        # Spliting dataset into predicting sets
        self.X_ori_test = predict_split(ori_Xy)
        X_test = predict_split(Xy)
        # Adding more features
        feature_open_test = predict_split(feature_open)
        feature_high_test = predict_split(feature_high)
        feature_low_test = predict_split(feature_low)
        feature_avg_test = predict_split(feature_avg)
        feature_vol_test = predict_split(feature_vol)
        X_test = np.append(X_test, feature_open_test, axis=2)
        X_test = np.append(X_test, feature_high_test, axis=2)
        X_test = np.append(X_test, feature_low_test, axis=2)
        X_test = np.append(X_test, feature_avg_test, axis=2)
        # X_test = np.append(X_test, feature_vol_test, axis=2)

        return self.X_ori_test, X_test

    def data_last_ndays_for_test(self, stock_id, ndays):
        return self.__data(stock_id, ndays)

    def data(self, stock_id):
        return self.__data(stock_id, ndays=0)

    def ori_train_data(self):
        return self.X_ori_train, self.y_ori_train

    def ori_test_data(self):
        return self.X_ori_test, self.y_ori_test

    def inverse_transform_prediction(self, prediction):
        prediction = (prediction + 1) * self.X_ori_test[:, 0]
        return prediction


class DataForStatefulModel(DataLoader):
    # Tuned params and variables
    # number of sequence data
    look_back = 5
    look_forward = 5

    def __init__(self):
        super(DataForStatefulModel, self).__init__()

    def __data(self, stock_id, ndays):
        # Taking 收盤價 as a predictor
        dataset = query_close_price(self.history, stock_id)
        # Transforming time series dataset into supervised dataset
        supervised = series_to_supervised(dataset, n_in=self.look_back, n_out=self.look_forward)
        # Normalize dataset if needed
        ori_Xy = copy.deepcopy(supervised)
        Xy = normalize_windows(supervised)
        # Converting array of list to numpy array
        ori_Xy = np.array(ori_Xy)
        Xy = np.array(Xy)
        # Spliting dataset into training and testing sets
        self.X_ori_train, self.X_ori_test, self.y_ori_train, self.y_ori_test = train_test_split(ori_Xy, test_samples=ndays, num_forecasts=self.look_forward)
        X_train, X_test, y_train, y_test = train_test_split(Xy, test_samples=ndays, num_forecasts=self.look_forward)

        # reshape for stateful LSTM
        y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))
        y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1], 1))

        return X_train, y_train, X_test, y_test

    def data_last_ndays_for_test(self, stock_id, ndays):
        return self.__data(stock_id, ndays)
