# -*- coding: utf-8 -*-

# Importing the libraries
import copy
import numpy as np
import pandas as pd

import abc
from abc import abstractmethod
from sklearn.preprocessing import MinMaxScaler

import settings
from util import load_csv, load_weighted_csv
from util import series_to_supervised, normalize_windows, train_test_split, predict_split
from util import query_close_price, query_open_price, query_volume, query_high_price, query_low_price
from util import query_weighted_open_price, query_weighted_close_price, query_weighted_high_price, query_weighted_low_price, query_weighted_avg_price
from util import query_avg_price, moving_average


class DataLoader(abc.ABC):
    def __init__(self, stock_id=None):
        self.stock_id = stock_id
        # Importing the dataset
        filepath = '{}/tetfp.csv'.format(settings.DATASET_PATH)
        self.history = load_csv(filepath, stock_id)
        # if self.stock_id is not None:
        #     filepath = '{}/weighted_stock_price_index.csv'.format(settings.DATASET_PATH)
        #     self.weighted_history = load_weighted_csv(filepath, self.history)

    def _set_look_back(self, stock_id):
        self.look_back = 60
        if stock_id in [690, 692, 701, 713]:
            self.look_back = 30
        print('set look back to {} for stock {}'.format(self.look_back, stock_id))


class DataForCNNModel(DataLoader):
    # Tuned params and variables
    # number of sequence data
    look_back = 60
    look_forward = 5

    def __init__(self, stock_id):
        super(DataForCNNModel, self).__init__(stock_id)

    def __prepare_data(self, stock_id):
        self._set_look_back(stock_id)
        print('look_back {}, look_forward {}'.format(self.look_back, self.look_forward))
        # Taking 收盤價 開盤價 高低均價 成交量 as a predictor
        dataset_close = query_close_price(self.history, int(stock_id))
        dataset_vol = query_volume(self.history, int(stock_id))
        # Feature Scaling for volume
        self.sc = MinMaxScaler(feature_range=(0, 1))
        scaled_dataset_close = self.sc.fit_transform(dataset_close)
        sc = MinMaxScaler(feature_range=(0, 1))
        scaled_dataset_vol = sc.fit_transform(dataset_vol)
        # Transforming time series dataset into supervised dataset
        self.supervised_ori_close = series_to_supervised(dataset_close, n_in=self.look_back, n_out=self.look_forward)
        self.supervised_close = series_to_supervised(scaled_dataset_close, n_in=self.look_back, n_out=self.look_forward)
        self.supervised_vol = np.array(series_to_supervised(scaled_dataset_vol, n_in=self.look_back, n_out=self.look_forward))

    def __data(self, stock_id, ndays):
        self.__prepare_data(stock_id)

        # Converting array of list to numpy array
        ori_Xy = np.array(self.supervised_ori_close)
        Xy = np.array(self.supervised_close)
        # Spliting dataset into training and testing sets
        self.X_ori_train, self.X_ori_test, self.y_ori_train, self.y_ori_test = train_test_split(ori_Xy, test_samples=ndays, num_forecasts=self.look_forward)
        X_train, X_test, y_train, y_test = train_test_split(Xy, test_samples=ndays, num_forecasts=self.look_forward)

        # Adding more features
        # feature_vol_train, feature_vol_test, _, _ = train_test_split(self.supervised_vol, test_samples=ndays, num_forecasts=5)

        # X_train = np.append(X_train, feature_vol_train, axis=2)

        # X_test = np.append(X_test, feature_vol_test, axis=2)

        return X_train, y_train, X_test, y_test

    def data_last_price(self, stock_id):
        # Taking 收盤價 as a predictor
        dataset = query_close_price(self.history, stock_id)
        return dataset[-1:]

    def data_for_prediction(self, stock_id):
        # tweek lookforard to 0 for prediction
        self.look_forward = 0

        self.__prepare_data(stock_id)

        # Converting array of list to numpy array
        ori_Xy = np.array(self.supervised_ori_close)
        Xy = np.array(self.supervised_close)
        # Spliting dataset into predicting sets
        self.X_ori_test = predict_split(ori_Xy)
        X_test = predict_split(Xy)

        # Adding more features
        # feature_vol_test = predict_split(self.supervised_vol)

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
        prediction = self.sc.inverse_transform(prediction)
        return prediction


class DataForStatelessModelMoreFeatures(DataLoader):
    # Tuned params and variables
    # number of sequence data
    look_back = 50
    look_forward = 5

    def __init__(self, stock_id=None):
        super(DataForStatelessModelMoreFeatures, self).__init__(stock_id)

    def __prepare_data(self, stock_id):
        self._set_look_back(stock_id)
        print('look_back {}, look_forward {}'.format(self.look_back, self.look_forward))
        # Taking 收盤價 開盤價 高低均價 成交量 as a predictor
        dataset_close = query_close_price(self.history, int(stock_id))
        # dataset_close_ma = moving_average(dataset_close, 5) # last 20
        # dataset_open = query_open_price(self.history, int(stock_id))
        # dataset_high = query_high_price(self.history, int(stock_id))
        # dataset_low = query_low_price(self.history, int(stock_id))
        # dataset_avg = query_avg_price(self.history, int(stock_id))
        # dataset_vol = query_volume(self.history, int(stock_id))
        # dataset_weighted_close = query_weighted_close_price(self.history)
        # dataset_weighted_close_ma = moving_average(dataset_weighted_close, 5) # last 20
        # Transforming time series dataset into supervised dataset
        supervised_close = series_to_supervised(dataset_close, n_in=self.look_back, n_out=self.look_forward)
        # supervised_close_ma = series_to_supervised(dataset_close_ma, n_in=self.look_back, n_out=self.look_forward)
        # supervised_open = series_to_supervised(dataset_open, n_in=self.look_back, n_out=self.look_forward)
        # supervised_high = series_to_supervised(dataset_high, n_in=self.look_back, n_out=self.look_forward)
        # supervised_low = series_to_supervised(dataset_low, n_in=self.look_back, n_out=self.look_forward)
        # supervised_avg = series_to_supervised(dataset_avg, n_in=self.look_back, n_out=self.look_forward)
        # supervised_vol = series_to_supervised(dataset_vol, n_in=self.look_back, n_out=self.look_forward)
        # supervised_weighted_close_ma = series_to_supervised(dataset_weighted_close_ma, n_in=self.look_back, n_out=self.look_forward)

        # Normalize dataset if needed
        self.ori_feature_close = copy.deepcopy(supervised_close)
        self.feature_close = normalize_windows(supervised_close)
        # self.feature_close_ma = normalize_windows(supervised_close_ma)
        # self.feature_open = normalize_windows(supervised_open)
        # self.feature_high = normalize_windows(supervised_high)
        # self.feature_low = normalize_windows(supervised_low)
        # self.feature_avg = normalize_windows(supervised_avg)
        # self.feature_vol = normalize_windows(supervised_vol)/100
        # Feature Scaling for volume
        # from sklearn.preprocessing import MinMaxScaler
        # sc = MinMaxScaler(feature_range=(0, 1))
        # self.feature_vol = sc.fit_transform(supervised_vol)
        # self.feature_weighted_close_ma = normalize_windows(supervised_weighted_close_ma)

    def __data(self, stock_id, ndays):
        self.__prepare_data(stock_id)

        # Converting array of list to numpy array
        ori_Xy = np.array(self.ori_feature_close)
        Xy = np.array(self.feature_close)
        # Spliting dataset into training and testing sets
        self.X_ori_train, self.X_ori_test, self.y_ori_train, self.y_ori_test = train_test_split(ori_Xy, test_samples=ndays, num_forecasts=self.look_forward)
        X_train, X_test, y_train, y_test = train_test_split(Xy, test_samples=ndays, num_forecasts=self.look_forward)
        # Adding more features
        # feature_close_ma_train, feature_close_ma_test, _, _ = train_test_split(self.feature_close_ma, test_samples=ndays, num_forecasts=5)
        # feature_open_train, feature_open_test, _, _ = train_test_split(self.feature_open, test_samples=ndays, num_forecasts=5)
        # feature_high_train, feature_high_test, _, _ = train_test_split(self.feature_high, test_samples=ndays, num_forecasts=5)
        # feature_low_train, feature_low_test, _, _ = train_test_split(self.feature_low, test_samples=ndays, num_forecasts=5)
        # feature_avg_train, feature_avg_test, _, _ = train_test_split(self.feature_avg, test_samples=ndays, num_forecasts=5)
        # feature_vol_train, feature_vol_test, _, _ = train_test_split(self.feature_vol, test_samples=ndays, num_forecasts=5)
        # feature_weighted_close_ma_train, feature_weighted_close_ma_test, _, _ = train_test_split(self.feature_weighted_close_ma, test_samples=ndays, num_forecasts=5)

        # X_train = np.append(X_train, feature_close_ma_train, axis=2)
        # X_train = np.append(X_train, feature_open_train, axis=2)
        # X_train = np.append(X_train, feature_high_train, axis=2)
        # X_train = np.append(X_train, feature_low_train, axis=2)
        # X_train = np.append(X_train, feature_avg_train, axis=2)
        # X_train = np.append(X_train, feature_vol_train, axis=2)
        # X_train = np.append(X_train, feature_weighted_close_ma_train, axis=2)

        # X_test = np.append(X_test, feature_close_ma_test, axis=2)
        # X_test = np.append(X_test, feature_open_test, axis=2)
        # X_test = np.append(X_test, feature_high_test, axis=2)
        # X_test = np.append(X_test, feature_low_test, axis=2)
        # X_test = np.append(X_test, feature_avg_test, axis=2)
        # X_test = np.append(X_test, feature_vol_test, axis=2)
        # X_test = np.append(X_test, feature_weighted_close_ma_test, axis=2)

        return X_train, y_train, X_test, y_test

    def data_last_price(self, stock_id):
        # Taking 收盤價 as a predictor
        dataset = query_close_price(self.history, stock_id)
        return dataset[-1:]

    def data_for_prediction(self, stock_id):
        # tweek lookforard to 0 for prediction
        self.look_forward = 0

        self.__prepare_data(stock_id)

        # Converting array of list to numpy array
        ori_Xy = np.array(self.ori_feature_close)
        Xy = np.array(self.feature_close)
        # Spliting dataset into predicting sets
        self.X_ori_test = predict_split(ori_Xy)
        X_test = predict_split(Xy)
        # Adding more features
        # feature_close_ma_test = predict_split(self.feature_close_ma)
        # feature_open_test = predict_split(self.feature_open)
        # feature_high_test = predict_split(self.feature_high)
        # feature_low_test = predict_split(self.feature_low)
        # feature_avg_test = predict_split(self.feature_avg)
        # feature_vol_test = predict_split(self.feature_vol)
        # feature_weighted_close_ma_test = predict_split(self.feature_weighted_close_ma)

        # X_test = np.append(X_test, feature_close_ma_test, axis=2)
        # X_test = np.append(X_test, feature_open_test, axis=2)
        # X_test = np.append(X_test, feature_high_test, axis=2)
        # X_test = np.append(X_test, feature_low_test, axis=2)
        # X_test = np.append(X_test, feature_avg_test, axis=2)
        # X_test = np.append(X_test, feature_vol_test, axis=2)
        # X_test = np.append(X_test, feature_weighted_close_ma_test, axis=2)

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


class DataForStatelessModelDiff(DataForStatelessModelMoreFeatures):
    def __init__(self, stock_id):
        super(DataForStatelessModelDiff, self).__init__(stock_id)

    def _set_look_back(self, stock_id):
        # default look_back
        self.look_back = 50
        print('set look back to {} for stock {}'.format(self.look_back, stock_id))

    def __data(self, stock_id, ndays):
        self._set_look_back(stock_id)

        # Taking 收盤價 開盤價 高低均價 成交量 as a predictor
        dataset = query_close_price(self.history, stock_id)
        dataset_open = query_open_price(self.history, int(stock_id))
        dataset_high = query_high_price(self.history, int(stock_id))
        dataset_low = query_low_price(self.history, int(stock_id))
        dataset_avg = query_avg_price(self.history, int(stock_id))
        # Calculating diff
        dataset = pd.DataFrame(dataset).diff(self.look_back).dropna().values
        dataset_open = pd.DataFrame(dataset_open).diff(self.look_back).dropna().values
        dataset_high = pd.DataFrame(dataset_high).diff(self.look_back).dropna().values
        dataset_low = pd.DataFrame(dataset_low).diff(self.look_back).dropna().values
        dataset_avg = pd.DataFrame(dataset_avg).diff(self.look_back).dropna().values
        # Transforming time series dataset into supervised dataset
        supervised = series_to_supervised(dataset, n_in=self.look_back, n_out=self.look_forward)
        supervised_open = series_to_supervised(dataset_open, n_in=self.look_back, n_out=self.look_forward)
        supervised_high = series_to_supervised(dataset_high, n_in=self.look_back, n_out=self.look_forward)
        supervised_low = series_to_supervised(dataset_low, n_in=self.look_back, n_out=self.look_forward)
        supervised_avg = series_to_supervised(dataset_avg, n_in=self.look_back, n_out=self.look_forward)
        # Normalize dataset if needed
        ori_Xy = copy.deepcopy(supervised)
        Xy = np.array(supervised)
        feature_open = np.array(supervised_open)
        feature_high = np.array(supervised_high)
        feature_low = np.array(supervised_low)
        feature_avg = np.array(supervised_avg)
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
        X_train = np.append(X_train, feature_open_train, axis=2)
        X_train = np.append(X_train, feature_high_train, axis=2)
        X_train = np.append(X_train, feature_low_train, axis=2)
        X_train = np.append(X_train, feature_avg_train, axis=2)
        X_test = np.append(X_test, feature_open_test, axis=2)
        X_test = np.append(X_test, feature_high_test, axis=2)
        X_test = np.append(X_test, feature_low_test, axis=2)
        X_test = np.append(X_test, feature_avg_test, axis=2)

        return X_train, y_train, X_test, y_test

    def data_for_prediction(self, stock_id):
        self._set_look_back(stock_id)

        # Taking 收盤價 開盤價 高低均價 成交量 as a predictor
        dataset = query_close_price(self.history, stock_id)
        dataset_open = query_open_price(self.history, int(stock_id))
        dataset_high = query_high_price(self.history, int(stock_id))
        dataset_low = query_low_price(self.history, int(stock_id))
        dataset_avg = query_avg_price(self.history, int(stock_id))
        # Calculating diff
        dataset = pd.DataFrame(dataset).diff(self.look_back).dropna().values
        dataset_open = pd.DataFrame(dataset_open).diff(self.look_back).dropna().values
        dataset_high = pd.DataFrame(dataset_high).diff(self.look_back).dropna().values
        dataset_low = pd.DataFrame(dataset_low).diff(self.look_back).dropna().values
        dataset_avg = pd.DataFrame(dataset_avg).diff(self.look_back).dropna().values
        # Transforming time series dataset into supervised dataset
        supervised = series_to_supervised(dataset, n_in=self.look_back, n_out=0)
        supervised_open = series_to_supervised(dataset_open, n_in=self.look_back, n_out=0)
        supervised_high = series_to_supervised(dataset_high, n_in=self.look_back, n_out=0)
        supervised_low = series_to_supervised(dataset_low, n_in=self.look_back, n_out=0)
        supervised_avg = series_to_supervised(dataset_avg, n_in=self.look_back, n_out=0)
        # Normalize dataset if needed
        ori_Xy = copy.deepcopy(supervised)
        Xy = np.array(supervised)
        feature_open = np.array(supervised_open)
        feature_high = np.array(supervised_high)
        feature_low = np.array(supervised_low)
        feature_avg = np.array(supervised_avg)
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
        X_test = np.append(X_test, feature_open_test, axis=2)
        X_test = np.append(X_test, feature_high_test, axis=2)
        X_test = np.append(X_test, feature_low_test, axis=2)
        X_test = np.append(X_test, feature_avg_test, axis=2)

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
        # prediction = (prediction + 1) * self.X_ori_test[:, 0]
        return prediction

class DataForStatelessModelWeighted(DataForStatelessModelMoreFeatures):
    def __init__(self, stock_id):
        super(DataForStatelessModelWeighted, self).__init__(stock_id)

    def __data(self, stock_id, ndays):
        super(DataForStatelessModelWeighted, self)._set_look_back(stock_id)

        # Taking 收盤價 開盤價 高低均價 成交量 as a predictor
        dataset = query_close_price(self.history, stock_id)
        dataset_open = query_open_price(self.history, int(stock_id))
        dataset_high = query_high_price(self.history, int(stock_id))
        dataset_low = query_low_price(self.history, int(stock_id))
        dataset_avg = query_avg_price(self.history, int(stock_id))
        dataset_vol = query_volume(self.history, int(stock_id))
        dataset_weighted_close = query_weighted_close_price(self.weighted_history)
        dataset_weighted_open = query_weighted_open_price(self.weighted_history)
        dataset_weighted_high = query_weighted_high_price(self.weighted_history)
        dataset_weighted_low = query_weighted_low_price(self.weighted_history)
        dataset_weighted_avg = query_weighted_avg_price(self.weighted_history)

        # Transforming time series dataset into supervised dataset
        supervised = series_to_supervised(dataset, n_in=self.look_back, n_out=self.look_forward)
        supervised_open = series_to_supervised(dataset_open, n_in=self.look_back, n_out=self.look_forward)
        supervised_high = series_to_supervised(dataset_high, n_in=self.look_back, n_out=self.look_forward)
        supervised_low = series_to_supervised(dataset_low, n_in=self.look_back, n_out=self.look_forward)
        supervised_avg = series_to_supervised(dataset_avg, n_in=self.look_back, n_out=self.look_forward)
        supervised_vol = series_to_supervised(dataset_vol, n_in=self.look_back, n_out=self.look_forward)
        supervised_weighted_close = series_to_supervised(dataset_weighted_close, n_in=self.look_back, n_out=self.look_forward)
        supervised_weighted_open = series_to_supervised(dataset_weighted_open, n_in=self.look_back, n_out=self.look_forward)
        supervised_weighted_high = series_to_supervised(dataset_weighted_high, n_in=self.look_back, n_out=self.look_forward)
        supervised_weighted_low = series_to_supervised(dataset_weighted_low, n_in=self.look_back, n_out=self.look_forward)
        supervised_weighted_avg = series_to_supervised(dataset_weighted_avg, n_in=self.look_back, n_out=self.look_forward)
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
        feature_weighted_close = normalize_windows(supervised_weighted_close)
        feature_weighted_open = normalize_windows(supervised_weighted_open)
        feature_weighted_high = normalize_windows(supervised_weighted_high)
        feature_weighted_low = normalize_windows(supervised_weighted_low)
        feature_weighted_avg = normalize_windows(supervised_weighted_avg)

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
        feature_weighted_close_train, feature_weighted_close_test, _, _ = train_test_split(feature_weighted_close, test_samples=ndays, num_forecasts=5)
        feature_weighted_open_train, feature_weighted_open_test, _, _ = train_test_split(feature_weighted_open, test_samples=ndays, num_forecasts=5)
        feature_weighted_high_train, feature_weighted_high_test, _, _ = train_test_split(feature_weighted_high, test_samples=ndays, num_forecasts=5)
        feature_weighted_low_train, feature_weighted_low_test, _, _ = train_test_split(feature_weighted_low, test_samples=ndays, num_forecasts=5)
        feature_weighted_avg_train, feature_weighted_avg_test, _, _ = train_test_split(feature_weighted_avg, test_samples=ndays, num_forecasts=5)

        X_train = np.append(X_train, feature_open_train, axis=2)
        X_train = np.append(X_train, feature_high_train, axis=2)
        X_train = np.append(X_train, feature_low_train, axis=2)
        X_train = np.append(X_train, feature_avg_train, axis=2)
        X_train = np.append(X_train, feature_vol_train, axis=2)
        X_train = np.append(X_train, feature_weighted_close_train, axis=2)
        X_train = np.append(X_train, feature_weighted_open_train, axis=2)
        X_train = np.append(X_train, feature_weighted_high_train, axis=2)
        X_train = np.append(X_train, feature_weighted_low_train, axis=2)
        X_train = np.append(X_train, feature_weighted_avg_train, axis=2)
        X_test = np.append(X_test, feature_open_test, axis=2)
        X_test = np.append(X_test, feature_high_test, axis=2)
        X_test = np.append(X_test, feature_low_test, axis=2)
        X_test = np.append(X_test, feature_avg_test, axis=2)
        X_test = np.append(X_test, feature_vol_test, axis=2)
        X_test = np.append(X_test, feature_weighted_close_test, axis=2)
        X_test = np.append(X_test, feature_weighted_open_test, axis=2)
        X_test = np.append(X_test, feature_weighted_high_test, axis=2)
        X_test = np.append(X_test, feature_weighted_low_test, axis=2)
        X_test = np.append(X_test, feature_weighted_avg_test, axis=2)
        return X_train, y_train, X_test, y_test

    def data_for_prediction(self, stock_id):
        super(DataForStatelessModelWeighted, self)._set_look_back(stock_id)

        # Taking 收盤價 開盤價 高低均價 成交量 as a predictor
        dataset = query_close_price(self.history, stock_id)
        dataset_open = query_open_price(self.history, int(stock_id))
        dataset_high = query_high_price(self.history, int(stock_id))
        dataset_low = query_low_price(self.history, int(stock_id))
        dataset_avg = query_avg_price(self.history, int(stock_id))
        dataset_vol = query_volume(self.history, int(stock_id))
        dataset_weighted_close = query_weighted_close_price(self.weighted_history)
        dataset_weighted_open = query_weighted_open_price(self.weighted_history)
        dataset_weighted_high = query_weighted_high_price(self.weighted_history)
        dataset_weighted_low = query_weighted_low_price(self.weighted_history)
        dataset_weighted_avg = query_weighted_avg_price(self.weighted_history)

        # Transforming time series dataset into supervised dataset
        supervised = series_to_supervised(dataset, n_in=self.look_back, n_out=0)
        supervised_open = series_to_supervised(dataset_open, n_in=self.look_back, n_out=0)
        supervised_high = series_to_supervised(dataset_high, n_in=self.look_back, n_out=0)
        supervised_low = series_to_supervised(dataset_low, n_in=self.look_back, n_out=0)
        supervised_avg = series_to_supervised(dataset_avg, n_in=self.look_back, n_out=0)
        supervised_vol = series_to_supervised(dataset_vol, n_in=self.look_back, n_out=0)
        supervised_weighted_close = series_to_supervised(dataset_weighted_close, n_in=self.look_back, n_out=self.look_forward)
        supervised_weighted_open = series_to_supervised(dataset_weighted_open, n_in=self.look_back, n_out=self.look_forward)
        supervised_weighted_high = series_to_supervised(dataset_weighted_high, n_in=self.look_back, n_out=self.look_forward)
        supervised_weighted_low = series_to_supervised(dataset_weighted_low, n_in=self.look_back, n_out=self.look_forward)
        supervised_weighted_avg = series_to_supervised(dataset_weighted_avg, n_in=self.look_back, n_out=self.look_forward)
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
        feature_weighted_close = normalize_windows(supervised_weighted_close)
        feature_weighted_open = normalize_windows(supervised_weighted_open)
        feature_weighted_high = normalize_windows(supervised_weighted_high)
        feature_weighted_low = normalize_windows(supervised_weighted_low)
        feature_weighted_avg = normalize_windows(supervised_weighted_avg)

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
        feature_weighted_close_test = predict_split(feature_weighted_close)
        feature_weighted_open_test = predict_split(feature_weighted_open)
        feature_weighted_high_test = predict_split(feature_weighted_high)
        feature_weighted_low_test = predict_split(feature_weighted_low)
        feature_weighted_avg_test = predict_split(feature_weighted_avg)

        X_test = np.append(X_test, feature_open_test, axis=2)
        X_test = np.append(X_test, feature_high_test, axis=2)
        X_test = np.append(X_test, feature_low_test, axis=2)
        X_test = np.append(X_test, feature_avg_test, axis=2)
        X_test = np.append(X_test, feature_vol_test, axis=2)
        X_test = np.append(X_test, feature_weighted_close_test, axis=2)
        X_test = np.append(X_test, feature_weighted_open_test, axis=2)
        X_test = np.append(X_test, feature_weighted_high_test, axis=2)
        X_test = np.append(X_test, feature_weighted_low_test, axis=2)
        X_test = np.append(X_test, feature_weighted_avg_test, axis=2)

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
