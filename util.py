# -*- coding: utf-8 -*-

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Spliting dataset into training and testing sets
from sklearn.model_selection import train_test_split as sklearn_train_test_split

from keras.callbacks import Callback


class SavePredictionCallback(Callback):
    counter = 0

    def __init__(self, predicted_prefix, X_train):
        self.predicted_prefix = predicted_prefix
        self.X_train = X_train
 
    def on_epoch_end(self, epoch, logs={}):
        # make prediction every 10 epoch and save it
        # self.counter = self.counter + 1
        # if self.counter % 10 == 0:
        #     predicted_price = self.model.predict(self.X_train, batch_size=1)        
        #     predicted_price = np.concatenate((predicted_price[0], np.array(predicted_price)[1:, -1]))
        #     np.save('{}_{}.npy'.format(self.predicted_prefix, self.counter), predicted_price)
        self.model.reset_states()
        return
 

def load_csv(filepath):
    """
    Load a dataset from csv file.
    Arguments:
        filepath: Relative or absolute file path to the historical stock price in csv format.
    Returns:
        Full historical stock prices as a Dataframe
    """
    print('historical data is loading from {}'.format(filepath))
    try:
        return pd.read_csv(filepath, encoding='big5-hkscs', thousands=',')
    except:
        return pd.read_csv(filepath, encoding='utf8', thousands=',')


def get_model_name(stock_id):
    return 'etf_{}_model.h5'.format(stock_id)


def query_open_price(dataset, stock_id):
    """
    Query open stock price by stock id.
    Arguments:
        dataset: Full historical stock prices as a Dataframe
        stock_id: A stock id
    Returns:
        Sequence of stock price as a NumPy array.
    """
    assert type(dataset) is pd.DataFrame, 'unexpected type of series: {}'.format(type(dataset))
    # Extracting/Filtering the training dataset by stock_id    
    column = dataset.columns[0]
    dataset = dataset.loc[dataset[column] == stock_id]
    assert dataset.size > 0, 'dataset is empty while quering stock id {}'.format(stock_id)
    # Returning 開盤價
    return dataset.iloc[:, 3:4].values


def query_close_price(dataset, stock_id):
    """
    Query close stock price by stock id.
    Arguments:
        dataset: Full historical stock prices as a Dataframe
        stock_id: A stock id
    Returns:
        Sequence of stock price as a NumPy array.
    """
    assert type(dataset) is pd.DataFrame, 'unexpected type of series: {}'.format(type(dataset))
    # Extracting/Filtering the training dataset by stock_id    
    column = dataset.columns[0]
    dataset = dataset.loc[dataset[column] == stock_id]
    assert dataset.size > 0, 'dataset is empty while quering stock id {}'.format(stock_id)
    # Returning 收盤價
    return dataset.iloc[:, 6:7].values


def query_high_price(dataset, stock_id):
    """
    Query high stock price by stock id.
    Arguments:
        dataset: Full historical stock prices as a Dataframe
        stock_id: A stock id
    Returns:
        Sequence of stock price as a NumPy array.
    """
    assert type(dataset) is pd.DataFrame, 'unexpected type of series: {}'.format(type(dataset))
    # Extracting/Filtering the training dataset by stock_id    
    column = dataset.columns[0]
    dataset = dataset.loc[dataset[column] == stock_id]
    assert dataset.size > 0, 'dataset is empty while quering stock id {}'.format(stock_id)
    # Returning 高價
    return dataset.iloc[:, 4:5].values


def query_low_price(dataset, stock_id):
    """
    Query low stock price by stock id.
    Arguments:
        dataset: Full historical stock prices as a Dataframe
        stock_id: A stock id
    Returns:
        Sequence of stock price as a NumPy array.
    """
    assert type(dataset) is pd.DataFrame, 'unexpected type of series: {}'.format(type(dataset))
    # Extracting/Filtering the training dataset by stock_id    
    column = dataset.columns[0]
    dataset = dataset.loc[dataset[column] == stock_id]
    assert dataset.size > 0, 'dataset is empty while quering stock id {}'.format(stock_id)
    # Returning 低價
    return dataset.iloc[:, 5:6].values


def query_avg_price(dataset, stock_id):
    """
    Query high/low stock price by stock id.
    Arguments:
        dataset: Full historical stock prices as a Dataframe
        stock_id: A stock id
    Returns:
        Sequence of stock price as a NumPy array.
    """
    assert type(dataset) is pd.DataFrame, 'unexpected type of series: {}'.format(type(dataset))
    # Extracting/Filtering the training dataset by stock_id    
    column = dataset.columns[0]
    dataset = dataset.loc[dataset[column] == stock_id]
    assert dataset.size > 0, 'dataset is empty while quering stock id {}'.format(stock_id)
    # Returning 高低價平均
    return dataset.iloc[:, 4:6].mean(axis=1).values.reshape(-1,1)


def query_volume(dataset, stock_id):
    """
    Query volume by stock id.
    Arguments:
        dataset: Full historical volume as a Dataframe
        stock_id: A stock id
    Returns:
        Sequence of volume as a NumPy array.
    """
    assert type(dataset) is pd.DataFrame, 'unexpected type of series: {}'.format(type(dataset))
    # Extracting/Filtering the training dataset by stock_id    
    column = dataset.columns[0]
    dataset = dataset.loc[dataset[column] == stock_id]
    assert dataset.size > 0, 'dataset is empty while quering stock id {}'.format(stock_id)
    # Returning 成交量
    return dataset.iloc[:, 7:8].values


def plot_stock_price(series, first_ndays=0, last_ndays=0, filename=None):
    """
    Plot stock price.
    Arguments:
        series: Sequence of observations as a NumPy array.
        first_ndays, last_ndays:
            If both are 0, plot whole series, otherwise plot first N days or plot last N days instead.
    Returns:
        N/A
    """
    assert type(series) is np.ndarray, 'unexpected type of series: {}'.format(type(series))
    if first_ndays == 0 and last_ndays == 0:
        plt.plot(series, color='blue', label='Stock Price')
    elif first_ndays > 0:
        plt.plot(series[:first_ndays, :], color='blue', label='Stock Price')
    else:
        plt.plot(series[-last_ndays:, :], color='blue', label='Stock Price')
    plt.title('Stock Price')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    if filename is not None:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


def plot_real_predicted_stock_price(real_price, predicted_price, title, first_ndays=0, last_ndays=0, filename=None):
    """
    Plot stock price.
    Arguments:
        real_price: Sequence of real price as a NumPy array.
        predicted_price: Sequence of predicted price as a NumPy array.
        first_ndays, last_ndays:
            If both are 0, plot whole series, otherwise plot first N days or plot last N days instead.
    Returns:
        N/A
    """
    assert type(real_price) is np.ndarray, 'unexpected type of real_price: {}'.format(type(real_price))
    assert type(predicted_price) is np.ndarray, 'unexpected type of predicted_price: {}'.format(type(predicted_price))
    assert(real_price.shape[0] == predicted_price.shape[0])
    if first_ndays == 0 and last_ndays == 0:
        plt.plot(real_price, color='red', label='Real Price')
        plt.plot(predicted_price, color = 'blue', label = 'Predicted Price')
    elif first_ndays > 0:
        plt.plot(real_price[:first_ndays, :], color='red', label='Real Price')
        plt.plot(predicted_price[:first_ndays, :], color = 'blue', label = 'Predicted Price')
    else:
        plt.plot(real_price[-last_ndays:, :], color='red', label='Real Price')    
        plt.plot(predicted_price[-last_ndays:, :], color = 'blue', label = 'Predicted Price')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    if filename is not None:
        plt.savefig(filename)
        plt.close()        
    else:
        plt.show()


def series_to_supervised(series, n_in, n_out):
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


def normalize_windows(series):
    """
    Normalize dataset to improve the convergence
    Normalize each value to reflect the percentage changes from starting point.
    Arguments:
        series: Sequence of observations as an array of NumPy array.
    Returns:
        NumPy array of normalized series for supervised learning.
    """
    assert type(series) is list, 'unexpected type of series: {}'.format(type(series))
    assert type(series[0]) is np.ndarray, 'unexpected type of series: {}'.format(type(series[0]))
    df = pd.DataFrame(series)
    df = df.div(df[0], axis=0) - 1
    return df.values


def predict_split(Xy):
    """
    Split supervised learning dataset into predicting sets
    Arguments:
        Xy: Two dimensions of sequence observations as a NumPy array.
    """
    assert type(Xy) is np.ndarray, 'unexpected type of Xy: {}'.format(type(Xy))
    assert(Xy.shape[0] > 1)

    # Historical price for prediction
    X = Xy[-1:, :]

    # Reshape the inputs from 1 dimenstion to 3 dimension
    # X.shape[0]: batch_size which is number of observations
    # X.shape[1]: timesteps which is look_back
    # 1: input_dim which is number of predictors
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X
    
    
def train_test_split(Xy, num_forecasts, test_samples=0):
    """
    Split supervised learning dataset into training and testing sets
    Arguments:
        Xy: Two dimensions of sequence observations as a NumPy array.
        test_samples:
            If test_samples > 0, reserve the last test_samples days (axis 0) of 
                sequence observations as test set.
            If test_samples = 0, random reserve 20% of sequence observations as test set.
        num_forecasts:
            Number of timesteps to be predicted base on remaining observations.
    """
    assert type(Xy) is np.ndarray, 'unexpected type of Xy: {}'.format(type(Xy))
    assert Xy.shape[0] > test_samples, 'Xy.shape[0] is {} and test_samples is {}'.format(Xy.shape[0], test_samples)
    assert(Xy.shape[1] > num_forecasts)

    # Historical price for regression
    X = Xy[:, :-num_forecasts]
    # Target price for regression
    y = Xy[:, -num_forecasts:]
        
    # Spliting dataset into training and testing sets    
    if test_samples > 0:
        # Select the last ndays working date for testing and the others for training.
        X_train = X[:-test_samples, :]
        y_train = y[:-test_samples, :]
        X_test = X[-test_samples:, :]
        y_test = y[-test_samples:, :]
    else:
        # Select 20% of the data for testing and 80% for training.
        # Shuffle the data in order to train in random order.
        X_train, X_test, y_train, y_test = sklearn_train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0) 

    # Reshape the inputs from 1 dimenstion to 3 dimension
    # X_train.shape[0]: batch_size which is number of observations
    # X_train.shape[1]: timesteps which is look_back
    # 1: input_dim which is number of predictors
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return X_train, X_test, y_train, y_test


def visualize_model(loader, model, stock_id, ndays, plot_prefix):
    X_train, y_train, X_test, y_test = loader.data_last_ndays_for_test(int(stock_id), ndays=ndays)
    X_ori_train, y_ori_train = loader.ori_train_data()
    X_ori_test, y_ori_test = loader.ori_test_data()

    # Normalized prediction
    real_price = y_test
    predicted_price = model.predict(X_test)
    predicted_price1 = predicted_price
    predicted_price2 = predicted_price

    if ndays > 1:
        real_price = np.concatenate((real_price[0], np.array(real_price)[1:, -1]))
        predicted_price1 = np.concatenate((predicted_price1[0], np.array(predicted_price1)[1:, -1]))
    else:
        real_price = real_price.transpose()
        predicted_price1 = predicted_price1.transpose()
    
    filename = '{}_normalized.png'.format(plot_prefix)
    plot_real_predicted_stock_price(
        real_price, 
        predicted_price1, 
        'Normalized Stock Price Prediction', 
        filename=filename)

    # Inversed transform prediction
    real_price2 = y_ori_test
    predicted_price2 = loader.inverse_transform_prediction(predicted_price)

    if ndays > 1:
        real_price2 = np.concatenate((real_price2[0], np.array(real_price2)[1:, -1]))
        predicted_price2 = np.concatenate((predicted_price2[0], np.array(predicted_price2)[1:, -1]))
    else:
        real_price2 = real_price2.transpose()
        predicted_price2 = predicted_price2.transpose()

    filename = '{}.png'.format(plot_prefix)
    plot_real_predicted_stock_price(
        real_price2, 
        predicted_price2, 
        'Stock Price Prediction', 
        filename=filename)
