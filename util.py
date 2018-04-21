# -*- coding: utf-8 -*-

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Spliting dataset into training and testing sets
from sklearn.model_selection import train_test_split as sklearn_train_test_split


def load_csv(filepath):
    """
    Load a dataset from csv file.
    Arguments:
        filepath: Relative or absolute file path to the historical stock price in csv format.
    Returns:
        Full historical stock prices as a Dataframe
    """
    print('historical data is loading from {}'.format(filepath))
    return pd.read_csv(filepath, encoding='big5-hkscs')


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
    dataset = dataset.loc[dataset['代碼'] == stock_id]
    # Returning 收盤價
    return dataset.iloc[:, 6:7].values


def plot_stock_price(series, first_ndays=0, last_ndays=0):
    """
    Plot stock price.
    Arguments:
        series: Sequence of observations as a NumPy array.
        first_ndays, last_ndays:
            If both are 0, plot whole series, otherwise plot first N days or plot last N days instead.
    Returns:
        N/A
    """
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
    plt.show()


def series_to_supervised(series, n_in, n_out=1):
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


def train_test_split(Xy, num_forecasts, test_samples=0):
    """
    Split supervised learning dataset into train and test sets
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
    assert(Xy.shape[0] > test_samples)
    assert(Xy.shape[1] > num_forecasts)
    # Spliting dataset into training and testing sets    
    if test_samples > 0:
        # Historical price for regression
        X = Xy[:, :-num_forecasts]
        # Target price for regression
        y = Xy[:, -num_forecasts:]
        # Select the last ndays working date for testing and the others for training.
        X_train = X[:-test_samples, :]
        y_train = y[:-test_samples, :]
        X_test = X[-test_samples:, :]
        y_test = y[-test_samples:, :]
    else:
        X = Xy[:, :-num_forecasts]
        y = Xy[:, -num_forecasts:]
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