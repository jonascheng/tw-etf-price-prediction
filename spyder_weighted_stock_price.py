#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 12:40:30 2018

@author: jonas
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#stock_id = '00713'
#stock_id = '0050'
stock_id = '0052'

###########################################################
import settings
from util import load_csv, load_weighted_csv
filepath = '{}/tetfp.csv'.format(settings.DATASET_PATH)
history = load_csv(filepath, int(stock_id))

filepath = '{}/weighted_stock_price_index.csv'.format(settings.DATASET_PATH)
weighted_history = load_weighted_csv(filepath, history)

# Extracting/Filtering the training dataset by stock_id
# Taking 收盤價 as a predictor
from util import query_close_price, query_open_price, query_avg_price, query_volume
from util import query_high_price, query_low_price
dataset = query_close_price(history, int(stock_id))

from util import query_weighted_close_price, query_weighted_open_price
from util import query_weighted_avg_price, query_weighted_high_price, query_weighted_low_price
dataset_weighted_open = query_weighted_open_price(weighted_history)
dataset_weighted_close = query_weighted_close_price(weighted_history)
dataset_weighted_high = query_weighted_high_price(weighted_history)
dataset_weighted_low = query_weighted_low_price(weighted_history)
dataset_weighted_avg = query_weighted_avg_price(weighted_history)

###########################################################
# Visualising the stock price
from util import plot_stock_price
plot_stock_price(dataset)
plot_stock_price(dataset_weighted_open)
plot_stock_price(dataset_weighted_close)
plot_stock_price(dataset_weighted_high)
plot_stock_price(dataset_weighted_low)
plot_stock_price(dataset_weighted_avg)

###########################################################
# series_to_supervised
from util import series_to_supervised
supervised = series_to_supervised(dataset, n_in=50, n_out=5)
supervised_weighted_close = series_to_supervised(dataset_weighted_close, n_in=50, n_out=5)

###########################################################
# Visualising the stock price
from util import plot_stock_price
plot_stock_price(dataset, first_ndays=55)
plot_stock_price(supervised[0].transpose())
plot_stock_price(dataset_weighted_close, first_ndays=55)
plot_stock_price(supervised_weighted_close[0].transpose())

###########################################################
# normalize_windows
import copy
from util import normalize_windows
ori_Xy = copy.deepcopy(supervised)
Xy = normalize_windows(supervised)
F1 = normalize_windows(supervised_weighted_close)

###########################################################
# Visualising the stock price
from util import plot_stock_price
plot_stock_price(Xy[0].transpose())

real_price = np.concatenate((dataset[0], np.array(dataset)[1:, -1]))
real_price = np.expand_dims(real_price, axis=1)
normalized_price = np.concatenate((Xy[0], np.array(Xy)[1:, -1]))
normalized_price = np.expand_dims(normalized_price, axis=1)
plot_stock_price(real_price, last_ndays=240)
plot_stock_price(normalized_price, last_ndays=240)

real_price = np.concatenate((dataset_weighted_close[0], np.array(dataset_weighted_close)[1:, -1]))
real_price = np.expand_dims(real_price, axis=1)
normalized_price = np.concatenate((F1[0], np.array(F1)[1:, -1]))
normalized_price = np.expand_dims(normalized_price, axis=1)
plot_stock_price(real_price, last_ndays=240)
plot_stock_price(normalized_price, last_ndays=240)
