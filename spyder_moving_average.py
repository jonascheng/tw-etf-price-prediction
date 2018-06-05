#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 21:32:49 2018

@author: jonas
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

stock_id = '00701'

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
from util import query_weighted_close_price
dataset = query_close_price(history, int(stock_id))
dataset_weighted = query_weighted_close_price(weighted_history)

window=5
dataset_ma = pd.DataFrame(dataset).rolling(window=window).mean().values
dataset_weighted_ma = pd.DataFrame(dataset_weighted).rolling(window=window).mean().values

sum = 0
for i in range(window):
    sum += dataset[i]
    dataset_ma[i] = sum/(i+1)

sum = 0
for i in range(window):
    sum += dataset_weighted[i]
    dataset_weighted_ma[i] = sum/(i+1)
    
###########################################################
# Visualising the stock price
last_ndays = 120
from util import plot_stock_price
plot_stock_price(dataset, last_ndays=last_ndays)
plot_stock_price(dataset_ma, last_ndays=last_ndays)
plot_stock_price(dataset_weighted_ma, last_ndays=last_ndays)

first_ndays = 120
from util import plot_stock_price
plot_stock_price(dataset, first_ndays=first_ndays)
plot_stock_price(dataset_ma, first_ndays=first_ndays)
plot_stock_price(dataset_weighted_ma, first_ndays=first_ndays)

###########################################################
# series_to_supervised
from util import series_to_supervised
supervised = series_to_supervised(dataset, n_in=60, n_out=5)
supervised_ma = series_to_supervised(dataset_ma, n_in=60, n_out=5)
supervised_weighted_ma = series_to_supervised(dataset_weighted_ma, n_in=60, n_out=5)

###########################################################
# normalize_windows
import copy
from util import normalize_windows
ori_Xy = copy.deepcopy(supervised)
Xy = normalize_windows(supervised)
F1 = normalize_windows(supervised_ma)
F2 = normalize_windows(supervised_weighted_ma)

###########################################################
# Visualising the stock price
from util import plot_stock_price

index = 60
plot_stock_price(ori_Xy[index])
plot_stock_price(Xy[index])
plot_stock_price(F1[index])
plot_stock_price(F2[index])