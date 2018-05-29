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

stock_id = '006201'

###########################################################
import settings
from util import load_csv
filepath = '{}/tetfp.csv'.format(settings.DATASET_PATH)
history = load_csv(filepath)

# Extracting/Filtering the training dataset by stock_id
# Taking 收盤價 as a predictor
from util import query_close_price, query_open_price, query_avg_price, query_volume
dataset = query_close_price(history, int(stock_id))

window=5
dataset_ma = pd.DataFrame(dataset).rolling(window=window).mean().values

sum = 0
for i in range(window):
    sum += dataset[i]
    dataset_ma[i] = sum/(i+1)
    
###########################################################
# Visualising the stock price
last_ndays = 120
from util import plot_stock_price
plot_stock_price(dataset, last_ndays=last_ndays)
plot_stock_price(dataset_ma, last_ndays=last_ndays)

first_ndays = 120
from util import plot_stock_price
plot_stock_price(dataset, first_ndays=first_ndays)
plot_stock_price(dataset_ma, first_ndays=first_ndays)

###########################################################
# series_to_supervised
from util import series_to_supervised
supervised = series_to_supervised(dataset, n_in=50, n_out=5)
supervised_ma = series_to_supervised(dataset_ma, n_in=50, n_out=5)

###########################################################
# normalize_windows
import copy
from util import normalize_windows
ori_Xy = copy.deepcopy(supervised)
Xy = normalize_windows(supervised)
F1 = normalize_windows(supervised_ma)

###########################################################
# Visualising the stock price
from util import plot_stock_price

index = 60
plot_stock_price(ori_Xy[index])
plot_stock_price(Xy[index])
plot_stock_price(F1[index])
