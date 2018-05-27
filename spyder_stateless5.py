#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 21:43:09 2018

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
dataset_open = query_open_price(history, int(stock_id))
dataset_close_open_diff = dataset_open - dataset

###########################################################
# Visualising the stock price
last_ndays = 240
from util import plot_stock_price
plot_stock_price(dataset, last_ndays=last_ndays)
plot_stock_price(dataset_open, last_ndays=last_ndays)
plot_stock_price(dataset_close_open_diff, last_ndays=last_ndays)

###########################################################
# series_to_supervised
from util import series_to_supervised
supervised = series_to_supervised(dataset, n_in=50, n_out=5)
supervised_open = series_to_supervised(dataset_open, n_in=50, n_out=5)
supervised_close_open_diff = series_to_supervised(dataset_close_open_diff, n_in=50, n_out=5)

###########################################################
# Visualising the stock price
from util import plot_stock_price
plot_stock_price(dataset, first_ndays=55)
plot_stock_price(supervised[0].transpose())
plot_stock_price(dataset_open, first_ndays=55)
plot_stock_price(supervised_open[0].transpose())
plot_stock_price(dataset_close_open_diff, first_ndays=55)
plot_stock_price(supervised_close_open_diff[0].transpose())

###########################################################
# normalize_windows
import copy
from util import normalize_windows
ori_Xy = copy.deepcopy(supervised)
Xy = normalize_windows(supervised)
F1 = normalize_windows(supervised_open)
F2 = normalize_windows(supervised_close_open_diff)

###########################################################
# Visualising the stock price
from util import plot_stock_price

index = 60
#plot_stock_price(ori_Xy[index])
plot_stock_price(Xy[index])
plot_stock_price(F1[index])
plot_stock_price(F2[index])
