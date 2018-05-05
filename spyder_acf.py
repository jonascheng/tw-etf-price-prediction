#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 22:39:49 2018

@author: jonas
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

stock_id = '0050'

###########################################################
import settings
from util import load_csv
filepath = '{}/tetfp.csv'.format(settings.DATASET_PATH)
history = load_csv(filepath)

# Extracting/Filtering the training dataset by stock_id
# Taking 收盤價 as a predictor
from util import query_close_price, query_open_price, query_avg_price, query_volume
from util import query_high_price, query_low_price
dataset = query_close_price(history, int(stock_id))
dataset_vol = query_volume(history, int(stock_id))

###########################################################
# Calculating percentage change
df = pd.DataFrame(dataset)
df = df.pct_change().fillna(0)
dataset_close = df.values

###########################################################
# Visualising the stock price
last_ndays = 365
from util import plot_stock_price
plot_stock_price(dataset, first_ndays=last_ndays)
plot_stock_price(dataset, last_ndays=last_ndays)
plot_stock_price(dataset_vol)
plot_stock_price(dataset_close)

###########################################################
# Visualising autocorrelation function
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(dataset, lags=100)
pyplot.show()

plot_acf(dataset_vol, lags=100)
pyplot.show()

plot_acf(dataset_close, lags=100)
pyplot.show()

###########################################################
# Visualising partial autocorrelation function
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(dataset, lags=110)
pyplot.show()