#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 26 11:55:37 2018

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

# Extracting/Filtering the training dataset by stock_id
# Taking 收盤價 as a predictor
from util import query_close_price, query_open_price, query_avg_price, query_volume
from util import query_high_price, query_low_price
dataset = query_close_price(history, int(stock_id))

###########################################################
# Visualising the stock price
from util import plot_stock_price
plot_stock_price(dataset)

# Now perform exponential moving average smoothing
# So the data will have a smoother curve than the original ragged data
EMA = 0.0
gamma = 0.1
nparray = dataset.reshape(-1, 1322)[0]
for index, value in np.ndenumerate(nparray):
  EMA = gamma*value + (1-gamma)*EMA
  nparray[index] = EMA
  
###########################################################
# Visualising the stock price
from util import plot_stock_price
plot_stock_price(dataset)
