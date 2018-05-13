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
stock_id = '0050'

###########################################################
import settings
from util import load_csv, load_weighted_csv
filepath = '{}/tetfp.csv'.format(settings.DATASET_PATH)
history = load_csv(filepath)
column = history.columns[0]
history = history.loc[history[column] == int(stock_id)]

filepath = '{}/weighted_stock_price_index.csv'.format(settings.DATASET_PATH)
weighted_history = load_weighted_csv(filepath)

