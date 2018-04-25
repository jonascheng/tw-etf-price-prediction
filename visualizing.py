# -*- coding: utf-8 -*-

# Importing the libraries
import os
import numpy as np
import matplotlib.pyplot as plt

state = 'stateful'
real_price = np.load('{}_real_price.npy'.format(state))

nb_epoch = 10
batch_size = 1
layers = 1
output_dim = 50
optimizer = 'adam'
dropout = 0

predicted_prefix = '{}_predicted_epoch{}_batch{}_layers{}_output{}_dropout{}_{}'.format(
        state, nb_epoch, batch_size, layers, output_dim, dropout, optimizer)

for i in range(0, 100, 10):
    filename = '{}_{}.npy'.format(predicted_prefix, i)
    if os.path.isfile(filename):
        predicted_price = np.load(filename)
        plt.plot(real_price, color = 'red', label = 'Real Price')
        plt.plot(predicted_price, color = 'blue', label = 'Predicted Price')
        plt.title('Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.show()
    else:
        break