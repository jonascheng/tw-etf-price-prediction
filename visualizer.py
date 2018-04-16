# -*- coding: utf-8 -*-

# Importing the libraries
import matplotlib.pyplot as plt

from keras.models import load_model


class Visualizer():
    def __init__(self):
        pass
    
    @staticmethod
    def show(real_price, predicted_price, title):
        plt.plot(real_price, color = 'red', label = 'Real Price')
        plt.plot(predicted_price, color = 'blue', label = 'Predicted Price')
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.show()
