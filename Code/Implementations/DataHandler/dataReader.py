from numpy import genfromtxt
import numpy as np
from keras.datasets import mnist

def csv(filename):
    return genfromtxt(filename, delimiter=',',dtype=None,encoding=None)

def loadMnist(size):
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    crop = size
    data = train_X[:crop]
    data = np.reshape(data, (crop, train_X.shape[1] * train_X.shape[2]))
    return data
