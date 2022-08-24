import numpy as np
from sklearn.datasets import make_swiss_roll
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt

def genRandomDataUniform(N,M,down,up): # Random uniform data in specific range
    res = np.asarray((up-down)*np.random.rand(N,M) + down)
    if M == 1:
        res = res.reshape((N,))
    return res

def genRandLabels(N,down,up): # Label between range
    return np.random.randint(down,up+1,N)

def generate_swiss_roll(n_samples, random_state = 1):
    X, color = make_swiss_roll(n_samples=n_samples, noise=0.00, random_state=random_state)
    return X,color

def makeCluster(plot=False,size=1000,features = 30):
    # centers = [(5,5),(-5,5),(5,-5),(-5,-5)] # To show working plot in 2d
    # cluster_std = [1,1,1,1]
    X, y = make_blobs(n_features = features,n_samples=size, random_state=1)
    if plot: # Case in 2d to plot
        plt.scatter(X[y == 0, 0], X[y == 0, 1], color="red", s=10, label="Cluster1")
        plt.scatter(X[y == 1, 0], X[y == 1, 1], color="blue", s=10, label="Cluster2")
        plt.scatter(X[y == 2, 0], X[y == 2, 1], color="green", s=10, label="Cluster3")
        plt.scatter(X[y == 3, 0], X[y == 3, 1], color="orange", s=10, label="Cluster4")
        plt.show()
    return X
