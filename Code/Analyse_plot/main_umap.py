from Code.Implementations.Umap import umap
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

## UMAP
N = 10
M = 15

MIN_DIST = 0.0001# Hyperparametre influant sur a et b mais ici initialise à 1 # In practice, UMAP finds a and b from non-linear least-square fitting to the piecewise function with the min_dist hyperparameter:

TARGET_DIM = 2 #hyperparam
TARGET_K = 10 # Number of nearest neighboor <=> k = Somme des pij sur une ligne cf p.15 doc officiel => remplace perplexité chez tsne (2^log(n) = n)

#n_epoch -> hyperparam
#n -> hyperparam , the number of neighbors to consider when approximating the local metric; utilise dans spectral embedding (n_spectral)


modes = ["random","gaussian","pca","laplace"]
data = np.random.rand(N,M)
Y,loss = umap.umap_reduction(data,MIN_DIST,TARGET_DIM,TARGET_K,initialization="pca",epochs=200)
print(Y)