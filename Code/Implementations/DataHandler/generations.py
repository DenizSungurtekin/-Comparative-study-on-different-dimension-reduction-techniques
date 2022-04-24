import numpy as np
from sklearn.datasets import make_swiss_roll

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
