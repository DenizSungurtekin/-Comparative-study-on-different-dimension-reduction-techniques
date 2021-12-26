import numpy as np

def genRandomDataUniform(N,M,down,up):
    res = np.asarray((up-down)*np.random.rand(N,M) + down)
    if M == 1:
        res = res.reshape((N,))
    return res

def genRandLabels(N,down,up):
    return np.random.randint(down,up+1,N)


