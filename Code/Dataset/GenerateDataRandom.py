import numpy as np

def genRandomDataUniform(N,M,down,up):
    return (up-down)*np.random.rand(N,M) + down

