import torch as t
import torch.nn.functional as F
# print(F.kl_div(a,b))

N = 100
M = 25
data = t.rand((N,M))
sigmas = t.rand((N,1)) #Randomly generated


def computeProba(center,neighbour,sigma):
    pdist = t.nn.PairwiseDistance(p=2)
    dist = pdist(center,neighbour)
    return t.exp((-(dist**2))/sigma)

def normalize(condProb):
    for i in range(condProb.shape[0]):
        condProb[i] = condProb[i]/t.sum(condProb[i])
    return condProb

def computeProbaMatrix(data):
    N = data.shape[0]
    condProb = t.empty((N,N))
    for i in range(N):
        for j in range(N):
            if i == j:
                condProb[i][j] = 0
            else:
                condProb[i][j] = computeProba(data[i],data[j],sigmas[i])
    return normalize(condProb)

m = computeProbaMatrix(data)
print(t.sum(m[1]))


