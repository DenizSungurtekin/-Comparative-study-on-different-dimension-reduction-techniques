import torch as t
import math as m
pdist = t.nn.PairwiseDistance(p=2)

def computeProba(center,neighbour,sigma):
    dist = pdist(center,neighbour)
    return t.exp((-(dist**2))/sigma) #Cause bcp de NaN value attention

def computeProbaEmbeed(center,neighbour):
    dist = pdist(center,neighbour)
    return 1/(1+(dist**2))

def normalize(condProb):
    for i in range(condProb.shape[0]):
        condProb[i] = condProb[i]/t.sum(condProb[i])
    return condProb

def computeProbaMatrix(data,sigmas):
    N = data.shape[0]
    condProb = t.empty((N,N))
    for i in range(N):
        for j in range(N):
            if i == j:
                condProb[i][j] = 0
            else:
                condProb[i][j] = computeProba(data[i],data[j],sigmas[i])
    return normalize(condProb)

def computeEmbeedProbaMatrix(embed):
    N = embed.shape[0]
    condProb = t.empty((N,N))
    for i in range(N):
        for j in range(N):
            if i == j:
                condProb[i][j] = 0
            else:
                condProb[i][j] = computeProbaEmbeed(embed[i],embed[j])
    return normalize(condProb)

def computeGrad(embeed_data,probMatrix,probEmbeed):
    N,M = embeed_data.shape
    gradients = t.full((N,M),0.)
    for i in range(N):
        grad = t.full((1,M),0.)
        for j in range(N):
            if i==j:
                continue
            else:
                dist = pdist(embeed_data[i], embeed_data[j])
                res = probMatrix[i][j]-probEmbeed[i][j]*(embeed_data[i]-embeed_data[j])*(1/(1+(dist**2)))
                grad += res
        grad = 4.*grad
        gradients[i] = grad
    return gradients

def update(gradients,embed,lr): #Without momentum ATM
    embed = embed - lr*gradients
    return embed

def reduction(data,embed_size,sigmas,lr=0.1,epochs = 20):
    N = data.shape[0]
    embed = t.normal(mean=0, std=10, size=(N, embed_size)) # y initialization with gaussian from vandermaaten paper (std = 0.0001 too small ?)
    print(embed)
    for i in range(epochs):
        embed = update(computeGrad(embed,computeProbaMatrix(data,sigmas),computeEmbeedProbaMatrix(embed)),embed,lr)
    return embed

N = 100
M = 25
data = t.rand((N,M))
sigmas = t.rand((N,1))+1 #Randomly generated: plus sgd grd plus perplexite et entropie grande -> cmt bien definir cette mesure?
embed_size = 2
epochs = 50

embed = reduction(data,2,sigmas)

print(embed)
