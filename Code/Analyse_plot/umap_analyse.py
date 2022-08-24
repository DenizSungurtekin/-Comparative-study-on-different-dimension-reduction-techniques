from Code.Implementations.Umap import umap
from Code.Analyse_plot import func_tools
from Code.Implementations.DataHandler import generations
from Code.Implementations.DataHandler import evaluations
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

## Generation of datasets
size = 200
features = 10
data1,color_swiss = generations.make_swiss_roll(size) # Swiss roll
data2 = generations.genRandomDataUniform(size,features,-5,5) #Not clustered
data3 = generations.makeCluster(size=size) #Clustered data

## Tsne reduction
data = data3   # Choose which data
epochs = 100
mode = "laplace"
lr = 0.02
min_dist = 0.1
k = 5
Y, loss, conditional_probas = umap.umap_reduction(data2,min_dist, 2, k, epochs=epochs, initialization=mode, lr=lr,SGD=False)


print("Stress value: ",evaluations.stress(data,Y)) # 0 to 1 -> 0 = perfect
print("Spearman rho value: ",evaluations.spearman(data,Y)) # -1 to 1, 1 = perfect

k1 = 2
k2 = 4
print("Konig's Measure with k1 = ",k1," and k2 = ",k2," :",evaluations.konig(data,Y,k1,k2)) # 0 to 1 -> 1 = perfect
score_x,score_y = evaluations.meanRelativeError(data,Y,2)
print("Mean relative error X: ",score_x,", Mean relative error Y: ",score_y)