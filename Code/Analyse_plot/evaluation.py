from Code.Implementations.Tsne import tsne
from Code.Implementations.Umap import umap
from Code.Analyse_plot import func_tools
from Code.Implementations.DataHandler import generations
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from Code.Implementations.DataHandler import evaluations
from Code.Analyse_plot.main_tsne import checkStability_tsne
from Code.Analyse_plot.main_umap import checkStability_umap

def plot(res):
    colors = ['b', 'c', 'y', 'm']
    markers = ["o", "x","s","*"]
    y = res[:, 0]

    labels = res[:, 1]
    x = [i for i in range(len(y))]

    size = len(y)
    indexs = [i for i in range(size)]
    indexs2 = np.asarray([[i] * len(colors) for i in range(int(size / 2))])
    indexs2 = indexs2.flatten()
    for i in indexs:
        plt.scatter(x[i], y[i], color=colors[indexs2[i]], marker=markers[i % len(markers)], label=labels[i])

    plt.legend(loc='upper left', ncol=4, fontsize=6)
    plt.show()

### Global methodology of analyse

## Generation of datasets
size = 500
features = 10

swiss_roll,color_swiss = generations.make_swiss_roll(size) # Swiss roll
non_cluster = generations.genRandomDataUniform(size,features,-5,5) #Not clustered value between -5 and 5
clustered = generations.makeCluster(size=size,features=features) #Clustered data

datas = [swiss_roll,non_cluster,clustered]

## Plot for stability score 0-1

# # # Plot for each configuration
# for data in datas:
#     res = checkStability_tsne(data = data,epochs=500)
#     plot(res)
#     res = checkStability_umap(data = data,epochs=500)
#     plot(res)


# res = checkStability_umap(data = datas[0],epochs=10)
# plot(res)


## Plot evaluation measure


## Swiss roll

# perplexities = [5,10,15,20,25,30,35,40,45,50]
# # perplexities = [5,10]
# min_dist = 0.01
#
# stress_values = []
# stress_values_u = []
#
# spearmans = []
# spearmans_u = []
#
# konigs_4_10 = []
# konigs_4_300 = []
# konigs_4_450 = []
# konigs_4_10_u = []
# konigs_4_300_u = []
# konigs_4_450_u = []
#
# konigs_2_10 = []
# konigs_3_10 = []
# konigs_5_10 = []
# konigs_2_10_u = []
# konigs_3_10_u = []
# konigs_5_10_u = []
#
# mrre_X_5 = []
# mrre_Y_5 = []
# mrre_X_5_u = []
# mrre_Y_5_u = []
#
# mrre_X_10 = []
# mrre_Y_10 = []
# mrre_X_10_u = []
# mrre_Y_10_u = []
#
# mrre_X_20 = []
# mrre_Y_20 = []
# mrre_X_20_u = []
# mrre_Y_20_u = []
#
# mrre_X_50 = []
# mrre_Y_50 = []
# mrre_X_50_u = []
# mrre_Y_50_u = []
#
# for perplexity in perplexities:
#     Y,_,_ = tsne.tsne_reduction(swiss_roll, 2, perplexity, epochs=500, initialization="pca", lr=100)
#     Y2,_,_ = umap.umap_reduction(swiss_roll, min_dist, 2, perplexity, epochs=500, initialization="random", lr=0.02,SGD=False)
#     stress_values.append(evaluations.stress(swiss_roll,Y))
#     stress_values_u.append(evaluations.stress(swiss_roll,Y2))
#     spearmans.append(evaluations.spearman(swiss_roll,Y))
#     spearmans_u.append(evaluations.spearman(swiss_roll,Y2))
#
#     konigs_4_10.append(evaluations.konig(swiss_roll,Y,4,10))
#     konigs_4_300.append(evaluations.konig(swiss_roll,Y,4,300))
#     konigs_4_450.append(evaluations.konig(swiss_roll,Y,4,450))
#     konigs_4_10_u.append(evaluations.konig(swiss_roll,Y2,4,10))
#     konigs_4_300_u.append(evaluations.konig(swiss_roll,Y2,4,300))
#     konigs_4_450_u.append(evaluations.konig(swiss_roll,Y2,4,450))
#
#     konigs_2_10.append(evaluations.konig(swiss_roll,Y,2,10))
#     konigs_3_10.append(evaluations.konig(swiss_roll,Y,3,10))
#     konigs_5_10.append(evaluations.konig(swiss_roll,Y,5,10))
#     konigs_2_10_u.append(evaluations.konig(swiss_roll,Y2,2,10))
#     konigs_3_10_u.append(evaluations.konig(swiss_roll,Y2,3,10))
#     konigs_5_10_u.append(evaluations.konig(swiss_roll,Y2,5,10))
#
#     score_XY = evaluations.meanRelativeError(swiss_roll,Y,5)
#     mrre_X_5.append(score_XY[0])
#     mrre_Y_5.append(score_XY[1])
#     score_XY_u = evaluations.meanRelativeError(swiss_roll,Y2,5)
#     mrre_X_5_u.append(score_XY_u[0])
#     mrre_Y_5_u.append(score_XY_u[1])
#
#     score_XY = evaluations.meanRelativeError(swiss_roll, Y, 10)
#     mrre_X_10.append(score_XY[0])
#     mrre_Y_10.append(score_XY[1])
#     score_XY_u = evaluations.meanRelativeError(swiss_roll, Y2, 10)
#     mrre_X_10_u.append(score_XY_u[0])
#     mrre_Y_10_u.append(score_XY_u[1])
#
#     score_XY = evaluations.meanRelativeError(swiss_roll, Y, 20)
#     mrre_X_20.append(score_XY[0])
#     mrre_Y_20.append(score_XY[1])
#     score_XY_u = evaluations.meanRelativeError(swiss_roll, Y2, 20)
#     mrre_X_20_u.append(score_XY_u[0])
#     mrre_Y_20_u.append(score_XY_u[1])
#
#     score_XY = evaluations.meanRelativeError(swiss_roll,Y,50)
#     mrre_X_50.append(score_XY[0])
#     mrre_Y_50.append(score_XY[1])
#     score_XY_u = evaluations.meanRelativeError(swiss_roll,Y2,50)
#     mrre_X_50_u.append(score_XY_u[0])
#     mrre_Y_50_u.append(score_XY_u[1])
#
#
# x = perplexities
#
# #first plot
# plt.plot(x,stress_values,label = "t-SNE")
# plt.plot(x,stress_values_u,label = "uMap")
# plt.title("Evolution of stress w.r.t perplexity or k")
# plt.xlabel("Perplexity/k")
# plt.ylabel("Measures")
# plt.legend(loc='upper left', ncol=2, fontsize=8)
# plt.show()
#
# plt.plot(x,spearmans,label = "t-SNE")
# plt.plot(x,spearmans_u,label = "uMap")
# plt.title("Evolution of spearman rho w.r.t perplexity or k")
# plt.xlabel("Perplexity/k")
# plt.ylabel("Measures")
# plt.legend(loc='upper left', ncol=2, fontsize=8)
# plt.show()
#
# #second plot -> variance of k2
# plt.plot(x,konigs_4_10,color = "blue",label = "k1=4,k2=10")
# plt.plot(x,konigs_4_300,color = "red",label = "k1=4,k2=300")
# plt.plot(x,konigs_4_450,color = "green",label = "k1=4,k2=450")
# plt.title("Evolution of konig measure w.r.t perplexity and k2 - t-SNE")
# plt.xlabel("Perplexity")
# plt.ylabel("Konig")
# plt.legend(loc='upper left', ncol=2, fontsize=6)
# plt.show()
# #second plot -> variance of k2
# plt.plot(x,konigs_4_10_u,color = "blue",label = "k1=4,k2=10")
# plt.plot(x,konigs_4_300_u,color = "red",label = "k1=4,k2=300")
# plt.plot(x,konigs_4_450_u,color = "green",label = "k1=4,k2=450")
# plt.title("Evolution of konig measure w.r.t k and k2 - uMap")
# plt.xlabel("k")
# plt.ylabel("Konig")
# plt.legend(loc='upper left', ncol=2, fontsize=6)
# plt.show()
#
# #third plot -> variance of k1
# plt.plot(x,konigs_2_10,color = "blue",label = "k1=2,k2=10")
# plt.plot(x,konigs_3_10,color = "red",label = "k1=3,k2=10")
# plt.plot(x,konigs_4_10,color = "green",label = "k1=4,k2=10")
# plt.plot(x,konigs_5_10,color = "black",label = "k1=5,k2=10")
# plt.title("Evolution of konig measure w.r.t perplexity and k1 - t-SNE")
# plt.xlabel("Perplexity")
# plt.ylabel("Konig")
# plt.legend(loc='upper left', ncol=2, fontsize=6)
# plt.show()
# #third plot -> variance of k1
# plt.plot(x,konigs_2_10_u,color = "blue",label = "k1=2,k2=10")
# plt.plot(x,konigs_3_10_u,color = "red",label = "k1=3,k2=10")
# plt.plot(x,konigs_4_10_u,color = "green",label = "k1=4,k2=10")
# plt.plot(x,konigs_5_10_u,color = "black",label = "k1=5,k2=10")
# plt.title("Evolution of konig measure w.r.t k and k1 - uMap")
# plt.xlabel("k")
# plt.ylabel("Konig")
# plt.legend(loc='upper left', ncol=2, fontsize=6)
# plt.show()
#
# #fourth plot -> mrreX
# plt.plot(x,mrre_X_5,color = "blue",label = "K = 5")
# plt.plot(x,mrre_X_10,color = "red",label = "K = 10")
# plt.plot(x,mrre_X_20,color = "green",label = "K = 20")
# plt.plot(x,mrre_X_50,color = "black",label = "K = 50")
# plt.title("Evolution of mean relative error X w.r.t perplexity - t-SNE")
# plt.xlabel("Perplexity")
# plt.ylabel("Mrre")
# plt.legend(loc='upper left', ncol=2, fontsize=6)
# plt.show()
# #fourth plot -> mrreX
# plt.plot(x,mrre_X_5_u,color = "blue",label = "K = 5")
# plt.plot(x,mrre_X_10_u,color = "red",label = "K = 10")
# plt.plot(x,mrre_X_20_u,color = "green",label = "K = 20")
# plt.plot(x,mrre_X_50_u,color = "black",label = "K = 50")
# plt.title("Evolution of mean relative error X w.r.t k - uMap")
# plt.xlabel("k")
# plt.ylabel("Mrre")
# plt.legend(loc='upper left', ncol=2, fontsize=6)
# plt.show()
#
# #fifth plot -> mrreY
# plt.plot(x,mrre_Y_5,color = "blue",label = "K = 5")
# plt.plot(x,mrre_Y_10,color = "red",label = "K = 10")
# plt.plot(x,mrre_Y_20,color = "green",label = "K = 20")
# plt.plot(x,mrre_Y_50,color = "black",label = "K = 50")
# plt.title("Evolution of mean relative error Y w.r.t perplexity - t-SNE")
# plt.xlabel("Perplexity")
# plt.ylabel("Mrre")
# plt.legend(loc='upper left', ncol=2, fontsize=6)
# plt.show()
# #fifth plot -> mrreY umap
# plt.plot(x,mrre_Y_5_u,color = "blue",label = "K = 5")
# plt.plot(x,mrre_Y_10_u,color = "red",label = "K = 10")
# plt.plot(x,mrre_Y_20_u,color = "green",label = "K = 20")
# plt.plot(x,mrre_Y_50_u,color = "black",label = "K = 50")
# plt.title("Evolution of mean relative error Y w.r.t k - uMap")
# plt.xlabel("k")
# plt.ylabel("Mrre")
# plt.legend(loc='upper left', ncol=2, fontsize=6)
# plt.show()


## Clustered and non clustered Data measure

# features = [4,5,6,7,8,9,10,15,20,25]
# k = 20
# min_dist = 0.1
# perplexity = 25
#
# stress_cluster = []
# stress_cluster_u = []
#
# stress_noncluster = []
# stress_noncluster_u = []
#
# spearman_cluster = []
# spearman_cluster_u = []
#
# spearman_noncluster = []
# spearman_noncluster_u = []
#
# konigs_4_10_cluster = []
# konigs_4_300_cluster = []
# konigs_4_450_cluster = []
# konigs_4_10_noncluster = []
# konigs_4_300_noncluster = []
# konigs_4_450_noncluster = []
#
# konigs_4_10_u_cluster = []
# konigs_4_300_u_cluster = []
# konigs_4_450_u_cluster = []
# konigs_4_10_u_noncluster = []
# konigs_4_300_u_noncluster = []
# konigs_4_450_u_noncluster = []
#
# konigs_2_10_cluster = []
# konigs_3_10_cluster = []
# konigs_5_10_cluster = []
# konigs_2_10_noncluster = []
# konigs_3_10_noncluster = []
# konigs_5_10_noncluster = []
#
# konigs_2_10_u_cluster = []
# konigs_3_10_u_cluster = []
# konigs_5_10_u_cluster = []
# konigs_2_10_u_noncluster = []
# konigs_3_10_u_noncluster = []
# konigs_5_10_u_noncluster = []
#
# mrre_X_5_cluster = []
# mrre_Y_5_cluster = []
# mrre_X_5_u_cluster = []
# mrre_Y_5_u_cluster = []
#
# mrre_X_10_cluster = []
# mrre_Y_10_cluster = []
# mrre_X_10_u_cluster = []
# mrre_Y_10_u_cluster = []
#
# mrre_X_20_cluster = []
# mrre_Y_20_cluster = []
# mrre_X_20_u_cluster = []
# mrre_Y_20_u_cluster = []
#
# mrre_X_50_cluster = []
# mrre_Y_50_cluster = []
# mrre_X_50_u_cluster = []
# mrre_Y_50_u_cluster = []
#
# mrre_X_5_noncluster = []
# mrre_Y_5_noncluster = []
# mrre_X_5_u_noncluster = []
# mrre_Y_5_u_noncluster = []
#
# mrre_X_10_noncluster = []
# mrre_Y_10_noncluster = []
# mrre_X_10_u_noncluster = []
# mrre_Y_10_u_noncluster = []
#
# mrre_X_20_noncluster = []
# mrre_Y_20_noncluster = []
# mrre_X_20_u_noncluster = []
# mrre_Y_20_u_noncluster = []
#
# mrre_X_50_noncluster = []
# mrre_Y_50_noncluster = []
# mrre_X_50_u_noncluster = []
# mrre_Y_50_u_noncluster = []
#
#
# for feature in features:
#     data = generations.makeCluster(size=size, features=feature)  # Clustered data
#     Y,_,_ = tsne.tsne_reduction(data, 2, perplexity, epochs=500, initialization="pca", lr=100)
#     Y2,_,_ = umap.umap_reduction(data, min_dist, 2, k, epochs=500, initialization="random", lr=0.02,SGD=False)
#
#     stress_cluster.append(evaluations.stress(data,Y))
#     stress_cluster_u.append(evaluations.stress(data, Y2))
#
#     spearman_cluster.append(evaluations.spearman(data,Y))
#     spearman_cluster_u.append(evaluations.spearman(data,Y2))
#
#     konigs_4_10_cluster.append(evaluations.konig(data,Y,4,10))
#     konigs_4_300_cluster.append(evaluations.konig(data,Y,4,300))
#     konigs_4_450_cluster.append(evaluations.konig(data,Y,4,450))
#
#     konigs_4_10_u_cluster.append(evaluations.konig(data,Y2,4,10))
#     konigs_4_300_u_cluster.append(evaluations.konig(data,Y2,4,300))
#     konigs_4_450_u_cluster.append(evaluations.konig(data,Y2,4,450))
#
#     konigs_2_10_cluster.append(evaluations.konig(data,Y,2,10))
#     konigs_3_10_cluster.append(evaluations.konig(data,Y,3,10))
#     konigs_5_10_cluster.append(evaluations.konig(data,Y,5,10))
#
#     konigs_2_10_u_cluster.append(evaluations.konig(data,Y2,2,10))
#     konigs_3_10_u_cluster.append(evaluations.konig(data,Y2,3,10))
#     konigs_5_10_u_cluster.append(evaluations.konig(data,Y2,5,10))
#
#     score_XY = evaluations.meanRelativeError(data,Y,5)
#     mrre_X_5_cluster.append(score_XY[0])
#     mrre_Y_5_cluster.append(score_XY[1])
#     score_XY_u = evaluations.meanRelativeError(data,Y2,5)
#     mrre_X_5_u_cluster.append(score_XY_u[0])
#     mrre_Y_5_u_cluster.append(score_XY_u[1])
#
#     score_XY = evaluations.meanRelativeError(data, Y, 10)
#     mrre_X_10_cluster.append(score_XY[0])
#     mrre_Y_10_cluster.append(score_XY[1])
#     score_XY_u = evaluations.meanRelativeError(data, Y2, 10)
#     mrre_X_10_u_cluster.append(score_XY_u[0])
#     mrre_Y_10_u_cluster.append(score_XY_u[1])
#
#     score_XY = evaluations.meanRelativeError(data, Y, 20)
#     mrre_X_20_cluster.append(score_XY[0])
#     mrre_Y_20_cluster.append(score_XY[1])
#     score_XY_u = evaluations.meanRelativeError(data, Y2, 20)
#     mrre_X_20_u_cluster.append(score_XY_u[0])
#     mrre_Y_20_u_cluster.append(score_XY_u[1])
#
#     score_XY = evaluations.meanRelativeError(data,Y,50)
#     mrre_X_50_cluster.append(score_XY[0])
#     mrre_Y_50_cluster.append(score_XY[1])
#     score_XY_u = evaluations.meanRelativeError(data,Y2,50)
#     mrre_X_50_u_cluster.append(score_XY_u[0])
#     mrre_Y_50_u_cluster.append(score_XY_u[1])
#
# x = features
#
# #first plot
# plt.plot(x,stress_cluster,label = "t-SNE")
# plt.plot(x,stress_cluster_u,label = "uMap")
# plt.title("Evolution of stress w.r.t initial dimension - Clustered data")
# plt.xlabel("Dimensions")
# plt.ylabel("Measures")
# plt.legend(loc='upper left', ncol=2, fontsize=8)
# plt.show()
#
# plt.plot(x,spearman_cluster,label = "t-SNE")
# plt.plot(x,spearman_cluster_u,label = "uMap")
# plt.title("Evolution of spearman rho w.r.t dimension - Clustered data")
# plt.xlabel("Dimensions")
# plt.ylabel("Measures")
# plt.legend(loc='upper left', ncol=2, fontsize=8)
# plt.show()
#
# #second plot -> variance of k2
# plt.plot(x,konigs_4_10_cluster,color = "blue",label = "k1=4,k2=10")
# plt.plot(x,konigs_4_450_cluster,color = "red",label = "k1=4,k2=300")
# plt.plot(x,konigs_4_450_cluster,color = "green",label = "k1=4,k2=450")
# plt.title("Evolution of konig measure w.r.t Dimensionality and k2 - t-SNE - Clustered data")
# plt.xlabel("Dimensionality")
# plt.ylabel("Konig")
# plt.legend(loc='upper left', ncol=2, fontsize=6)
# plt.show()
# #second plot -> variance of k2
# plt.plot(x,konigs_4_10_u_cluster,color = "blue",label = "k1=4,k2=10")
# plt.plot(x,konigs_4_300_u_cluster,color = "red",label = "k1=4,k2=300")
# plt.plot(x,konigs_4_450_u_cluster,color = "green",label = "k1=4,k2=450")
# plt.title("Evolution of konig measure w.r.t Dimensionality and k2 - uMap - Clustered data")
# plt.xlabel("Dimensions")
# plt.ylabel("Konig")
# plt.legend(loc='upper left', ncol=2, fontsize=6)
# plt.show()
#
# #third plot -> variance of k1
# plt.plot(x,konigs_2_10_cluster,color = "blue",label = "k1=2,k2=10")
# plt.plot(x,konigs_3_10_cluster,color = "red",label = "k1=3,k2=10")
# plt.plot(x,konigs_4_10_cluster,color = "green",label = "k1=4,k2=10")
# plt.plot(x,konigs_5_10_cluster,color = "black",label = "k1=5,k2=10")
# plt.title("Evolution of konig measure w.r.t dimensionality and k1 - t-SNE - Clustered data")
# plt.xlabel("Dimensions")
# plt.ylabel("Konig")
# plt.legend(loc='upper left', ncol=2, fontsize=6)
# plt.show()
# #third plot -> variance of k1
# plt.plot(x,konigs_2_10_u_cluster,color = "blue",label = "k1=2,k2=10")
# plt.plot(x,konigs_3_10_u_cluster,color = "red",label = "k1=3,k2=10")
# plt.plot(x,konigs_4_10_u_cluster,color = "green",label = "k1=4,k2=10")
# plt.plot(x,konigs_5_10_u_cluster,color = "black",label = "k1=5,k2=10")
# plt.title("Evolution of konig measure w.r.t dimension and k1 - uMap - Clustered data")
# plt.xlabel("dimension")
# plt.ylabel("Konig")
# plt.legend(loc='upper left', ncol=2, fontsize=6)
# plt.show()
#
# #fourth plot -> mrreX
# plt.plot(x,mrre_X_5_cluster,color = "blue",label = "K = 5")
# plt.plot(x,mrre_X_10_cluster,color = "red",label = "K = 10")
# plt.plot(x,mrre_X_20_cluster,color = "green",label = "K = 20")
# plt.plot(x,mrre_X_50_cluster,color = "black",label = "K = 50")
# plt.title("Evolution of mean relative error X w.r.t dimensions - t-SNE - Clustered data")
# plt.xlabel("Dimensions")
# plt.ylabel("Mrre")
# plt.legend(loc='upper left', ncol=2, fontsize=6)
# plt.show()
# #fourth plot -> mrreX
# plt.plot(x,mrre_X_5_u_cluster,color = "blue",label = "K = 5")
# plt.plot(x,mrre_X_10_u_cluster,color = "red",label = "K = 10")
# plt.plot(x,mrre_X_20_u_cluster,color = "green",label = "K = 20")
# plt.plot(x,mrre_X_50_u_cluster,color = "black",label = "K = 50")
# plt.title("Evolution of mean relative error X w.r.t dimensions - uMap - Clustered data")
# plt.xlabel("Dimensions")
# plt.ylabel("Mrre")
# plt.legend(loc='upper left', ncol=2, fontsize=6)
# plt.show()
#
# #fifth plot -> mrreY
# plt.plot(x,mrre_Y_5_cluster,color = "blue",label = "K = 5")
# plt.plot(x,mrre_Y_10_cluster,color = "red",label = "K = 10")
# plt.plot(x,mrre_Y_20_cluster,color = "green",label = "K = 20")
# plt.plot(x,mrre_Y_50_cluster,color = "black",label = "K = 50")
# plt.title("Evolution of mean relative error Y w.r.t dimensions - t-SNE - Clustered data")
# plt.xlabel("Dimensions")
# plt.ylabel("Mrre")
# plt.legend(loc='upper left', ncol=2, fontsize=6)
# plt.show()
# #fifth plot -> mrreY umap
# plt.plot(x,mrre_Y_5_u_cluster,color = "blue",label = "K = 5")
# plt.plot(x,mrre_Y_10_u_cluster,color = "red",label = "K = 10")
# plt.plot(x,mrre_Y_20_u_cluster,color = "green",label = "K = 20")
# plt.plot(x,mrre_Y_50_u_cluster,color = "black",label = "K = 50")
# plt.title("Evolution of mean relative error Y w.r.t dimensions - uMap - Clustered data")
# plt.xlabel("Dimensions")
# plt.ylabel("Mrre")
# plt.legend(loc='upper left', ncol=2, fontsize=6)
# plt.show()
#
# ## Non clustered
# for feature in features:
#     data = generations.genRandomDataUniform(size, feature, -5, 5)  # Non clustered value between -5 and 5
#     Y, _, _ = tsne.tsne_reduction(data, 2, perplexity, epochs=500, initialization="pca", lr=100)
#     Y2, _, _ = umap.umap_reduction(data, min_dist, 2, perplexity, epochs=500, initialization="random", lr=0.02,SGD=False)
#
#     stress_noncluster.append(evaluations.stress(data, Y))
#     stress_noncluster_u.append(evaluations.stress(data, Y2))
#
#     spearman_noncluster.append(evaluations.spearman(data, Y))
#     spearman_noncluster_u.append(evaluations.spearman(data, Y2))
#
#     konigs_4_10_noncluster.append(evaluations.konig(data, Y, 4, 10))
#     konigs_4_300_noncluster.append(evaluations.konig(data, Y, 4, 300))
#     konigs_4_450_noncluster.append(evaluations.konig(data, Y, 4, 450))
#
#     konigs_4_10_u_noncluster.append(evaluations.konig(data, Y2, 4, 10))
#     konigs_4_300_u_noncluster.append(evaluations.konig(data, Y2, 4, 300))
#     konigs_4_450_u_noncluster.append(evaluations.konig(data, Y2, 4, 450))
#
#     konigs_2_10_noncluster.append(evaluations.konig(data, Y, 2, 10))
#     konigs_3_10_noncluster.append(evaluations.konig(data, Y, 3, 10))
#     konigs_5_10_noncluster.append(evaluations.konig(data, Y, 5, 10))
#
#     konigs_2_10_u_noncluster.append(evaluations.konig(data, Y2, 2, 10))
#     konigs_3_10_u_noncluster.append(evaluations.konig(data, Y2, 3, 10))
#     konigs_5_10_u_noncluster.append(evaluations.konig(data, Y2, 5, 10))
#
#     score_XY = evaluations.meanRelativeError(data, Y, 5)
#     mrre_X_5_noncluster.append(score_XY[0])
#     mrre_Y_5_noncluster.append(score_XY[1])
#     score_XY_u = evaluations.meanRelativeError(data, Y2, 5)
#     mrre_X_5_u_noncluster.append(score_XY_u[0])
#     mrre_Y_5_u_noncluster.append(score_XY_u[1])
#
#     score_XY = evaluations.meanRelativeError(data, Y, 10)
#     mrre_X_10_noncluster.append(score_XY[0])
#     mrre_Y_10_noncluster.append(score_XY[1])
#     score_XY_u = evaluations.meanRelativeError(data, Y2, 10)
#     mrre_X_10_u_noncluster.append(score_XY_u[0])
#     mrre_Y_10_u_noncluster.append(score_XY_u[1])
#
#     score_XY = evaluations.meanRelativeError(data, Y, 20)
#     mrre_X_20_noncluster.append(score_XY[0])
#     mrre_Y_20_noncluster.append(score_XY[1])
#     score_XY_u = evaluations.meanRelativeError(data, Y2, 20)
#     mrre_X_20_u_noncluster.append(score_XY_u[0])
#     mrre_Y_20_u_noncluster.append(score_XY_u[1])
#
#     score_XY = evaluations.meanRelativeError(data, Y, 50)
#     mrre_X_50_noncluster.append(score_XY[0])
#     mrre_Y_50_noncluster.append(score_XY[1])
#     score_XY_u = evaluations.meanRelativeError(data, Y2, 50)
#     mrre_X_50_u_noncluster.append(score_XY_u[0])
#     mrre_Y_50_u_noncluster.append(score_XY_u[1])
#
# x = features
#
# # first plot
# plt.plot(x, stress_noncluster, label="t-SNE")
# plt.plot(x, stress_noncluster_u, label="uMap")
# plt.title("Evolution of stress w.r.t initial dimension - Non-clustered data")
# plt.xlabel("Dimensions")
# plt.ylabel("Measures")
# plt.legend(loc='upper left', ncol=2, fontsize=8)
# plt.show()
#
# plt.plot(x, spearman_noncluster, label="t-SNE")
# plt.plot(x, spearman_noncluster_u, label="uMap")
# plt.title("Evolution of spearman rho w.r.t dimension - Non-clustered data")
# plt.xlabel("Dimensions")
# plt.ylabel("Measures")
# plt.legend(loc='upper left', ncol=2, fontsize=8)
# plt.show()
#
# # second plot -> variance of k2
# plt.plot(x, konigs_4_10_noncluster, color="blue", label="k1=4,k2=10")
# plt.plot(x, konigs_4_450_noncluster, color="red", label="k1=4,k2=300")
# plt.plot(x, konigs_4_450_noncluster, color="green", label="k1=4,k2=450")
# plt.title("Evolution of konig measure w.r.t Dimensionality and k2 - t-SNE - Non-clustered data")
# plt.xlabel("Dimensionality")
# plt.ylabel("Konig")
# plt.legend(loc='upper left', ncol=2, fontsize=6)
# plt.show()
# # second plot -> variance of k2
# plt.plot(x, konigs_4_10_u_noncluster, color="blue", label="k1=4,k2=10")
# plt.plot(x, konigs_4_300_u_noncluster, color="red", label="k1=4,k2=300")
# plt.plot(x, konigs_4_450_u_noncluster, color="green", label="k1=4,k2=450")
# plt.title("Evolution of konig measure w.r.t Dimensionality and k2 - uMap - Non-clustered data")
# plt.xlabel("Dimensions")
# plt.ylabel("Konig")
# plt.legend(loc='upper left', ncol=2, fontsize=6)
# plt.show()
#
# # third plot -> variance of k1
# plt.plot(x, konigs_2_10_noncluster, color="blue", label="k1=2,k2=10")
# plt.plot(x, konigs_3_10_noncluster, color="red", label="k1=3,k2=10")
# plt.plot(x, konigs_4_10_noncluster, color="green", label="k1=4,k2=10")
# plt.plot(x, konigs_5_10_noncluster, color="black", label="k1=5,k2=10")
# plt.title("Evolution of konig measure w.r.t dimensionality and k1 - t-SNE - Non-clustered data")
# plt.xlabel("Dimensions")
# plt.ylabel("Konig")
# plt.legend(loc='upper left', ncol=2, fontsize=6)
# plt.show()
# # third plot -> variance of k1
# plt.plot(x, konigs_2_10_u_noncluster, color="blue", label="k1=2,k2=10")
# plt.plot(x, konigs_3_10_u_noncluster, color="red", label="k1=3,k2=10")
# plt.plot(x, konigs_4_10_u_noncluster, color="green", label="k1=4,k2=10")
# plt.plot(x, konigs_5_10_u_noncluster, color="black", label="k1=5,k2=10")
# plt.title("Evolution of konig measure w.r.t dimension and k1 - uMap - Non-clustered data")
# plt.xlabel("dimension")
# plt.ylabel("Konig")
# plt.legend(loc='upper left', ncol=2, fontsize=6)
# plt.show()
#
# # fourth plot -> mrreX
# plt.plot(x, mrre_X_5_noncluster, color="blue", label="K = 5")
# plt.plot(x, mrre_X_10_noncluster, color="red", label="K = 10")
# plt.plot(x, mrre_X_20_noncluster, color="green", label="K = 20")
# plt.plot(x, mrre_X_50_noncluster, color="black", label="K = 50")
# plt.title("Evolution of mean relative error X w.r.t dimensions - t-SNE - Non-clustered data")
# plt.xlabel("Dimensions")
# plt.ylabel("Mrre")
# plt.legend(loc='upper left', ncol=2, fontsize=6)
# plt.show()
# # fourth plot -> mrreX
# plt.plot(x, mrre_X_5_u_noncluster, color="blue", label="K = 5")
# plt.plot(x, mrre_X_10_u_noncluster, color="red", label="K = 10")
# plt.plot(x, mrre_X_20_u_noncluster, color="green", label="K = 20")
# plt.plot(x, mrre_X_50_u_noncluster, color="black", label="K = 50")
# plt.title("Evolution of mean relative error X w.r.t dimensions - uMap - Non-clustered data")
# plt.xlabel("Dimensions")
# plt.ylabel("Mrre")
# plt.legend(loc='upper left', ncol=2, fontsize=6)
# plt.show()
#
# # fifth plot -> mrreY
# plt.plot(x, mrre_Y_5_noncluster, color="blue", label="K = 5")
# plt.plot(x, mrre_Y_10_noncluster, color="red", label="K = 10")
# plt.plot(x, mrre_Y_20_noncluster, color="green", label="K = 20")
# plt.plot(x, mrre_Y_50_noncluster, color="black", label="K = 50")
# plt.title("Evolution of mean relative error Y w.r.t dimensions - t-SNE - Non-clustered data")
# plt.xlabel("Dimensions")
# plt.ylabel("Mrre")
# plt.legend(loc='upper left', ncol=2, fontsize=6)
# plt.show()
# # fifth plot -> mrreY umap
# plt.plot(x, mrre_Y_5_u_noncluster, color="blue", label="K = 5")
# plt.plot(x, mrre_Y_10_u_noncluster, color="red", label="K = 10")
# plt.plot(x, mrre_Y_20_u_noncluster, color="green", label="K = 20")
# plt.plot(x, mrre_Y_50_u_noncluster, color="black", label="K = 50")
# plt.title("Evolution of mean relative error Y w.r.t dimensions - uMap - Non-clustered data")
# plt.xlabel("Dimensions")
# plt.ylabel("Mrre")
# plt.legend(loc='upper left', ncol=2, fontsize=6)
# plt.show()
#
#
