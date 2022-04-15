from Code.Implementations.Tsne import tsne
from Code.Analyse_plot import func_tools
from Code.Implementations.DataHandler import dataReader
import numpy as np
import matplotlib.pyplot as plt
import mpl_interactions.ipyplot as iplt
import matplotlib.image as mpimg


# Load data MNIST
size = 100 # The first size element of MNIST
data = dataReader.loadMnist(size)

## TSNE
PERPLEXITY = 5
TARGET_DIMENSION = 2

mu_noise = 0

modes = ["random","gaussian","pca","laplace"]
colors = ["blue","red","orange","black"]
epochs = 200

# for el,color in zip(modes,colors): ## PLOT WITH EACH INITIALIZIATION
#
#     Y,loss = tsne.tsne_reduction(data,TARGET_DIMENSION,PERPLEXITY,epochs = epochs,initialization = el,lr=0.01)
#     title = "tsne with initialization: " + str(el)
#     func_tools.scatter2d(Y,title=title,color = color,label=str(el))
#     plt.title(title)
#     save_name = el
#     func_tools.save_plot_tsne(save_name)
#     plt.show()


# stds = [1,5,10,20]
# for el,color in zip(modes,colors): ## PLOT WITH EACH INITIALIZIATION and different noise
#     for std in stds:
#         data2 = data
#         noise = np.random.normal(mu_noise, std, size=data.shape).astype(np.uint8)
#         data2 += noise
#         Y,loss = tsne.tsne_reduction(data2,TARGET_DIMENSION,PERPLEXITY,epochs = epochs,initialization = el,lr=0.01)
#         title = "tsne with initialization: " + str(el) + " sigma="+str(std)
#         func_tools.scatter2d(Y,title=title,color = color,label=str(el))
#         plt.title(title)
#         save_name = el+"std"+str(std)
#         func_tools.save_plot_tsne(save_name)
#         plt.show()

# perplexities = [5,15,25,50]
# for el,color in zip(modes,colors): ## PLOT WITH EACH INITIALIZIATION and different Perplexity
#     for perplexity in perplexities:
#         Y,loss = tsne.tsne_reduction(data,TARGET_DIMENSION,perplexity,epochs = epochs,initialization = el,lr=0.01)
#         title = "tsne with initialization: " + str(el) + " perplexity= "+str(perplexity)
#         func_tools.scatter2d(Y,title=title,color = color,label=str(el))
#         plt.title(title)
#         save_name = el+"perplex"+str(perplexity)
#         func_tools.save_plot_tsne(save_name)
#         plt.show()


# for el,color in zip(modes,colors): ## PLOT LOSS WITH EACH INITIALIZIATION
#
#     Y,loss = tsne.tsne_reduction(data,TARGET_DIMENSION,PERPLEXITY,epochs = epochs,initialization = el,lr=0.001)
#     print(loss)
#     title = "tnsne KL loss with initialization: " + str(el)+" and learning rate= 0.01"
#     plt.plot(loss)
#     plt.title(title)
#     save_name = "loss"+el
#     func_tools.save_plot_tsne(save_name)
#     plt.show()
print(data)
Y= tsne.tsne_reduction(data,TARGET_DIMENSION,PERPLEXITY,lr=0.001)







