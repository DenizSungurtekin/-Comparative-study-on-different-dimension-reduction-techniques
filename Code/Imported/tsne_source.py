import numpy as np
import matplotlib
from sklearn.manifold import TSNE
from Dataset import generations as gen
import matplotlib.pyplot as plt

data = gen.genRandomDataUniform(200,15,0,10)
labels = gen.genRandLabels(200,0,1)

X_embedded = TSNE(n_components=2,init='random').fit_transform(data)
colors = ['red','green']

plt.scatter(X_embedded[:,0], X_embedded[:,1], c=labels, cmap=matplotlib.colors.ListedColormap(colors))

cb = plt.colorbar()
loc = np.arange(0,max(labels),max(labels)/float(len(colors)))
cb.set_ticks(loc)
cb.set_ticklabels(["Class 0","Class 1"])
plt.title("Tsne")
plt.show()

