from Dataset import generations as gen
import umap.plot

data = gen.genRandomDataUniform(200,15,0,10)
labels = gen.genRandLabels(200,0,1)

mapper1 = umap.UMAP().fit(data)
# print(mapper1.embedding_)
p = umap.plot.points(mapper1, labels=labels)
umap.plot.plt.show()