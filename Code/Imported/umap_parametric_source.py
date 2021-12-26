from umap.parametric_umap import ParametricUMAP
from Dataset import generations as gen
import umap.plot

data = gen.genRandomDataUniform(200,15,0,10)
labels = gen.genRandLabels(200,0,1)

embedder = ParametricUMAP()
wrap= embedder.fit(data)

p = umap.plot.points(wrap, labels=labels)
umap.plot.plt.show()