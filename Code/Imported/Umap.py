import sklearn.datasets
import umap.plot


data, labels = sklearn.datasets.fetch_openml(
    'mnist_784', version=1, return_X_y=True
)

print(data.shape)


mapper1 = umap.UMAP().fit(data)
p = umap.plot.points(mapper1, labels=labels)
umap.plot.plt.show()
