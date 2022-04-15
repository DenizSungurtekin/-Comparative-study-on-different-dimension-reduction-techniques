import matplotlib.pyplot as plt

def scatter2d(Y,title = "No title",color = "blue",label = "No label"):
    x = Y[0:, 0]
    y = Y[0:, 1]
    plt.scatter(x,y,color = color,label = label) # need to call plt.show() after using thise function
    plt.legend()
    plt.title(title)

def scatter3d(Y,title = "No title",color = "blue",label = "No label",first = True):
    if first:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    x = Y[0:, 0]
    y = Y[0:, 1]
    z = Y[0:, 2]
    ax.scatter3D(x,y,z,color = color,label = label) # need to call plt.show() after using thise function
    ax.legend()
    plt.title(title)

def save_plot_tsne(name):
    str = 'plots/tsne/' + name+ ".png"
    plt.savefig(str)

def save_plot_umap(name):
    str = 'plots/umap/' + name + ".png"
    plt.savefig(str)


## EXEMPLE INTERACTIVE PLOT
# x = np.linspace(0, np.pi, 100) ## EXEMPLE INTERACTIVE PLOT
# tau = np.linspace(0.5, 10, 100)
#
# def f1(x, tau, beta):
#     return np.sin(x * tau) * x * beta
# def f2(x, tau, beta):
#     return np.sin(x * beta) * x * tau
#
#
# fig, ax = plt.subplots()
# controls = iplt.plot(x, f1, tau=tau, beta=(1, 10, 100), label="f1")
# iplt.plot(x, f2, controls=controls, label="f2")
# _ = plt.legend()
# plt.show()