import matplotlib.pyplot as plt

def scatter2d(Y,title = "No title",color = ["blue"],label = "No label",plot = True): # data is a 2d matrix
    x = Y[0:, 0]
    y = Y[0:, 1]
    plt.scatter(x,y,c = color,label = label) # need to call plt.show() after using thise function
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title(title)
    if plot:
        plt.show()

def scatter3d(Y,title = "No title",color = ["blue"],label = "No label",plot = True): # data is a 3d matrix
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x = Y[0:, 0]
    y = Y[0:, 1]
    z = Y[0:, 2]
    ax.scatter3D(x,y,z,c = color,label = label) # need to call plt.show() after using thise function
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title(title)
    if plot:
        plt.show()

def compute_peak(p_conditional,threshold):
    counts = []
    for i in range(len(p_conditional)):
        count_neigh = 0
        for j in range(p_conditional.shape[1]):
            if p_conditional[i][j]>=threshold:
                count_neigh += 1
        counts.append(count_neigh)
    return counts

def plot_conditional_density(p_conditional,threshold,plot = True):
    x = [i for i in range(len(p_conditional))]
    counts = compute_peak(p_conditional,threshold)
    plt.bar(x,counts)
    plt.xlabel("x")
    plt.ylabel("Number of significant neighboors with pi|j >"+str(threshold))
    plt.title("Density of data")
    if plot:
        plt.show()

def shepard_diag(dist,dist_embeed,plot = True):
    x = dist.flatten()
    y = dist_embeed.flatten()
    plt.xlabel("Input distance")
    plt.ylabel("Output distance")
    plt.scatter(x,y,s=1)
    plt.title("Shepard diagram")
    if plot:
        plt.show()

def draw_dist_neighboorhood(p_conditional,index):
    x = [i for i in range(p_conditional.shape[1])]
    plt.plot(x, p_conditional[index])
    plt.title("Distribution of pi|j for i = "+str(index))
    plt.xlabel("Neighboor")
    plt.ylabel("pi|j")
    plt.show()

def save_plot_tsne(name):
    str = 'plots/tsne/' + name+ ".png"
    plt.savefig(str)

def save_plot_umap(name):
    str = 'plots/umap/' + name + ".png"
    plt.savefig(str)
